// SPDX-License-Identifier: MPL-2.0
// Copyright © 2023 Skyline Team and Contributors (https://github.com/skyline-emu/)

#pragma once

#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <common/base.h>
#include <common/spin_lock.h>
#include <gpu/tag_allocator.h>
#include <gpu/memory_manager.h>
#include "common.h"
#include "gpu/usage_tracker.h"
#include "guest_texture.h"

namespace skyline::gpu {
    namespace texture {
        enum class RenderPassUsage : u8 {
            None,
            Descriptor,
            RenderTarget
        };
    }

    class TextureUsageTracker;
    class TextureManager;
    struct TextureViewRequestInfo;
    class Texture;
    class HostTexture;
    struct GuestTexture;

    namespace interconnect {
        class CommandExecutor;
    }
    namespace interconnect::node {
        struct SyncNode;
    }

    struct TextureSyncRequestArgs {
        bool isRead;
        bool isWritten;
        vk::PipelineStageFlags usedStage;
        vk::AccessFlags usedFlags;
        vk::ImageLayout optimalLayout; // TODO: use this
    };

    /**
     * @brief A view into a specific subresource of a Texture
     * @note This class conforms to the Lockable and BasicLockable C++ named requirements
     */
    struct HostTextureView {
        Texture *texture; //!< The backing texture for this view, this is set to null when the host texture is destroyed
        HostTexture *hostTexture; //!< The backing host texture for this view, this is set to null when the host texture is destroyed
        bool stale{false}; //!< If the view is stale and should no longer be used in any future operations, this doesn't imply that the backing is destroyed
        vk::ImageViewType type;
        texture::Format format;
        vk::ComponentMapping components;
        vk::ImageSubresourceRange range;
        vk::raii::ImageView vkView; //!< The backing Vulkan image view for this view, this is destroyed with the texture

        HostTextureView(HostTexture *hostTexture, vk::ImageViewType type, texture::Format format, vk::ComponentMapping components, vk::ImageSubresourceRange range, vk::raii::ImageView &&vkView);

        /**
         * @brief Acquires an exclusive lock on the backing texture for the calling thread
         * @note Naming is in accordance to the BasicLockable named requirement
         */
        void lock();

        /**
         * @brief Acquires an exclusive lock on the texture for the calling thread
         * @param tag A tag to associate with the lock, future invocations with the same tag prior to the unlock will acquire the lock without waiting (0 is not a valid tag value and will disable tag behavior)
         * @return If the lock was acquired by this call rather than having the same tag as the holder
         * @note All locks using the same tag **must** be from the same thread as it'll only have one corresponding unlock() call
         */
        bool LockWithTag(ContextTag tag);

        /**
         * @brief Relinquishes an existing lock on the backing texture by the calling thread
         * @note Naming is in accordance to the BasicLockable named requirement
         */
        void unlock();

        /**
         * @brief Attempts to acquire an exclusive lock on the backing texture but returns immediately if it's captured by another thread
         * @note Naming is in accordance to the Lockable named requirement
         */
        bool try_lock();

        /**
         * @brief Performs ondemand synchronization of the texture, using an executor
         * @note If want to synchronize the texture without the executor you can use Texture::SynchronizeGuest or Texture::SynchronizeHost
         */
        void RequestSync(interconnect::CommandExecutor &executor, const TextureSyncRequestArgs &args);

        /**
         * @brief Performs ondemand synchronization of the texture, using an executor
         * @note If want to synchronize the texture without the executor you can use Texture::SynchronizeGuest or Texture::SynchronizeHost
         */
        void RequestRPSync(interconnect::CommandExecutor &executor, const TextureSyncRequestArgs &args, interconnect::node::SyncNode *syncNode);
    };

    class Texture;

    /**
     * @brief A texture which is backed by host constructs while being synchronized with the underlying guest texture
     * @note This class conforms to the Lockable and BasicLockable C++ named requirements
     */
    class HostTexture {
      private:
        Texture &texture;
        GuestTexture &guest;

        memory::Image backing; //!< The Vulkan image that backs this texture, it is nullable
        enum class DirtyState {
            Clean, //!< The CPU mappings are in sync with the GPU textures
            GuestDirty, //!< The CPU mappings have been modified but the GPU textures is not up to date
            OtherHostDirty, //!< Another `HostTexture` is dirty and this `HostTexture`/CPU mappings are not up to date
            HostDirty //!< This `HostTexture` is dirty and the other (may not be all) `HostTexture`s/CPU mappings are not up to date
        } dirtyState; //!< The state of the CPU mappings with respect to the GPU texture

        u32 lastRenderPassIndex{}; //!< The index of the last render pass that used this `HostTexture`
        texture::RenderPassUsage lastRenderPassUsage{texture::RenderPassUsage::None}; //!< The type of usage in the last render pass
        vk::ImageLayout layout;
        texture::Format format;
        std::vector<texture::MipLevelLayout> copyLayouts;
        u32 copySize;

        std::vector<HostTextureView *> views;

        friend TextureManager;
        friend Texture;
        friend HostTextureView;
        friend TextureUsageTracker;
        friend interconnect::CommandExecutor;

        /**
         * @brief An implementation function for guest -> host texture synchronization, it allocates and copies data into a staging buffer or directly into a linear host texture
         * @return If a staging buffer was required for the texture sync, it's returned filled with guest texture data and must be copied to the host texture by the callee
         */
        std::shared_ptr<memory::StagingBuffer> SynchronizeHostImpl();

        /**
         * @brief Records commands for copying data from a staging buffer to the texture's backing into the supplied command buffer
         */
        void CopyFromStagingBuffer(const vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<memory::StagingBuffer> &stagingBuffer);

        /**
         * @brief Records commands for copying data from the texture's backing to a staging buffer into the supplied command buffer
         * @note Any caller **must** ensure that the layout is not `eUndefined`
         */
        void CopyIntoStagingBuffer(const vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<memory::StagingBuffer> &stagingBuffer);

        /**
         * @brief Copies data from the supplied host buffer into the guest texture
         * @note The host buffer must be large enough to contain the entire image
         */
        void CopyToGuest(u8 *hostBuffer);

        /**
         * @return A vector of all the buffer image copies that need to be done for every aspect of every level of every layer of the texture
         */
        boost::container::small_vector<vk::BufferImageCopy, 10> GetBufferImageCopies();

        void RequestSync(interconnect::CommandExecutor &executor, const TextureSyncRequestArgs &args, vk::ImageSubresourceRange &viewRange, interconnect::node::SyncNode *syncNode);

      public:
        texture::Dimensions dimensions;
        vk::SampleCountFlagBits sampleCount;
        texture::Format guestFormat; //!< The format used by the guest, this will differ from `format` if the host doesn't support this
        bool needsDecompression; //!< If the guest format is compressed and needs to be decompressed before being used on the host
        vk::ImageType imageType;
        static constexpr vk::ImageTiling tiling{vk::ImageTiling::eOptimal}; //!< Code for linear tiled textures kept in case if it's useful later on
        vk::ImageCreateFlags flags;
        vk::ImageUsageFlags usage;

        struct ExecutorTrackingInfo {
            vk::PipelineStageFlags lastUsedStage{vk::PipelineStageFlagBits::eTopOfPipe};
            vk::AccessFlags lastUsedAccessFlag{};
            vk::PipelineStageFlags waitedStages{};
        } trackingInfo;

        bool usedInTP{};
        bool usedInRP{};

        static vk::ImageType ConvertViewType(vk::ImageViewType viewType, texture::Dimensions dimensions);

        /**
         * @brief Checks if the previous usage in the renderpass is compatible with the current one
         * @return If the new usage is compatible with the previous usage
         */
        bool ValidateRenderPassUsage(u32 renderPassIndex, texture::RenderPassUsage renderPassUsage);

        /**
         * @brief Updates renderpass usage tracking information
         */
        void UpdateRenderPassUsage(u32 renderPassIndex, texture::RenderPassUsage renderPassUsage);

        /**
         * @return The last usage of the texture
         */
        texture::RenderPassUsage GetLastRenderPassUsage();

        bool RequiresRPBreak(bool isDescriptor);

        constexpr const vk::ImageLayout GetLayout() const {
            if (layout == vk::ImageLayout::eUndefined)
                return vk::ImageLayout::eGeneral;
            else
                return layout;
        }

        /**
         * @brief Creates a texture object wrapping the supplied backing with the supplied attributes
         */
        HostTexture(Texture& texture, TextureViewRequestInfo &info, vk::ImageType imageType, bool mutableFormat);

        ~HostTexture();

        void Access(const vk::raii::CommandBuffer &commandBuffer, const ExecutorTrackingInfo &trackingInfo, vk::PipelineStageFlags dstStage, vk::AccessFlags dstAccess, bool forWrite);

        void AccessForTransfer(const vk::raii::CommandBuffer &commandBuffer, const ExecutorTrackingInfo &trackingInfo, bool forWrite);

        constexpr const vk::Image GetImage() const {
            return backing.vkImage;
        }
    };
}
