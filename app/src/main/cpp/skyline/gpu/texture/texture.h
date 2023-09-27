// SPDX-License-Identifier: MPL-2.0
// Copyright © 2020 Skyline Team and Contributors (https://github.com/skyline-emu/)

#pragma once

#include <nce.h>
#include <gpu/tag_allocator.h>
#include <gpu/memory_manager.h>
#include <gpu/usage_tracker.h>
#include "guest_texture.h"
#include "host_texture.h"

namespace skyline::gpu {
    class TextureManager;
    struct TextureViewRequestInfo;

    class Texture : public std::enable_shared_from_this<Texture> {
        using DirtyState = HostTexture::DirtyState;
      private:
        GPU &gpu;

        span<u8> mirror{}; //!< A contiguous mirror of all the guest mappings to allow linear access on the CPU
        std::optional<nce::NCE::TrapHandle> trapHandle{}; //!< The handle of the traps for the guest mappings
        TextureUsageTracker::TextureHandle usageHandle;
        std::mutex accessMutex; //!< Synchronizes access to the dirtyState of any host of this texture

        std::atomic<ContextTag> tag{}; //!< The tag associated with the last lock call on this texture
        std::mutex userMutex; //!< Synchronizes execution access to the texture making sure it's not used out of order

        // TODO: make this per host
        bool mutableFormat; //!< If the format of all the host textures is mutable for views

        bool stale{}; //!< Wheter the texture is scheduled for destruction
        bool isRT{}; //!< Wheter the texture was created as an RenderTarget or not

        static constexpr size_t SkipReadbackHackWaitCountThreshold{6}; //!< Threshold for the number of times a texture can be waited on before it should be considered for the readback hack
        static constexpr std::chrono::nanoseconds SkipReadbackHackWaitTimeThreshold{constant::NsInSecond / 4}; //!< Threshold for the amount of time a texture can be waited on before it should be considered for the readback hack, `SkipReadbackHackWaitCountThreshold` needs to be hit before this
        size_t accumulatedGuestWaitCounter{}; //!< Total number of times the texture has been waited on
        std::chrono::nanoseconds accumulatedGuestWaitTime{}; //!< Amount of time the texture has been waited on for since the `SkipReadbackHackWaitCountThreshold`th wait on it by the guest

        friend class HostTexture;
        friend class TextureManager;
        friend class TextureUsageTracker;

        /**
         * @brief Sets up mirror mappings for the guest mappings, this must be called after construction for the mirror to be valid
         * @note This is seperated from the constructor due to weak_from_this() from not being callable inside one
         */
        void SetupGuestMappings();

        // TODO: recreate comments after rewrite
        void SynchronizeHostInline(const vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &cycle, HostTexture &toSync);

        void SynchronizeGuest(bool isWritten);

      public:
        std::shared_ptr<memory::StagingBuffer> syncStagingBuffer{}; // Used if it's needed for synchronisation by an execution, discarded afterwards

        std::shared_ptr<FenceCycle> cycle{}; //!< A fence cycle for when any host operation mutating the texture has completed, it must be waited on prior to any changes

        std::list<HostTexture> hosts{};
        GuestTexture guest;

        Texture() = delete;
        Texture(GPU &gpu, TextureViewRequestInfo &info, bool mutableFormat = false);

        ~Texture();

        /**
         * @brief Acquires an exclusive lock on the texture for the calling thread
         * @note Naming is in accordance to the BasicLockable named requirement
         */
        void lock();

        /**
         * @brief Acquires an exclusive lock on the texture for the calling thread
         * @param tag A tag to associate with the lock, future invocations with the same tag prior to the unlock will acquire the lock without waiting (A default initialised tag will disable this behaviour)
         * @return If the lock was acquired by this call as opposed to the texture already being locked with the same tag
         * @note All locks using the same tag **must** be from the same thread as it'll only have one corresponding unlock() call
         */
        bool LockWithTag(ContextTag tag);

        /**
         * @brief Relinquishes an existing lock on the texture by the calling thread
         * @note Naming is in accordance to the BasicLockable named requirement
         */
        void unlock();

        /**
         * @brief Attempts to acquire an exclusive lock but returns immediately if it's captured by another thread
         * @note Naming is in accordance to the Lockable named requirement
         */
        bool try_lock();

        /**
         * @brief Syncronises the specified host texture
         * @note The texture **must** be locked during this
         */
        void SynchronizeHost(HostTexture &toSyncFrom, vk::PipelineStageFlags waitStage, vk::AccessFlags waitFlags);

        /**
         * @brief Attempts to find or create a host texture view for the given parameters, this may result in the creation of a new host texture
         * @return A pointer to the host texture view, this may be null if a host texture is compatible but needs VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT
         */
        HostTextureView *FindOrCreateView(TextureViewRequestInfo &info, vk::ImageSubresourceRange viewRange, u32 actualBaseMip = 0);

        /**
         * @brief Waits on a fence cycle if it exists till it's signalled and resets it after
         * @note The texture **must** be locked prior to calling this
         */
        void WaitOnFence();

        /**
         * @brief Attaches a fence cycle to the texture, chaining it to the existing fence cycle if it exists
         */
        void AttachCycle(const std::shared_ptr<FenceCycle>& cycle);

        /**
         * @brief Marks the texture as being GPU dirty
         */
        void MarkGpuDirty(UsageTracker &usageTracker);

        void OnExecStart(interconnect::CommandExecutor &executor);

        void OnExecEnd();
    };
}
