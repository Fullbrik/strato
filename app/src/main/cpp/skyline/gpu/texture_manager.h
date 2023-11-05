// SPDX-License-Identifier: MPL-2.0
// Copyright © 2021 Skyline Team and Contributors (https://github.com/skyline-emu/)

#pragma once

#include <common/linear_allocator.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_structs.hpp>
#include "texture/texture.h"

namespace skyline::gpu {
    namespace interconnect {
        class CommandExecutor;
    }

    struct TextureViewRequestInfo {
        ContextTag tag;
        bool isRT{}; //!< Wheter the texture is needed as a render target for the maxwell3d engine. Such textures require special cases when matching.
        texture::Mappings mappings; //!< The CPU addresses to which the textures are mapped to
        texture::Dimensions sampleDimensions; //!< The dimensions of the guest texture, this includes all samples from MSAA textures
        texture::Dimensions imageDimensions{}; //!< An optional hint of the size of the image without MSAA, if this isn't specified then it'll be inferred along with the sample count based on any matches
        vk::SampleCountFlagBits sampleCount{vk::SampleCountFlagBits::e1}; //!< The sample count of the guest texture, this is ignored if imageDimensions isn't specified
        texture::TileConfig tileConfig;
        u16 layerCount{1};
        u16 levelCount{1}; //!< The minimum amount of mip levels the texture the view comes from has
        u32 layerStride; //!< The amount of bytes between the start of each layer, or the size of the layer if layerCount equals 1
        texture::Format viewFormat; //!< The format of the returned view if the host gpu supports it
        vk::ImageAspectFlags viewAspect; //!< The format aspects of the returned view
        vk::ImageViewType viewType{vk::ImageViewType::e2D};
        vk::ComponentMapping viewComponents{};
        u16 viewMipBase{}; //!< The base mip level of the view, this is used to create a view of a subset of the texture (however the view you get isn't guaranteed to be the exact mip level you specify depending on what the view request matches)
        u16 viewMipCount{}; //!< The number of mip levels in the view, if zero then levelCount - viewMipBase is used
        u16 viewLayerBase{}; //!< The base layer of the returned view
        vk::ImageUsageFlags extraUsageFlags{}; //!< Extra vk::ImageUsageFlags that are required for the vkImage the view is created from
    };

    /**
     * @brief The Texture Manager is responsible for maintaining a global view of textures being mapped from the guest to the host, any lookups and creation of host texture from equivalent guest textures alongside reconciliation of any overlaps with existing textures
     */
    class TextureManager {
      private:
        /**
         * @brief A single contiguous mapping of a texture in the CPU address space
         */
        struct TextureMapping {
            std::shared_ptr<Texture> texture;
            texture::Mappings mappings;
        };

        GPU &gpu;
        std::list<TextureMapping> textures; //!< A sorted vector of all texture mappings
        std::vector<std::shared_ptr<Texture>> texturesPendingDestruction; //!< A vector of textures that will be destroyed once execution is submitted

        std::optional<memory::Image> nullImage;
        vk::raii::ImageView nullImageView{nullptr}; //!< A cached null texture view to avoid unnecessary recreation

        std::shared_ptr<Texture> CreateTexture(TextureViewRequestInfo &info, bool mutableFormat = false);

        void DestroyTexture(const std::list<TextureMapping>::iterator &texture);

        HostTextureView *FindOrCreateView(const std::list<TextureMapping>::iterator &it, TextureViewRequestInfo &&info, vk::ImageSubresourceRange &viewRange);

        SpinLock mutex; //!< The mutex used to lock the texture manager, this is used to prevent concurrent lookups and (re)creation of textures

      public:
        LinearAllocatorState<> viewAllocatorState; //!< Linear allocator used to allocate texture views

        TextureManager(GPU &gpu);

        void DestroyStaleTextures();

        /**
         * @note Read TextureViewRequestInfo for a description of the arguments
         * @return A pre-existing or newly created HostTextureView which matches the specified criteria
         */
        HostTextureView *FindOrCreate(TextureViewRequestInfo &&info);

        /**
         * @return A 2D 1x1 RGBA8888 null texture view with (0,0,0,1) component mappings
         */
        vk::ImageView GetNullView();

        /**
         * @brief Attempts to perform garbage collection on all existing textures
         * @param level The lower this is the more aggressive garbage collection is
         */
        void OnTrim(i32 level);
    };
}
