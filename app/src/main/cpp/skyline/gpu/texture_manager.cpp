// SPDX-License-Identifier: MPL-2.0
// Copyright © 2021 Skyline Team and Contributors (https://github.com/skyline-emu/)

#include <common/trace.h>
#include <gpu.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>
#include "texture/common.h"
#include "texture/guest_texture.h"
#include "texture_manager.h"
#include "texture/layout.h"

namespace skyline::gpu {
    std::shared_ptr<Texture> TextureManager::CreateTexture(TextureViewRequestInfo &info, bool mutableFormat) {
        LOGD("Texture created: {} tile mode, {} levels, {} layers, 0x{:X} layer stride, mutableFormat: {}", info.tileConfig.mode, info.levelCount, info.layerCount, info.layerStride, mutableFormat);

        auto texture{std::make_shared<Texture>(gpu, info, mutableFormat)};
        texture->SetupGuestMappings();
        textures.emplace_back(TextureMapping{.texture = texture->shared_from_this(), .mappings = info.mappings});

        return texture;
    }

    void TextureManager::DestroyTexture(const std::list<TextureMapping>::iterator &it) {
        auto &texture{it->texture};
        for (const auto &host : texture->hosts)
            for (auto &view : host.views)
                view->stale = true;

        texture->stale = true;

        texturesPendingDestruction.emplace_back(texture->shared_from_this());

        textures.erase(it);
    }

    HostTextureView *TextureManager::FindOrCreateView(const std::list<TextureMapping>::iterator &it, TextureViewRequestInfo &&info, vk::ImageSubresourceRange &viewRange) {
        auto &texture{it->texture};
        ContextLock lock{info.tag, *texture};
        auto &targetHost{texture->hosts.front()};
        auto &targetGuest{texture->guest};
        auto targetTexture{texture};

        if (texture->isRT && !info.isRT) {
            if (targetGuest.mipLayouts[viewRange.baseMipLevel].dimensions > info.viewFormat->GetDimensionsInBytes(info.sampleDimensions)) {
                DestroyTexture(it);

                targetTexture = CreateTexture(info, texture->mutableFormat);
                ContextLock newLock{info.tag, *targetTexture};
            } else {
                texture->isRT = false;
            }
        } else if (!texture->isRT && info.isRT) {
            texture::Dimensions unAlignedDimensions{info.viewFormat->GetDimensionsFromBytes(targetHost.format->GetDimensionsInBytes(targetHost.dimensions))};
            info.sampleDimensions = info.imageDimensions = unAlignedDimensions;
        }

        u32 actualBaseMip{viewRange.baseMipLevel};
        viewRange.baseMipLevel += info.viewMipBase;
        viewRange.levelCount = info.viewMipCount;
        viewRange.baseArrayLayer += info.viewLayerBase;
        viewRange.layerCount = static_cast<u32>(info.layerCount - info.viewLayerBase);

        auto view{targetTexture->FindOrCreateView(info, viewRange, actualBaseMip)};
        if (view)
            return view;

        // We need to create a successor texture with host mutability to allow for the view to be created
        DestroyTexture(std::ranges::find_if(textures, [&targetTexture](const auto texture) {
            return texture.texture.get() == targetTexture.get();
        }));
        auto successor{CreateTexture(info, true)};
        return successor->FindOrCreateView(info, viewRange, actualBaseMip);
    }

    TextureManager::TextureManager(GPU &gpu) : gpu(gpu) {}

    void TextureManager::DestroyStaleTextures() {
        std::unique_lock texlock{mutex};

        // FIXME: We have to make sure this is the only remaining reference to the texture, we don't want the cycle waiter to destroy the texture because that will cause deadlocks
        std::erase_if(texturesPendingDestruction, [](const auto &texture) {
            return texture.use_count() == 1;
        });
    }

    HostTextureView *TextureManager::FindOrCreate(TextureViewRequestInfo &&info) {
        if (info.viewMipCount == 0)
            info.viewMipCount = info.levelCount - info.viewMipBase;

        std::unique_lock texlock{mutex};

        /*
         * Iterate over all textures that overlap with the first mapping of the guest texture and compare the mappings:
         * 1) All mappings match up perfectly, we check that the rest of the supplied mappings correspond to mappings in the texture
         * 1.1) If they match as well, we check for format/dimensions/tiling config matching the texture and return or move onto (3)
         * 2) Only a contiguous range of mappings match, we check for if the overlap is meaningful and compatible with layout math, it can go two ways:
         * 2.1) If there is a meaningful overlap, we return a view to the texture
         * 2.2) If there isn't, we move onto (3)
         * 3) If there's another overlap we go back to (1) else we go to (4)
         * 4) We check all the overlapping texture for if they're in the texture pool:
         * 4.1) If they are, we do nothing to them
         * 4.2) If they aren't, we delete them from the map
         * 5) Create a new texture and insert it in the map then return it
         */

        for (auto it{textures.begin()}; it != textures.end(); ++it) {
            auto &texture{it->texture};
            auto &targetGuest{texture->guest};

            if (!texture->stale && targetGuest.Contains(info.mappings)) {
                auto &targetHost{texture->hosts.front()};

                if (targetHost.format->IsCompressed() != info.viewFormat->IsCompressed())
                    continue;

                u32 offset{targetGuest.OffsetFrom(info.mappings)}; //!< Offset of the first mapping in the source texture in the target texture
                auto subresource{targetGuest.CalculateSubresource(info.tileConfig, offset, info.viewFormat->GetDimensionsInBytes(info.sampleDimensions), info.levelCount, info.layerCount, info.layerStride, info.viewAspect)};
                if (!subresource)
                    continue;

                if (texture->isRT == info.isRT && targetGuest.mipLayouts[subresource->baseMipLevel].dimensions != info.viewFormat->GetDimensionsInBytes(info.sampleDimensions))
                    continue;

                if (texture->isRT && !info.isRT && targetGuest.mipLayouts[subresource->baseMipLevel].dimensions < info.viewFormat->GetDimensionsInBytes(info.sampleDimensions))
                    continue;

                if (!info.imageDimensions) {
                    info.imageDimensions = CalculateBaseDimensions(info.sampleDimensions, (texture::MsaaConfig)targetHost.sampleCount);
                    info.sampleCount = targetHost.sampleCount;
                }

                return FindOrCreateView(it, std::forward<decltype(info)>(info), *subresource);
            }
        }

        if (!info.imageDimensions) {
            // If there's no texture to match, we assume the texture has no MSAA
            info.imageDimensions = info.sampleDimensions;
            info.sampleCount = vk::SampleCountFlagBits::e1;
        }

        auto newTexture{CreateTexture(info, false)};
        ContextLock lock{info.tag, *newTexture};
        newTexture->isRT = info.isRT;

        auto &newGuest{newTexture->guest};
        for (auto texture{textures.begin()}; texture != textures.end();) {
            ContextLock otherLock{info.tag, *texture->texture};
            auto &textureMappings{texture->mappings};

            if (newGuest.Contains(textureMappings) && !texture->texture->stale && texture->texture.get() != newTexture.get()) {
                auto &targetGuest{texture->texture->guest};
                u32 offset{newGuest.OffsetFrom(textureMappings)}; //!< Offset of the first mapping in the source texture in the target texture
                auto subresource{newGuest.CalculateSubresource(targetGuest.tileConfig, offset, targetGuest.mipLayouts[0].dimensions, targetGuest.levelCount, targetGuest.layerCount, targetGuest.layerStride, info.viewFormat->vkAspect)};
                if (!subresource || newGuest.mipLayouts[subresource->baseMipLevel].dimensions > targetGuest.mipLayouts[0].dimensions) {
                    ++texture;
                    continue;
                }

                DestroyTexture(texture);
            } else {
                ++texture;
            }
        }

        return newTexture->FindOrCreateView(info, vk::ImageSubresourceRange{info.viewAspect, info.viewMipBase, info.viewMipCount, info.viewLayerBase, static_cast<u32>(info.layerCount - info.viewLayerBase)});
    }

    vk::ImageView TextureManager::GetNullView() {
        if (*nullImageView) [[likely]]
            return *nullImageView;

        std::unique_lock lock{mutex};
        if (*nullImageView)
            // Check again in case another thread created the null texture
            return *nullImageView;

        constexpr texture::Format NullImageFormat{format::R8G8B8A8Unorm};
        constexpr texture::Dimensions NullImageDimensions{1, 1, 1};
        constexpr vk::ImageLayout NullImageInitialLayout{vk::ImageLayout::eUndefined};
        constexpr vk::ImageTiling NullImageTiling{vk::ImageTiling::eOptimal};
        constexpr vk::ImageCreateFlags NullImageFlags{};
        constexpr vk::ImageUsageFlags NullImageUsage{vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage};

        nullImage = gpu.memory.AllocateImage({
            .flags = NullImageFlags,
            .imageType = vk::ImageType::e2D,
            .format = NullImageFormat->vkFormat,
            .extent = NullImageDimensions,
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = NullImageTiling,
            .usage = NullImageUsage,
            .sharingMode = vk::SharingMode::eExclusive,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &gpu.vkQueueFamilyIndex,
            .initialLayout = NullImageInitialLayout
        });

        gpu.scheduler.Submit([&](vk::raii::CommandBuffer &commandBuffer) {
            commandBuffer.pipelineBarrier(
                vk::PipelineStageFlagBits::eTopOfPipe,
                vk::PipelineStageFlagBits::eBottomOfPipe,
                {},
                {},
                {},
                vk::ImageMemoryBarrier{
                    .oldLayout = NullImageInitialLayout,
                    .newLayout = vk::ImageLayout::eGeneral,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = nullImage->vkImage,
                    .subresourceRange = vk::ImageSubresourceRange{
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .levelCount = 1,
                        .layerCount = 1
                    }
                });
        })->Wait();

        nullImageView = vk::raii::ImageView(
            gpu.vkDevice,
            vk::ImageViewCreateInfo{
                .image = nullImage->vkImage,
                .viewType = vk::ImageViewType::e2D,
                .format = NullImageFormat->vkFormat,
                .components = vk::ComponentMapping{
                    .r = vk::ComponentSwizzle::eZero,
                    .g = vk::ComponentSwizzle::eZero,
                    .b = vk::ComponentSwizzle::eZero,
                    .a = vk::ComponentSwizzle::eOne
                },
                .subresourceRange = vk::ImageSubresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .levelCount = 1,
                    .layerCount = 1
                }
            }
        );

        return *nullImageView;
    }

    // TODO: Implement this using an specialised LRFU algorithm
    void TextureManager::OnTrim(i32 level) {
        std::unique_lock lock{mutex};

        LOGW("Texture garbage collection triggered: level: {}", level);

        u32 i{};
        u32 maxI{static_cast<u32>(textures.size()) / 4};
        for (auto texture{textures.begin()}; texture != textures.end();) {
            if (texture->texture->try_lock()) {
                DestroyTexture(texture);
                texture->texture->unlock();
            }

            if (i < maxI)
                break;

            ++i;
        }
    }
}
