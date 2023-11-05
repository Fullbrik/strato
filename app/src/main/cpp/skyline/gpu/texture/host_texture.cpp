// SPDX-License-Identifier: MPL-2.0
// Copyright © 2023 Skyline Team and Contributors (https://github.com/skyline-emu/)

#include <gpu.h>
#include <common/trace.h>
#include "common.h"
#include "host_texture.h"
#include "texture.h"
#include "bc_decoder.h"
#include "layout.h"
#include <gpu/interconnect/command_executor.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>

namespace skyline::gpu {
    constexpr bool FormatSupportsStorageImage(gpu::GPU &gpu, vk::Format format) {
        const auto &physicalDevice{gpu.vkPhysicalDevice};
        auto formatPropereties{physicalDevice.getFormatProperties(format)};

        return static_cast<bool>(formatPropereties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eStorageImage);
    }

    HostTextureView::HostTextureView(HostTexture *hostTexture, vk::ImageViewType type, texture::Format format, vk::ComponentMapping components, vk::ImageSubresourceRange range, vk::raii::ImageView &&vkView) : hostTexture{hostTexture}, texture{&hostTexture->texture}, type{type}, format{format}, components{components}, range{range}, vkView{std::move(vkView)} {}

    void HostTextureView::lock() {
        if (texture)
            texture->lock();
    }

    bool HostTextureView::LockWithTag(ContextTag tag) {
        if (texture)
            return texture->LockWithTag(tag);
        else
            return false;
    }

    void HostTextureView::unlock() {
        if (texture)
            texture->unlock();
    }

    bool HostTextureView::try_lock() {
        if (texture)
            return texture->try_lock();
        else
            return false;
    }

    void HostTextureView::RequestSync(interconnect::CommandExecutor &executor, const TextureSyncRequestArgs &args) {
        if (hostTexture) {
            executor.AttachTextureView(this);
            hostTexture->RequestSync(executor, args, range);
        }
    }

    std::shared_ptr<memory::StagingBuffer> HostTexture::SynchronizeHostImpl() {
        u8 *guestInput{texture.mirror.data()};

        u8 *bufferData;
        auto stagingBuffer{[&]() -> std::shared_ptr<memory::StagingBuffer> {
            if constexpr (tiling == vk::ImageTiling::eOptimal) {
                // We need a staging buffer for all optimal copies (since we aren't aware of the host optimal layout) and linear textures which we cannot map on the CPU since we do not have access to their backing VkDeviceMemory
                if (!texture.syncStagingBuffer) [[unlikely]]
                    texture.syncStagingBuffer = texture.gpu.memory.AllocateStagingBuffer(copySize);
                bufferData = texture.syncStagingBuffer->data();
                return texture.syncStagingBuffer->shared_from_this();
            } else if constexpr (tiling == vk::ImageTiling::eLinear) {
                // We can optimize linear texture sync on a UMA by mapping the texture onto the CPU and copying directly into it rather than a staging buffer
                bufferData = backing.data();
                return nullptr;
            } else {
                throw exception("Guest -> Host synchronization of images tiled as '{}' isn't implemented", vk::to_string(tiling));
            }
        }()};

        std::vector<u8> deswizzleBuffer{};
        u8 *deswizzledOutput;
        if (guestFormat != format) {
            deswizzleBuffer.resize(guest.linearSize);
            deswizzledOutput = deswizzleBuffer.data();
        } else [[likely]] {
            deswizzledOutput = bufferData;
        }

        if (guest.levelCount == 1) {
            u8 *hostOutput{deswizzledOutput};
            for (size_t layer{}; layer < guest.layerCount; ++layer) {
                if (guest.tileConfig.mode == texture::TileMode::Block) [[likely]]
                    texture::CopyBlockLinearToLinear(guestMipLayouts[0].dimensions,
                                                     guestFormat->blockWidth, guestFormat->blockHeight, guestFormat->bpb,
                                                     guestMipLayouts[0].blockHeight, guestMipLayouts[0].blockDepth, guest.tileConfig.sparseBlockWidth,
                                                     guestInput, hostOutput
                    );
                else if (guest.tileConfig.mode == texture::TileMode::Pitch)
                    texture::CopyPitchLinearToLinear(guest, guestInput, hostOutput);
                else if (guest.tileConfig.mode == texture::TileMode::Linear)
                    std::memcpy(hostOutput, guestInput, guestMipLayouts[0].linearSize);
                guestInput += guest.layerStride;
                hostOutput += guest.linearLayerStride;
            }
        } else if (guest.levelCount > 1 && guest.tileConfig.mode == texture::TileMode::Block) {
            // We need to generate a buffer that has all layers for a given mip level while Tegra X1 layout holds all mip levels for a given layer
            u8 *input{guestInput}, *output{deswizzledOutput};
            for (const auto &level : guestMipLayouts) {
                u32 outputOffset{};
                for (size_t layer{}, layerOffset{}; layer < guest.layerCount; ++layer, outputOffset += level.linearSize, layerOffset += guest.layerStride) {
                    texture::CopyBlockLinearToLinear(level.dimensions,
                                                     guestFormat->blockWidth, guestFormat->blockHeight, guestFormat->bpb,
                                                     level.blockHeight, level.blockDepth, guest.tileConfig.sparseBlockWidth,
                                                     input + layerOffset, output + outputOffset
                    );
                }
                input += level.blockLinearSize;
                output += level.linearSize * guest.layerCount;
            }
        } else if (guest.levelCount != 0) [[unlikely]] {
            throw exception("Mipmapped textures with tiling mode '{}' aren't supported", static_cast<i32>(guest.tileConfig.mode));
        }

        if (needsDecompression) {
            for (const auto &level : guestMipLayouts) {
                size_t levelHeight{level.dimensions.height * guest.layerCount}; //!< The height of an image representing all layers in the entire level
                switch (guestFormat->vkFormat) {
                    case vk::Format::eBc1RgbaUnormBlock:
                    case vk::Format::eBc1RgbaSrgbBlock:
                        bcn::DecodeBc1(deswizzledOutput, bufferData, level.dimensions.width, levelHeight, true);
                        break;

                    case vk::Format::eBc2UnormBlock:
                    case vk::Format::eBc2SrgbBlock:
                        bcn::DecodeBc2(deswizzledOutput, bufferData, level.dimensions.width, levelHeight);
                        break;

                    case vk::Format::eBc3UnormBlock:
                    case vk::Format::eBc3SrgbBlock:
                        bcn::DecodeBc3(deswizzledOutput, bufferData, level.dimensions.width, levelHeight);
                        break;

                    case vk::Format::eBc4UnormBlock:
                        bcn::DecodeBc4(deswizzledOutput, bufferData, level.dimensions.width, levelHeight, false);
                        break;
                    case vk::Format::eBc4SnormBlock:
                        bcn::DecodeBc4(deswizzledOutput, bufferData, level.dimensions.width, levelHeight, true);
                        break;

                    case vk::Format::eBc5UnormBlock:
                        bcn::DecodeBc5(deswizzledOutput, bufferData, level.dimensions.width, levelHeight, false);
                        break;
                    case vk::Format::eBc5SnormBlock:
                        bcn::DecodeBc5(deswizzledOutput, bufferData, level.dimensions.width, levelHeight, true);
                        break;

                    case vk::Format::eBc6HUfloatBlock:
                        bcn::DecodeBc6(deswizzledOutput, bufferData, level.dimensions.width, levelHeight, false);
                        break;
                    case vk::Format::eBc6HSfloatBlock:
                        bcn::DecodeBc6(deswizzledOutput, bufferData, level.dimensions.width, levelHeight, true);
                        break;

                    case vk::Format::eBc7UnormBlock:
                    case vk::Format::eBc7SrgbBlock:
                        bcn::DecodeBc7(deswizzledOutput, bufferData, level.dimensions.width, levelHeight);
                        break;

                    default:
                        throw exception("Unsupported guest format '{}'", vk::to_string(guestFormat->vkFormat));
                }

                deswizzledOutput += level.linearSize * guest.layerCount;
                bufferData += format->GetSize(level.dimensions) * guest.layerCount;
            }
        }

        return stagingBuffer;
    }

    std::vector<vk::BufferImageCopy> HostTexture::GetBufferImageCopies() {
        std::vector<vk::BufferImageCopy> bufferImageCopies{};
        bufferImageCopies.reserve(guest.levelCount);

        auto pushBufferImageCopyWithAspect{[&](vk::ImageAspectFlagBits aspect) {
            vk::DeviceSize bufferOffset{};
            u32 mipLevel{};
            const auto &mipLayouts{needsDecompression ? *hostMipLayouts : guestMipLayouts};
            for (const auto &level : mipLayouts) {
                bufferImageCopies.emplace_back(
                    vk::BufferImageCopy {
                        .bufferOffset = bufferOffset,
                        .imageSubresource = {
                            .aspectMask = aspect,
                            .mipLevel = mipLevel++,
                            .layerCount = guest.layerCount
                        },
                        .imageExtent = level.dimensions
                    }
                );
                bufferOffset += level.linearSize * guest.layerCount;
            }
        }};

        if (format->vkAspect & vk::ImageAspectFlagBits::eColor)
            pushBufferImageCopyWithAspect(vk::ImageAspectFlagBits::eColor);
        else if (format->vkAspect & vk::ImageAspectFlagBits::eDepth)
            pushBufferImageCopyWithAspect(vk::ImageAspectFlagBits::eDepth);
        else if (format->vkAspect & vk::ImageAspectFlagBits::eStencil)
            pushBufferImageCopyWithAspect(vk::ImageAspectFlagBits::eStencil);

        return bufferImageCopies;
    }

    // TODO: make use of the rest of the arguments
    void HostTexture::RequestSync(interconnect::CommandExecutor &executor, const TextureSyncRequestArgs &args, vk::ImageSubresourceRange &viewRange) {
        texture.MarkGpuDirty(executor.usageTracker); //!< Force all overlapping buffers to sync

        std::unique_lock lock{texture.accessMutex};

        bool createTransferPass{(writtenSinceTP || readSinceTP) && !usedInRP};

        isUTpending = texture.gpu.textureUsageTracker.ShouldSyncHost(texture.usageHandle);

        const TextureSyncRequestArgs &transferArgs{isUTpending ? TextureSyncRequestArgs{
            .isReadInTP = true,
            .isRead = false,
            .isWritten = false,
            .usedStage = vk::PipelineStageFlagBits::eTransfer,
            .usedFlags = vk::AccessFlagBits::eTransferRead
        } : args
        };

        if (dirtyState == DirtyState::OtherHostDirty) {
            HostTexture *toSyncFrom{};
            bool skipSync{};
            for (auto &host : texture.hosts) {
                if (host.dirtyState == DirtyState::HostDirty) {
                    toSyncFrom = &host;
                    skipSync = toSyncFrom->needsDecompression || needsDecompression || toSyncFrom->format->vkAspect == (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil) || format->vkAspect == (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil);

                    if (!skipSync)
                        executor.AddTextureBarrier(*toSyncFrom, {
                            .isReadInTP = true,
                            .isRead = false,
                            .isWritten = false,
                            .usedStage = vk::PipelineStageFlagBits::eTransfer,
                            .usedFlags = vk::AccessFlagBits::eTransferRead
                        });

                    toSyncFrom->trackingInfo.waitedStages |= vk::PipelineStageFlagBits::eTransfer;
                    break;
                }
            }

            if (skipSync) {
                executor.AddTextureBarrier(*this, transferArgs);
            } else {
                if (!texture.syncStagingBuffer) [[unlikely]] {
                    texture.syncStagingBuffer = texture.gpu.memory.AllocateStagingBuffer(copySize);
                    executor.cycle->AttachObject(texture.syncStagingBuffer->shared_from_this());
                }

                executor.AddStagedTextureTransferCommand(*this, transferArgs, *texture.syncStagingBuffer, [stagingBuffer = texture.syncStagingBuffer->shared_from_this(), toSyncFrom](const vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &, GPU &){
                    toSyncFrom->CopyIntoStagingBuffer(commandBuffer, stagingBuffer);
                }, [this, stagingBuffer = texture.syncStagingBuffer->shared_from_this()](const vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &, GPU &){
                    CopyFromStagingBuffer(commandBuffer, stagingBuffer);
                });
            }

            if (transferArgs.isWritten) {
                trackingInfo.lastUsedStage = transferArgs.usedStage;
                trackingInfo.lastUsedAccessFlag = transferArgs.usedFlags;

                for (auto &host : texture.hosts)
                    host.dirtyState = DirtyState::OtherHostDirty;

                texture.gpu.state.nce->TrapRegions(*texture.trapHandle, false);

                trackingInfo.waitedStages = {};
            } else {
                trackingInfo.lastUsedStage = vk::PipelineStageFlagBits::eTransfer;
                trackingInfo.lastUsedAccessFlag = vk::AccessFlagBits::eTransferWrite;

                trackingInfo.waitedStages = transferArgs.usedStage;
            }

            dirtyState = DirtyState::HostDirty;
        } else if (dirtyState == DirtyState::GuestDirty) {
            executor.AddTextureTransferCommand(*this, transferArgs, [this](const vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &pCycle, GPU &){
                texture.SynchronizeHostInline(commandBuffer, pCycle, *this);
            });

            if (transferArgs.isWritten) {
                trackingInfo.lastUsedStage = transferArgs.usedStage;
                trackingInfo.lastUsedAccessFlag = transferArgs.usedFlags;

                texture.gpu.state.nce->TrapRegions(*texture.trapHandle, false);

                for (auto &host : texture.hosts)
                    host.dirtyState = DirtyState::OtherHostDirty;

                dirtyState = DirtyState::HostDirty;
                trackingInfo.waitedStages = {};
            } else {
                trackingInfo.lastUsedStage = vk::PipelineStageFlagBits::eTransfer;
                trackingInfo.lastUsedAccessFlag = vk::AccessFlagBits::eTransferWrite;

                texture.gpu.state.nce->TrapRegions(*texture.trapHandle, true);

                dirtyState = DirtyState::Clean;
                trackingInfo.waitedStages = transferArgs.usedStage;
            }
        } else {
            if (args.isWritten) {
                if (!usedInRP && !isUTpending)
                    executor.AddTextureBarrier(*this, args);

                texture.gpu.state.nce->TrapRegions(*texture.trapHandle, false);

                for (auto &host : texture.hosts)
                    host.dirtyState = DirtyState::OtherHostDirty;

                dirtyState = DirtyState::HostDirty;

                trackingInfo.lastUsedStage = args.usedStage;
                trackingInfo.lastUsedAccessFlag = args.usedFlags;

                trackingInfo.waitedStages = {};
            } else {
                if (!usedInRP && !isUTpending)
                    executor.AddTextureBarrier(*this, args);

                trackingInfo.waitedStages |= args.usedStage;
            }
        }

        texture.gpu.textureUsageTracker.RequestSync(executor, texture.usageHandle, this, args, createTransferPass);
        isUTpending = false;
    }

    void HostTexture::CopyFromStagingBuffer(const vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<memory::StagingBuffer> &stagingBuffer) {
        auto bufferImageCopies{GetBufferImageCopies()};
        commandBuffer.copyBufferToImage(stagingBuffer->vkBuffer, backing.vkImage, GetLayout(), bufferImageCopies);
    }

    void HostTexture::CopyIntoStagingBuffer(const vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<memory::StagingBuffer> &stagingBuffer) {
        auto bufferImageCopies{GetBufferImageCopies()};
        commandBuffer.copyImageToBuffer(backing.vkImage, GetLayout(), stagingBuffer->vkBuffer, bufferImageCopies);
    }

    void HostTexture::CopyToGuest(u8 *hostBuffer) {
        u8 *guestOutput{texture.mirror.data()};

        if (guest.levelCount == 1) {
            for (size_t layer{}; layer < guest.layerCount; ++layer) {
                if (guest.tileConfig.mode == texture::TileMode::Block)
                    texture::CopyLinearToBlockLinear(guestMipLayouts[0].dimensions,
                                                     guestFormat->blockWidth, guestFormat->blockHeight, guestFormat->bpb,
                                                     guestMipLayouts[0].blockHeight, guestMipLayouts[0].blockDepth, guest.tileConfig.sparseBlockWidth,
                                                     hostBuffer, guestOutput
                    );
                else if (guest.tileConfig.mode == texture::TileMode::Pitch)
                    texture::CopyLinearToPitchLinear(guest, hostBuffer, guestOutput);
                else if (guest.tileConfig.mode == texture::TileMode::Linear)
                    std::memcpy(guestOutput, hostBuffer, guestMipLayouts[0].linearSize);
                guestOutput += guest.layerStride;
                hostBuffer += guest.linearLayerStride;
            }
        } else if (guest.levelCount > 1 && guest.tileConfig.mode == texture::TileMode::Block) {
            // We need to copy into the Tegra X1 layout holds all mip levels for a given layer while the input buffer has all layers for a given mip level
            // Note: See SynchronizeHostImpl for additional comments
            u8 *input{hostBuffer}, *output{guestOutput};
            for (const auto &level : guestMipLayouts) {
                u32 inputOffset{};
                for (size_t layer{}, layerOffset{}; layer < guest.layerCount; ++layer, inputOffset += level.linearSize, layerOffset += guest.layerStride) {
                    texture::CopyLinearToBlockLinear(level.dimensions,
                                                     guestFormat->blockWidth, guestFormat->blockHeight, guestFormat->bpb,
                                                     level.blockHeight, level.blockDepth, guest.tileConfig.sparseBlockWidth,
                                                     input + inputOffset, output + layerOffset
                    );
                }
                input += level.linearSize * guest.layerCount;
                output += level.blockLinearSize;
            }

        } else if (guest.levelCount != 0) [[unlikely]] {
            throw exception("Mipmapped textures with tiling mode '{}' aren't supported", static_cast<i32>(tiling));
        }
    }

    vk::ImageType HostTexture::ConvertViewType(vk::ImageViewType viewType, texture::Dimensions dimensions) {
        switch (viewType) {
            case vk::ImageViewType::e1D:
            case vk::ImageViewType::e1DArray:
                return vk::ImageType::e1D;
            case vk::ImageViewType::e2D:
            case vk::ImageViewType::e2DArray:
                // If depth is > 1 this is a 2D view into a 3D texture so the underlying image needs to be created as 3D
                if (dimensions.depth > 1)
                    return vk::ImageType::e3D;
                else
                    return vk::ImageType::e2D;
            case vk::ImageViewType::eCube:
            case vk::ImageViewType::eCubeArray:
                return vk::ImageType::e2D;
            case vk::ImageViewType::e3D:
                return vk::ImageType::e3D;
        }
    }

    bool HostTexture::ValidateRenderPassUsage(u32 renderPassIndex, texture::RenderPassUsage renderPassUsage, bool isWrite) {
        return lastRenderPassIndex != renderPassIndex || (lastRenderPassUsage != texture::RenderPassUsage::None && lastRenderPassUsage != renderPassUsage) || RequiresNewTP(isWrite);
    }

    bool HostTexture::RequiresNewTP(bool willWrite) {
        //return true;

        auto checkUT{[&]() -> bool {
            auto overlaps{texture.gpu.textureUsageTracker.GetOverlaps(texture.usageHandle)};

            return std::ranges::any_of(overlaps, [](auto &overlap) {
                return overlap.get().dirtyTexture->writtenSinceTP || overlap.get().dirtyTexture->readSinceTP;
            });
        }};

        HostTexture *dirtyTexture;
        if (dirtyState == DirtyState::OtherHostDirty) {
            for (auto &host : texture.hosts) {
                if (host.dirtyState == DirtyState::HostDirty) {
                    dirtyTexture = &host;
                    break;
                }
            }
        }

        if (writtenSinceTP || (dirtyState == DirtyState::OtherHostDirty && (dirtyTexture->writtenSinceTP || dirtyTexture->writtenInTP || dirtyTexture->readInTP || dirtyTexture->readSinceTP)))
            return true;

        if (!usedInRP) {
            if (willWrite && (readSinceTP || ((writtenInTP || readInTP) && dirtyState == DirtyState::GuestDirty)))
                return true;
            else if (!willWrite && ((writtenInTP || readInTP || readSinceTP) && dirtyState == DirtyState::GuestDirty))
                return true;
        }

        return checkUT();
    }

    HostTexture::HostTexture(Texture &texture, TextureViewRequestInfo &info, vk::ImageType imageType, bool mutableFormat)
        : texture{texture},
          guest{texture.guest},
          dimensions{info.imageDimensions},
          sampleCount{info.sampleCount},
          dirtyState{DirtyState::GuestDirty},
          guestFormat{info.viewFormat},
          format{format::ConvertHostCompatibleFormat(info.viewFormat, texture.gpu.traits)},
          needsDecompression{info.viewFormat != format},
          guestMipLayouts{texture::CalculateMipLayout(
              info.imageDimensions,
              guestFormat->blockHeight, guestFormat->blockWidth, guestFormat->bpb,
              texture.guest.tileConfig.blockHeight, texture.guest.tileConfig.blockDepth, texture.guest.tileConfig.sparseBlockWidth,
              texture.guest.levelCount)},
          hostMipLayouts{needsDecompression ? texture::CalculateMipLayout(
              info.imageDimensions,
              format->blockHeight, format->blockWidth, format->bpb,
              texture.guest.tileConfig.blockHeight, texture.guest.tileConfig.blockDepth, texture.guest.tileConfig.sparseBlockWidth,
              texture.guest.levelCount) : std::optional<std::vector<texture::MipLevelLayout>>{}},
          imageType{imageType},
          flags{mutableFormat ? vk::ImageCreateFlagBits::eMutableFormat | vk::ImageCreateFlagBits::eExtendedUsage : vk::ImageCreateFlags{}},
          usage{vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled},
          copySize{static_cast<u32>(texture::CalculateLinearLayerStride(needsDecompression ? *hostMipLayouts : guestMipLayouts) * guest.layerCount)} {
        auto &gpu{texture.gpu};

        for (auto &host : texture.hosts) {
            if (host.dirtyState == DirtyState::HostDirty) {
                dirtyState = DirtyState::OtherHostDirty;
                break;
            }
        }

        if (info.extraUsageFlags & vk::ImageUsageFlagBits::eStorage) {
            if (FormatSupportsStorageImage(gpu, *format))
                usage |= vk::ImageUsageFlagBits::eStorage;
            else
                LOGE("Requested format doesn't support being used as a storage image on host: {}", vk::to_string(info.viewFormat->vkFormat));
        }

        if ((format->vkAspect & vk::ImageAspectFlagBits::eColor) && !format->IsCompressed())
            usage |= vk::ImageUsageFlagBits::eColorAttachment;
        if (format->vkAspect & (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil))
            usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;

        if constexpr (tiling == vk::ImageTiling::eOptimal)
            layout = vk::ImageLayout::eUndefined;
        else
            layout = vk::ImageLayout::ePreinitialized;

        if (imageType == vk::ImageType::e2D && dimensions.width == dimensions.height && guest.layerCount >= 6)
            flags |= vk::ImageCreateFlagBits::eCubeCompatible;
        else if (imageType == vk::ImageType::e3D)
            flags |= vk::ImageCreateFlagBits::e2DArrayCompatible;

        vk::ImageCreateInfo imageCreateInfo{
            .flags = flags,
            .imageType = imageType,
            .format = *format,
            .extent = dimensions,
            .mipLevels = guest.levelCount,
            .arrayLayers = guest.layerCount,
            .samples = sampleCount,
            .tiling = tiling,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
            .queueFamilyIndexCount = 1,
            .pQueueFamilyIndices = &gpu.vkQueueFamilyIndex,
            .initialLayout = layout
        };
        backing = tiling != vk::ImageTiling::eLinear ? gpu.memory.AllocateImage(imageCreateInfo) : gpu.memory.AllocateMappedImage(imageCreateInfo);

        LOGD("Variant created: {}x{}x{}, sample count: {}, format: {}, image type: {}, mapped range: {} - {}, {}",
             dimensions.width, dimensions.height, dimensions.depth, static_cast<i32>(sampleCount), vk::to_string(guestFormat->vkFormat), vk::to_string(imageType), fmt::ptr(guest.mappings.front().data()), fmt::ptr(guest.mappings.front().end().base()), fmt::ptr(&texture));
    }

    HostTexture::~HostTexture() {
        for (auto &view : views) {
            view->texture = nullptr;
            view->hostTexture = nullptr;
            view->stale = true;
            view->vkView = nullptr;
        }
    }
}
