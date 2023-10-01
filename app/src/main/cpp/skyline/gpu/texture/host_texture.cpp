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

    HostTextureView::HostTextureView(HostTexture *hostTexture, vk::ImageViewType type, texture::Format format, vk::ComponentMapping components, vk::ImageSubresourceRange range, vk::raii::ImageView &&vkView) : hostTexture{hostTexture}, texture{&hostTexture->texture}, type{type}, format{std::move(format)}, components{components}, range{range}, vkView{std::move(vkView)} {}

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
            hostTexture->RequestSync(executor, args, range, nullptr);
        }
    }

    void HostTextureView::RequestRPSync(interconnect::CommandExecutor &executor, const TextureSyncRequestArgs &args, interconnect::node::SyncNode *syncNode) {
        if (hostTexture) {
            executor.AttachTextureView(this);
            hostTexture->RequestSync(executor, args, range, syncNode);
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
        u8 *deswizzleOutput;
        if (guestFormat != format) {
            deswizzleBuffer.resize(guest.linearSize);
            deswizzleOutput = deswizzleBuffer.data();
        } else [[likely]] {
            deswizzleOutput = bufferData;
        }

        if (guest.levelCount == 1) {
            u8 *hostOutput{deswizzleOutput};
            for (size_t layer{}; layer < guest.layerCount; layer++) {
                if (guest.tileConfig.mode == texture::TileMode::Block) [[likely]]
                    texture::CopyBlockLinearToLinear(copyLayouts[0].dimensions,
                                                     guestFormat->blockWidth, guestFormat->blockHeight, guestFormat->bpb,
                                                     copyLayouts[0].blockHeight, copyLayouts[0].blockDepth, guest.tileConfig.sparseBlockWidth,
                                                     guestInput, hostOutput
                    );
                else if (guest.tileConfig.mode == texture::TileMode::Pitch)
                    texture::CopyPitchLinearToLinear(guest, guestInput, hostOutput);
                else if (guest.tileConfig.mode == texture::TileMode::Linear)
                    std::memcpy(hostOutput, guestInput, copyLayouts[0].linearSize);
                guestInput += guest.layerStride;
                hostOutput += guest.linearLayerStride;
            }
        } else if (guest.levelCount > 1 && guest.tileConfig.mode == texture::TileMode::Block) {
            // We need to generate a buffer that has all layers for a given mip level while Tegra X1 layout holds all mip levels for a given layer
            u8 *input{guestInput}, *output{deswizzleOutput};
            for (const auto &level : copyLayouts) {
                u32 outputOffset{};
                for (size_t layer{}, layerOffset{}; layer < guest.layerCount; ++layer, outputOffset += level.linearSize, layerOffset += guest.layerStride) {
                    texture::CopyBlockLinearToLinear(level.dimensions,
                                                     guestFormat->blockWidth, guestFormat->blockHeight, guestFormat->bpb,
                                                     level.blockHeight, level.blockDepth, guest.tileConfig.sparseBlockWidth,
                                                     input + layerOffset, output + outputOffset);
                }
                input += level.blockLinearSize;
                output += level.linearSize * guest.layerCount;
            }
        } else if (guest.levelCount != 0) [[unlikely]] {
            throw exception("Mipmapped textures with tiling mode '{}' aren't supported", static_cast<i32>(guest.tileConfig.mode));
        }

        if (needsDecompression) {
            for (const auto &level : copyLayouts) {
                size_t levelHeight{level.dimensions.height * guest.layerCount}; //!< The height of an image representing all layers in the entire level
                switch (guestFormat->vkFormat) {
                    case vk::Format::eBc1RgbaUnormBlock:
                    case vk::Format::eBc1RgbaSrgbBlock:
                        bcn::DecodeBc1(deswizzleOutput, bufferData, level.dimensions.width, levelHeight, true);
                        break;

                    case vk::Format::eBc2UnormBlock:
                    case vk::Format::eBc2SrgbBlock:
                        bcn::DecodeBc2(deswizzleOutput, bufferData, level.dimensions.width, levelHeight);
                        break;

                    case vk::Format::eBc3UnormBlock:
                    case vk::Format::eBc3SrgbBlock:
                        bcn::DecodeBc3(deswizzleOutput, bufferData, level.dimensions.width, levelHeight);
                        break;

                    case vk::Format::eBc4UnormBlock:
                        bcn::DecodeBc4(deswizzleOutput, bufferData, level.dimensions.width, levelHeight, false);
                        break;
                    case vk::Format::eBc4SnormBlock:
                        bcn::DecodeBc4(deswizzleOutput, bufferData, level.dimensions.width, levelHeight, true);
                        break;

                    case vk::Format::eBc5UnormBlock:
                        bcn::DecodeBc5(deswizzleOutput, bufferData, level.dimensions.width, levelHeight, false);
                        break;
                    case vk::Format::eBc5SnormBlock:
                        bcn::DecodeBc5(deswizzleOutput, bufferData, level.dimensions.width, levelHeight, true);
                        break;

                    case vk::Format::eBc6HUfloatBlock:
                        bcn::DecodeBc6(deswizzleOutput, bufferData, level.dimensions.width, levelHeight, false);
                        break;
                    case vk::Format::eBc6HSfloatBlock:
                        bcn::DecodeBc6(deswizzleOutput, bufferData, level.dimensions.width, levelHeight, true);
                        break;

                    case vk::Format::eBc7UnormBlock:
                    case vk::Format::eBc7SrgbBlock:
                        bcn::DecodeBc7(deswizzleOutput, bufferData, level.dimensions.width, levelHeight);
                        break;

                    default:
                        throw exception("Unsupported guest format '{}'", vk::to_string(guestFormat->vkFormat));
                }

                deswizzleOutput += level.linearSize * guest.layerCount;
                bufferData += format->GetSize(level.dimensions) * guest.layerCount;
            }
        }

        return stagingBuffer;
    }

    boost::container::small_vector<vk::BufferImageCopy, 10> HostTexture::GetBufferImageCopies() {
        boost::container::small_vector<vk::BufferImageCopy, 10> bufferImageCopies;

        auto pushBufferImageCopyWithAspect{[&](vk::ImageAspectFlagBits aspect) {
            vk::DeviceSize bufferOffset{};
            u32 mipLevel{};
            for (auto &level : copyLayouts) {
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
                bufferOffset += (needsDecompression ? format->GetSize(level.dimensions) : level.linearSize) * guest.layerCount;
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
    void HostTexture::RequestSync(interconnect::CommandExecutor &executor, const TextureSyncRequestArgs &args, vk::ImageSubresourceRange &viewRange, interconnect::node::SyncNode *syncNode) {
        //if (optimalRpLayout != optimalLayout) {
        //    switch (optimalRpLayout) {
        //        case vk::ImageLayout::eUndefined:
        //        case vk::ImageLayout::ePreinitialized:
        //            optimalRpLayout = optimalLayout;
        //            break;
        //        default:
        //            optimalRpLayout = vk::ImageLayout::eGeneral; // No other Vk 1.0 layouts are compatible with each other
        //            break;
        //    }
        //}

        bool syncedFromOverlap{};
        syncedFromOverlap = texture.gpu.textureUsageTracker.RequestSync(executor, texture.usageHandle, this, args.isWritten);
        // Force all overlapping buffers to sync
        texture.MarkGpuDirty(executor.usageTracker);

        std::unique_lock lock{texture.accessMutex};

        if (!syncedFromOverlap) {
            if (dirtyState == DirtyState::OtherHostDirty) {
                HostTexture *toSyncFrom{};
                for (auto &host : texture.hosts) {
                    if (host.dirtyState == DirtyState::HostDirty) {
                        toSyncFrom = &host;
                        executor.AddTextureBarrier(*toSyncFrom, {
                            .isRead = true,
                            .isWritten = false,
                            .usedStage = vk::PipelineStageFlagBits::eTransfer,
                            .usedFlags = vk::AccessFlagBits::eTransferRead
                        });
                        break;
                    }
                }

                if (!texture.syncStagingBuffer) [[unlikely]] {
                    texture.syncStagingBuffer = texture.gpu.memory.AllocateStagingBuffer(toSyncFrom->copySize);
                    executor.cycle->AttachObject(texture.syncStagingBuffer);
                }

                executor.AddStagedTextureTransferCommand(*this, args, [this, toSyncFrom](const vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &pCycle, GPU &){
                    if (toSyncFrom->needsDecompression || needsDecompression || toSyncFrom->format->vkAspect == (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil) || format->vkAspect == (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil))
                        return;

                    toSyncFrom->CopyIntoStagingBuffer(commandBuffer, texture.syncStagingBuffer);
                }, [this, toSyncFrom](const vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &pCycle, GPU &){
                    if (toSyncFrom->needsDecompression || needsDecompression || toSyncFrom->format->vkAspect == (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil) || format->vkAspect == (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil))
                        return;

                    CopyFromStagingBuffer(commandBuffer, texture.syncStagingBuffer);
                });

                toSyncFrom->trackingInfo.waitedStages |= vk::PipelineStageFlagBits::eTransfer;

                if (args.isWritten) {
                    trackingInfo.lastUsedStage = args.usedStage;
                    trackingInfo.lastUsedAccessFlag = args.usedFlags;

                    for (auto &host : texture.hosts)
                        host.dirtyState = DirtyState::OtherHostDirty;

                    texture.gpu.state.nce->TrapRegions(*texture.trapHandle, false);

                    trackingInfo.waitedStages = {};
                } else {
                    trackingInfo.lastUsedStage = vk::PipelineStageFlagBits::eTransfer;
                    trackingInfo.lastUsedAccessFlag = vk::AccessFlagBits::eTransferWrite;

                    trackingInfo.waitedStages = args.usedStage;
                }

                dirtyState = DirtyState::HostDirty;
            } else if (dirtyState == DirtyState::GuestDirty) {
                executor.AddTextureTransferCommand(*this, args, [this](const vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &pCycle, GPU &){
                    texture.SynchronizeHostInline(commandBuffer, pCycle, *this);
                });

                if (args.isWritten) {
                    trackingInfo.lastUsedStage = args.usedStage;
                    trackingInfo.lastUsedAccessFlag = args.usedFlags;

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
                    trackingInfo.waitedStages = args.usedStage;
                }
            } else {
                if (args.isWritten) {
                    if (!syncNode || !usedInRP)
                        executor.AddTextureBarrier(*this, args);

                    if (syncNode && !usedInRP)
                        usedInRP = true;

                    texture.gpu.state.nce->TrapRegions(*texture.trapHandle, false);

                    for (auto &host : texture.hosts)
                        host.dirtyState = DirtyState::OtherHostDirty;

                    dirtyState = DirtyState::HostDirty;

                    trackingInfo.lastUsedStage = args.usedStage;
                    trackingInfo.lastUsedAccessFlag = args.usedFlags;

                    trackingInfo.waitedStages = {};
                } else if (!(trackingInfo.waitedStages & args.usedStage)) {
                    if (!syncNode || !usedInRP)
                        executor.AddTextureBarrier(*this, args);

                    if (syncNode && !usedInRP)
                        usedInRP = true;

                    trackingInfo.waitedStages |= args.usedStage;
                }
            }
        } else {
            executor.AddOutsideRpCommand([this, args](const vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &, GPU &) {
                commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, args.usedStage, {}, {}, {}, vk::ImageMemoryBarrier{
                    .image = backing.vkImage,
                    .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .dstAccessMask = args.usedFlags,
                    .oldLayout = GetLayout(),
                    .newLayout = GetLayout(),
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .subresourceRange = {
                        .aspectMask = format->vkAspect,
                        .levelCount = texture.guest.levelCount,
                        .layerCount = texture.guest.layerCount,
                    }
                });
            });

            if (args.isWritten) {
                trackingInfo.lastUsedStage = args.usedStage;
                trackingInfo.lastUsedAccessFlag = args.usedFlags;
                trackingInfo.waitedStages = {};

                texture.gpu.state.nce->TrapRegions(*texture.trapHandle, false);
            } else {
                trackingInfo.lastUsedStage = vk::PipelineStageFlagBits::eTransfer;
                trackingInfo.lastUsedAccessFlag = vk::AccessFlagBits::eTransferWrite;
                trackingInfo.waitedStages = args.usedStage;

                texture.gpu.state.nce->TrapRegions(*texture.trapHandle, true);
            }
        }
    }

    void HostTexture::CopyFromStagingBuffer(const vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<memory::StagingBuffer> &stagingBuffer) {
        auto bufferImageCopies{GetBufferImageCopies()};
        commandBuffer.copyBufferToImage(stagingBuffer->vkBuffer, backing.vkImage, GetLayout(), vk::ArrayProxy(static_cast<u32>(bufferImageCopies.size()), bufferImageCopies.data()));
    }

    void HostTexture::CopyIntoStagingBuffer(const vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<memory::StagingBuffer> &stagingBuffer) {
        auto bufferImageCopies{GetBufferImageCopies()};
        commandBuffer.copyImageToBuffer(backing.vkImage, GetLayout(), stagingBuffer->vkBuffer, vk::ArrayProxy(static_cast<u32>(bufferImageCopies.size()), bufferImageCopies.data()));
    }

    void HostTexture::CopyToGuest(u8 *hostBuffer) {
        u8 *guestOutput{texture.mirror.data()};

        if (guest.levelCount == 1) {
            for (size_t layer{}; layer < guest.layerCount; ++layer) {
                if (guest.tileConfig.mode == texture::TileMode::Block)
                    texture::CopyLinearToBlockLinear(copyLayouts[0].dimensions,
                                                     guestFormat->blockWidth, guestFormat->blockHeight, guestFormat->bpb,
                                                     copyLayouts[0].blockHeight, copyLayouts[0].blockDepth, guest.tileConfig.sparseBlockWidth,
                                                     hostBuffer, guestOutput
                    );
                else if (guest.tileConfig.mode == texture::TileMode::Pitch)
                    texture::CopyLinearToPitchLinear(guest, hostBuffer, guestOutput);
                else if (guest.tileConfig.mode == texture::TileMode::Linear)
                    std::memcpy(guestOutput, hostBuffer, copyLayouts[0].linearSize);
                guestOutput += guest.layerStride;
                hostBuffer += guest.linearLayerStride;
            }
        } else if (guest.levelCount > 1 && guest.tileConfig.mode == texture::TileMode::Block) {
            // We need to copy into the Tegra X1 layout holds all mip levels for a given layer while the input buffer has all layers for a given mip level
            // Note: See SynchronizeHostImpl for additional comments
            u8 *input{hostBuffer}, *output{guestOutput};
            for (const auto &level : copyLayouts) {
                u32 inputOffset{};
                for (size_t layer{}, layerOffset{}; layer < guest.layerCount; ++layer, inputOffset += level.linearSize, layerOffset += guest.layerStride) {
                    texture::CopyLinearToBlockLinear(level.dimensions,
                                                     guestFormat->blockWidth, guestFormat->blockHeight, guestFormat->bpb,
                                                     level.blockHeight, level.blockDepth, guest.tileConfig.sparseBlockWidth,
                                                     input + inputOffset, output + layerOffset);
                }
                input += level.linearSize * guest.layerCount;
                output += level.blockLinearSize;
            }

        } else if (guest.levelCount != 0) [[unlikely]] {
            throw exception("Mipmapped textures with tiling mode '{}' aren't supported", static_cast<i32>(tiling));
        }
    }

    texture::Format ConvertHostCompatibleFormat(texture::Format format, const TraitManager &traits) {
        auto &bcnSupport{traits.bcnSupport};
        if (bcnSupport.all())
            return format;

        switch (format->vkFormat) {
            case vk::Format::eBc1RgbaUnormBlock:
                return bcnSupport[0] ? format : format::R8G8B8A8Unorm;
            case vk::Format::eBc1RgbaSrgbBlock:
                return bcnSupport[0] ? format : format::R8G8B8A8Srgb;

            case vk::Format::eBc2UnormBlock:
                return bcnSupport[1] ? format : format::R8G8B8A8Unorm;
            case vk::Format::eBc2SrgbBlock:
                return bcnSupport[1] ? format : format::R8G8B8A8Srgb;

            case vk::Format::eBc3UnormBlock:
                return bcnSupport[2] ? format : format::R8G8B8A8Unorm;
            case vk::Format::eBc3SrgbBlock:
                return bcnSupport[2] ? format : format::R8G8B8A8Srgb;

            case vk::Format::eBc4UnormBlock:
                return bcnSupport[3] ? format : format::R8Unorm;
            case vk::Format::eBc4SnormBlock:
                return bcnSupport[3] ? format : format::R8Snorm;

            case vk::Format::eBc5UnormBlock:
                return bcnSupport[4] ? format : format::R8G8Unorm;
            case vk::Format::eBc5SnormBlock:
                return bcnSupport[4] ? format : format::R8G8Snorm;

            case vk::Format::eBc6HUfloatBlock:
            case vk::Format::eBc6HSfloatBlock:
                return bcnSupport[5] ? format : format::R16G16B16A16Float; // This is a signed 16-bit FP format, we don't have an unsigned 16-bit FP format

            case vk::Format::eBc7UnormBlock:
                return bcnSupport[6] ? format : format::R8G8B8A8Unorm;
            case vk::Format::eBc7SrgbBlock:
                return bcnSupport[6] ? format : format::R8G8B8A8Srgb;

            default:
                return format;
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

    bool HostTexture::ValidateRenderPassUsage(u32 renderPassIndex, texture::RenderPassUsage renderPassUsage) {
        return (lastRenderPassUsage == renderPassUsage || lastRenderPassIndex != renderPassIndex || lastRenderPassUsage == texture::RenderPassUsage::None) && !RequiresRPBreak(renderPassUsage == texture::RenderPassUsage::Descriptor);
    }

    void HostTexture::UpdateRenderPassUsage(u32 renderPassIndex, texture::RenderPassUsage renderPassUsage) {
        lastRenderPassUsage = renderPassUsage;
        lastRenderPassIndex = renderPassIndex;
    }

    texture::RenderPassUsage HostTexture::GetLastRenderPassUsage() {
        return lastRenderPassUsage;
    }

    bool HostTexture::RequiresRPBreak(bool isDescriptor) {
        if (isDescriptor)
            return usedInTP || texture.gpu.textureUsageTracker.ShouldSyncHost(texture.usageHandle);
        else
            return (usedInTP && (dirtyState == DirtyState::OtherHostDirty || dirtyState == DirtyState::GuestDirty)) || texture.gpu.textureUsageTracker.ShouldSyncHost(texture.usageHandle);
    }

    HostTexture::HostTexture(Texture &texture, TextureViewRequestInfo &info, vk::ImageType imageType, bool mutableFormat)
        : texture{texture},
          guest{texture.guest},
          dimensions{info.imageDimensions},
          sampleCount{info.sampleCount},
          dirtyState{DirtyState::GuestDirty},
          format{ConvertHostCompatibleFormat(info.viewFormat, texture.gpu.traits)},
          guestFormat{info.viewFormat},
          copyLayouts{texture::CalculateMipLayout(
              info.imageDimensions,
              format->blockHeight, format->blockWidth, format->bpb,
              texture.guest.tileConfig.blockHeight, texture.guest.tileConfig.blockDepth, texture.guest.tileConfig.sparseBlockWidth,
              texture.guest.levelCount)},
          needsDecompression{info.viewFormat != format},
          imageType{imageType},
          flags{mutableFormat ? vk::ImageCreateFlagBits::eMutableFormat | vk::ImageCreateFlagBits::eExtendedUsage : vk::ImageCreateFlags{}},
          usage{vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled},
          copySize{static_cast<u32>(texture::CalculateLinearLayerStride(copyLayouts) * guest.layerCount)} {
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
                LOGE("Requested format doesn't support being used as a storage image: {}", vk::to_string(info.viewFormat->vkFormat));
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

    void HostTexture::Access(const vk::raii::CommandBuffer &commandBuffer, const ExecutorTrackingInfo &trackingInfo, vk::PipelineStageFlags dstStage, vk::AccessFlags dstAccess, bool forWrite) {
        if (forWrite) {
            commandBuffer.pipelineBarrier(trackingInfo.lastUsedStage | trackingInfo.waitedStages, dstStage, {}, {}, {}, vk::ImageMemoryBarrier{
                .image = backing.vkImage,
                .srcAccessMask = trackingInfo.lastUsedAccessFlag,
                .dstAccessMask = dstAccess,
                .oldLayout = GetLayout(),
                .newLayout = GetLayout(),
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .subresourceRange = {
                    .aspectMask = format->vkAspect,
                    .levelCount = texture.guest.levelCount,
                    .layerCount = texture.guest.layerCount
                }
            });
        } else {
            commandBuffer.pipelineBarrier(trackingInfo.lastUsedStage, dstStage, {}, {}, {}, vk::ImageMemoryBarrier{
                .image = backing.vkImage,
                .srcAccessMask = trackingInfo.lastUsedAccessFlag,
                .dstAccessMask = dstAccess,
                .oldLayout = GetLayout(),
                .newLayout = GetLayout(),
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .subresourceRange = {
                    .aspectMask = format->vkAspect,
                    .levelCount = texture.guest.levelCount,
                    .layerCount = texture.guest.layerCount
                }
            });
        }
    }

    void HostTexture::AccessForTransfer(const vk::raii::CommandBuffer &commandBuffer, const ExecutorTrackingInfo &trackingInfo, bool forWrite) {
        if (forWrite) {
            commandBuffer.pipelineBarrier(trackingInfo.lastUsedStage | trackingInfo.waitedStages, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, vk::ImageMemoryBarrier{
                .image = backing.vkImage,
                .srcAccessMask = trackingInfo.lastUsedAccessFlag,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                .oldLayout = GetLayout(),
                .newLayout = GetLayout(),
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .subresourceRange = {
                    .aspectMask = format->vkAspect,
                    .levelCount = texture.guest.levelCount,
                    .layerCount = texture.guest.layerCount
                }
            });
        } else {
            commandBuffer.pipelineBarrier(trackingInfo.lastUsedStage, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, vk::ImageMemoryBarrier{
                .image = backing.vkImage,
                .srcAccessMask = trackingInfo.lastUsedAccessFlag,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                .oldLayout = GetLayout(),
                .newLayout = GetLayout(),
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .subresourceRange = {
                    .aspectMask = format->vkAspect,
                    .levelCount = texture.guest.levelCount,
                    .layerCount = texture.guest.layerCount
                }
            });
        }
    }
}
