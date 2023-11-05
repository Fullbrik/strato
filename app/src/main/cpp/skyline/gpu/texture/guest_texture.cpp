// SPDX-License-Identifier: MPL-2.0
// Copyright © 2023 Skyline Team and Contributors (https://github.com/skyline-emu/)

#include "guest_texture.h"

#include <utility>
#include "layout.h"

namespace skyline::gpu {
    namespace texture {
        size_t CalculateLinearLayerStride(const std::vector<texture::MipLevelLayout> &mipLayouts) {
            size_t layerStride{};
            for (const auto &level : mipLayouts)
                layerStride += level.linearSize;
            return layerStride;
        }

        size_t CalculateUnalignedLayerStride(const std::vector<texture::MipLevelLayout> &mipLayouts) {
            size_t layerStride{};
            for (const auto &level : mipLayouts)
                layerStride += level.blockLinearSize;
            return layerStride;
        }

        u32 CalculateLayerStride(texture::Dimensions dimensions, texture::Format format, texture::TileConfig tileConfig, u32 levelCount, u32 layerCount) {
            switch (tileConfig.mode) {
                case texture::TileMode::Linear:
                    return static_cast<u32>(format->GetSize(dimensions));
                case texture::TileMode::Pitch:
                    return util::DivideCeil<u32>(dimensions.height, format->blockHeight) * tileConfig.pitch * dimensions.depth;
                case texture::TileMode::Block:
                    return static_cast<u32>(texture::GetBlockLinearLayerSize(dimensions, format->blockHeight, format->blockWidth, format->bpb, tileConfig.blockHeight, tileConfig.blockDepth, tileConfig.sparseBlockWidth, levelCount, layerCount > 1));
            }
        }
    }

    std::optional<texture::TextureCopies> GuestTexture::CalculateOffsetlessOverlap(GuestTexture &other, texture::Format format, texture::Format otherFormat) {
        texture::TextureCopies result{};

        if (tileConfig != other.tileConfig)
            return CalculateReinterpretCopy(other, format, otherFormat);

        if (levelCount == other.levelCount) {
            if ((util::AlignUp(mipLayouts[0].dimensions.width, 64U) != util::AlignUp(other.mipLayouts[0].dimensions.width, 64U) ||
                util::AlignUp(mipLayouts[0].dimensions.height, 8U * mipLayouts[0].blockHeight) != util::AlignUp(other.mipLayouts[0].dimensions.height, 8U * other.mipLayouts[0].blockHeight))
                && levelCount > 1) {
                return CalculateReinterpretCopy(other, format, otherFormat);
            } else {
                u32 level{};
                for (auto &mip : mipLayouts) {
                    texture::Dimensions minDim{std::min(mip.dimensions.width, other.mipLayouts[level].dimensions.width), std::min(mip.dimensions.height, other.mipLayouts[level].dimensions.height), std::min(mip.dimensions.depth, other.mipLayouts[level].dimensions.depth)};

                    result.toStaging.push_back(vk::BufferImageCopy{
                        .imageSubresource = {
                            .layerCount = std::min(layerCount, other.layerCount),
                            .aspectMask = otherFormat->vkAspect,
                            .mipLevel = level
                        },
                        .bufferOffset = result.stagingBufferSize,
                        .imageExtent = otherFormat->GetDimensionsFromBytes(minDim)
                    });
                    result.fromStaging.push_back(vk::BufferImageCopy{
                        .imageSubresource = {
                            .layerCount = std::min(layerCount, other.layerCount),
                            .aspectMask = format->vkAspect,
                            .mipLevel = level
                        },
                        .bufferOffset = result.stagingBufferSize,
                        .imageExtent = format->GetDimensionsFromBytes(minDim)
                    });
                    result.stagingBufferSize += minDim.width * minDim.height * minDim.depth;
                    ++level;
                }

                return {std::move(result)};
            }
        } else {
            if (layerCount == 1 && other.layerCount == 1) {
                if ((util::AlignUp(mipLayouts[0].dimensions.width, 64U) != util::AlignUp(other.mipLayouts[0].dimensions.width, 64U) ||
                    util::AlignUp(mipLayouts[0].dimensions.height, 8U * mipLayouts[0].blockHeight) != util::AlignUp(other.mipLayouts[0].dimensions.height, 8U * other.mipLayouts[0].blockHeight))
                    && levelCount > 1)
                    return CalculateReinterpretCopy(other, format, otherFormat);

                u32 minlevel{std::min(levelCount, other.levelCount)};
                for (u32 level{}; level < minlevel; ++level) {
                    texture::Dimensions minDim{std::min(mipLayouts[level].dimensions.width, other.mipLayouts[level].dimensions.width), std::min(mipLayouts[level].dimensions.height, other.mipLayouts[level].dimensions.height), std::min(mipLayouts[level].dimensions.depth, other.mipLayouts[level].dimensions.depth)};

                    result.toStaging.push_back(vk::BufferImageCopy{
                        .imageSubresource = {
                            .layerCount = 1,
                            .aspectMask = otherFormat->vkAspect,
                            .mipLevel = level
                        },
                        .bufferOffset = result.stagingBufferSize,
                        .imageExtent = otherFormat->GetDimensionsFromBytes(minDim)
                    });
                    result.fromStaging.push_back(vk::BufferImageCopy{
                        .imageSubresource = {
                            .layerCount = 1,
                            .aspectMask = format->vkAspect,
                            .mipLevel = level
                        },
                        .bufferOffset = result.stagingBufferSize,
                        .imageExtent = format->GetDimensionsFromBytes(minDim)
                    });
                    result.stagingBufferSize += minDim.width * minDim.height * minDim.depth;
                }

                return {std::move(result)};
            } else {
                return CalculateReinterpretCopy(other, format, otherFormat);
            }
        }
    }

    GuestTexture::GuestTexture(texture::Mappings mappings, texture::Dimensions sampleDimensions, texture::Format format, texture::TileConfig tileConfig, u32 levelCount, u32 layerCount, u32 layerStride)
        : mappings{std::move(mappings)},
          tileConfig{tileConfig},
          levelCount{levelCount},
          layerCount{layerCount},
          layerStride{layerStride},
          size{layerStride * layerCount},
          mipLayouts{
              texture::CalculateMipLayout(
                  format->GetDimensionsInBytes(sampleDimensions),
                  1, 1, 1,
                  tileConfig.blockHeight, tileConfig.blockDepth, tileConfig.sparseBlockWidth,
                  levelCount
              )
          } {
        layerSize = tileConfig.mode == texture::TileMode::Block ? static_cast<u32>(texture::CalculateUnalignedLayerStride(mipLayouts)) : layerStride;
        linearLayerStride = CalculateLinearLayerStride(mipLayouts);
        linearSize = linearLayerStride * layerCount;
        if (tileConfig.mode == texture::TileMode::Block) {
            tileConfig.blockHeight = static_cast<u8>(mipLayouts[0].blockHeight);
            tileConfig.blockDepth = static_cast<u8>(mipLayouts[0].blockDepth);
        }
    }

    std::optional<vk::ImageSubresourceRange> GuestTexture::CalculateSubresource(texture::TileConfig pTileConfig, u32 offset, texture::Dimensions pTextureSizes, u32 pLevelCount, u32 pLayerCount, u32 pLayerStride, vk::ImageAspectFlags aspectMask) {
        if (offset >= size)
            return std::nullopt;

        if (pTileConfig != tileConfig)
            return std::nullopt; // The tiling mode is not compatible, this is a hard requirement

        if (layerCount > 1 || pLayerCount > 1) {
            if (pLayerStride != layerStride)
                return std::nullopt; // The layer stride is not compatible, if the stride doesn't match then layers won't be aligned
        }

        u32 layer{offset / layerStride}, level{};
        if (tileConfig.mode == texture::TileMode::Block) {
            u32 layerOffset{layer * layerStride}, levelOffset{};
            while (level < levelCount && (layerOffset + levelOffset) < offset) {
                levelOffset += mipLayouts[level].blockLinearSize;
                ++level;
            }

            if (offset - layerOffset != levelOffset)
                return std::nullopt; // The offset is not aligned to the start of a level

            if (util::AlignUp(mipLayouts[level].dimensions.width, 64U) != util::AlignUp(pTextureSizes.width, 64U) || util::AlignUp(mipLayouts[level].dimensions.height, 8U * mipLayouts[level].blockHeight) != util::AlignUp(pTextureSizes.height, 8 * mipLayouts[level].blockHeight))
                return std::nullopt; // The texture has incompatible dimensions
        } else {
            if (levelCount != 1) [[unlikely]]
                LOGE("Mipmapped pitch textures are not supported! levelCount: {}", levelCount);

            if (util::AlignUp(mipLayouts[level].dimensions.width, 64) != util::AlignUp(pTextureSizes.width, 64) || mipLayouts[0].dimensions.height != pTextureSizes.height)
                return std::nullopt; // The texture has incompatible dimensions
        }

        if (mipLayouts[level].dimensions.depth != pTextureSizes.depth)
            return std::nullopt; // The texture has incompatible dimensions

        if (layer + pLayerCount > layerCount || level + pLevelCount > levelCount)
            return std::nullopt; // The layer/level count is out of bounds

        return vk::ImageSubresourceRange{
            .aspectMask = aspectMask,
            .baseMipLevel = level,
            .levelCount = pLevelCount,
            .baseArrayLayer = layer,
            .layerCount = pLayerCount,
        };
    }

    std::optional<texture::TextureCopies> GuestTexture::CalculateCopy(GuestTexture &other, texture::Format format, texture::Format otherFormat) {
        if (tileConfig.mode != other.tileConfig.mode) [[unlikely]] {
            LOGW("Overlaps of different tile modes are not supported!");
            return std::nullopt;
        }

        if (format->IsCompressed() || otherFormat->IsCompressed()) [[unlikely]] {
            LOGW("Overlaps of compressed textures are not supported");
            return std::nullopt;
        } else if ((format->vkAspect == (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)) || (otherFormat->vkAspect == (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil))) {
            LOGW("Overlaps of depth stencil textures are unimplemented");
            return std::nullopt;
        }

        u32 dstOffset{OffsetFrom(other.mappings)}, srcOffset{other.OffsetFrom(mappings)};

        if (dstOffset == 0) {
            return CalculateOffsetlessOverlap(other, format, otherFormat);
        } else {
            bool flip{dstOffset == UINT32_MAX};
            u32 &actualOffset{flip ? srcOffset : dstOffset};

            if (actualOffset == UINT32_MAX)
                return std::nullopt;

            return CalculateReinterpretCopy(other, format, otherFormat);
        }
    }

    std::optional<texture::TextureCopies> GuestTexture::CalculateReinterpretCopy(GuestTexture &other, texture::Format format, texture::Format otherFormat) {
        u32 srcOffset{OffsetFrom(other.mappings)}, dstOffset{other.OffsetFrom(mappings)};
        bool flip{srcOffset == UINT32_MAX}; //!< If true then the start of the src/other texture is before the dst/this texture. Otherwise it's false

        texture::TextureCopies results{};

        if (srcOffset == UINT32_MAX && dstOffset == UINT32_MAX)
            return std::nullopt;

        u32 dstLayer{}, dstLevel{};
        u32 srcLayer{}, srcLevel{};

        if (flip)
            dstLayer = dstOffset / layerStride;
        else
            srcLayer = srcOffset / other.layerStride;

        if (tileConfig.mode == texture::TileMode::Block) {
            LOGI("Src: {}x{}x{} format: {}, levels: {}, layers: {}, address: {}-{}, blockHeight: {}, blockDepth: {}", other.mipLayouts[0].dimensions.width / otherFormat->bpb, other.mipLayouts[0].dimensions.height, other.mipLayouts[0].dimensions.depth, vk::to_string(otherFormat->vkFormat), other.levelCount, other.layerCount, fmt::ptr(other.mappings.front().data()), fmt::ptr(other.mappings.front().end().base()), other.mipLayouts[0].blockHeight, other.mipLayouts[0].blockDepth);
            LOGI("Dst: {}x{}x{} format: {}, levels: {}, layers: {}, address: {}-{}, blockHeight: {}, blockDepth: {}", mipLayouts[0].dimensions.width / format->bpb, mipLayouts[0].dimensions.height, mipLayouts[0].dimensions.depth, vk::to_string(format->vkFormat), levelCount, layerCount, fmt::ptr(mappings.front().data()), fmt::ptr(mappings.front().end().base()), mipLayouts[0].blockHeight, mipLayouts[0].blockDepth);

            // Blocklinear texture addresses must be aligned to the size of 1 Gob (512 bytes), however workarounds in the blit engine can cause textures to not fit this requirement
            if (!util::IsAligned(mappings.front().data(), 64U * 8U) || !util::IsAligned(other.mappings.front().data(), 64U * 8U))
                return std::nullopt;

            // FIXME: Figure out a more optimised (and less ugly) approach to this
            u32 inMipOffset{};
            if (flip) {
                u32 layerOffset{dstLayer * layerStride}, levelOffset{};

                while (dstLevel < levelCount && (layerOffset + levelOffset) < dstOffset) {
                    levelOffset += mipLayouts[dstLevel].blockLinearSize;
                    ++dstLevel;
                }

                if (dstOffset != (levelOffset + layerOffset)) {
                    --dstLevel;
                    inMipOffset = static_cast<u32>(mipLayouts[dstLevel].blockLinearSize) - ((layerOffset + levelOffset) - dstOffset);

                    if (dstOffset > (layerOffset + levelOffset)) {
                        LOGW("Layer offsets are unimplemented! dstOffset: 0x{:X}, mipOffset: 0x{:X}", dstOffset, layerOffset + levelOffset);
                        return std::nullopt;
                    }

                    if (dstLevel == (levelCount - 1) && inMipOffset >= mipLayouts[dstLevel].blockLinearSize) {
                        LOGE("Layer offsets broke! inMipOffset: 0x{:X}, dstOffset: 0x{:X}", inMipOffset, dstOffset);
                        return std::nullopt;
                    }

                    if (inMipOffset >= mipLayouts[dstLevel].blockLinearSize)
                        LOGE("Level offsets broke! inMipOffset: 0x{:X}, dstOffset: 0x{:X}", inMipOffset, dstOffset);
                }
            } else {
                u32 layerOffset{srcLayer * other.layerStride}, levelOffset{};

                while (srcLevel < other.levelCount && (layerOffset + levelOffset) < srcOffset) {
                    levelOffset += other.mipLayouts[srcLevel].blockLinearSize;
                    ++srcLevel;
                }

                if (srcOffset != (levelOffset + layerOffset)) {
                    --srcLevel;
                    inMipOffset = static_cast<u32>(other.mipLayouts[srcLevel].blockLinearSize) - ((layerOffset + levelOffset) - srcOffset);

                    if (srcOffset > (layerOffset + levelOffset)) {
                        LOGW("Layer offsets are unimplemented! srcOffset: 0x{:X}, mipOffset: 0x{:X}", srcOffset, layerOffset + levelOffset);
                        return std::nullopt;
                    }

                    if (srcLevel == (other.levelCount - 1) && inMipOffset >= other.mipLayouts[srcLevel].blockLinearSize) {
                        LOGE("Layer offsets broke! inMipOffset: 0x{:X}, srcOffset: 0x{:X}", inMipOffset, srcOffset);
                        return std::nullopt;
                    }

                    if (inMipOffset >= other.mipLayouts[srcLevel].blockLinearSize)
                        LOGE("Level offsets broke! inMipOffset: 0x{:X}, srcOffset: 0x{:X}", inMipOffset, srcOffset);
                }
            }

            u32 dstSliceSize, dstBlockSize;
            u32 dstGob{}, dstSlice{}, dstBlock{}, dstRob{}, dstMob{};
            u32 dstRobWidthBlocks, dstAlignedRobWidthBlocks, dstRobCount, dstMobCount;
            vk::Offset3D dstImageOffset{};
            vk::Extent3D dstGobExtent{64U / format->bpb, 8, 1};

            u32 srcSliceSize, srcBlockSize;
            u32 srcGob{}, srcSlice{}, srcBlock{}, srcRob{}, srcMob{};
            u32 srcRobWidthBlocks, srcAlignedRobWidthBlocks, srcRobCount, srcMobCount;
            vk::Offset3D srcImageOffset{};
            vk::Extent3D srcGobExtent{64U / otherFormat->bpb, 8, 1};

            for (; dstLayer < layerCount && srcLayer < other.layerCount;) {
                for (; dstLevel < levelCount && srcLevel < other.levelCount;) {
                    texture::Dimensions &dstImageExtent{mipLayouts[dstLevel].dimensions}; //!< The dimensions of the destination texture in bytes

                    dstSliceSize = 64U * 8U * static_cast<u32>(mipLayouts[dstLevel].blockHeight);
                    dstBlockSize = dstSliceSize * static_cast<u32>(mipLayouts[dstLevel].blockDepth);

                    dstRobWidthBlocks = util::DivideCeil(mipLayouts[dstLevel].dimensions.width, 64U);
                    dstAlignedRobWidthBlocks = util::AlignUp(dstRobWidthBlocks, tileConfig.sparseBlockWidth);
                    dstRobCount = util::DivideCeil(mipLayouts[dstLevel].dimensions.height, 8U * static_cast<u32>(mipLayouts[dstLevel].blockHeight));
                    dstMobCount = util::DivideCeil(mipLayouts[dstLevel].dimensions.depth, static_cast<u32>(mipLayouts[dstLevel].blockDepth));

                    texture::Dimensions &srcImageExtent{other.mipLayouts[srcLevel].dimensions}; //!< The dimensions of the source texture in bytes

                    srcSliceSize = 64U * 8U * static_cast<u32>(other.mipLayouts[srcLevel].blockHeight);
                    srcBlockSize = srcSliceSize * static_cast<u32>(other.mipLayouts[srcLevel].blockDepth);

                    srcRobWidthBlocks = util::DivideCeil(other.mipLayouts[srcLevel].dimensions.width, 64U);
                    srcAlignedRobWidthBlocks = util::AlignUp(srcRobWidthBlocks, other.tileConfig.sparseBlockWidth);
                    srcRobCount = util::DivideCeil(other.mipLayouts[srcLevel].dimensions.height, 8U * static_cast<u32>(other.mipLayouts[srcLevel].blockHeight));
                    srcMobCount = util::DivideCeil(other.mipLayouts[srcLevel].dimensions.depth, static_cast<u32>(other.mipLayouts[srcLevel].blockDepth));

                    if (inMipOffset) {
                        if (flip) {
                            dstMob = inMipOffset / (dstBlockSize * dstAlignedRobWidthBlocks * dstRobCount);
                            inMipOffset = inMipOffset % (dstBlockSize * dstAlignedRobWidthBlocks * dstRobCount);

                            dstRob = inMipOffset / (dstBlockSize * dstAlignedRobWidthBlocks);
                            inMipOffset = inMipOffset % (dstBlockSize * dstAlignedRobWidthBlocks);

                            dstBlock = inMipOffset / dstBlockSize;
                            inMipOffset = inMipOffset % dstBlockSize;

                            dstSlice = inMipOffset / dstSliceSize;
                            inMipOffset = inMipOffset % dstSliceSize;

                            dstGob = inMipOffset / (64U * 8U);

                            dstImageOffset.x = static_cast<i32>(dstBlock * 64);
                            dstImageOffset.y = static_cast<i32>((dstRob * mipLayouts[dstLevel].blockHeight) + (dstGob * 8U));
                            dstImageOffset.z = static_cast<i32>((dstMob * mipLayouts[dstLevel].blockDepth) + dstSlice);
                        } else {
                            srcMob = inMipOffset / (srcBlockSize * srcAlignedRobWidthBlocks * srcRobCount);
                            inMipOffset = inMipOffset % (srcBlockSize * srcAlignedRobWidthBlocks * srcRobCount);

                            srcRob = inMipOffset / (srcBlockSize * srcAlignedRobWidthBlocks);
                            inMipOffset = inMipOffset % (srcBlockSize * srcAlignedRobWidthBlocks);

                            srcBlock = inMipOffset / srcBlockSize;
                            inMipOffset = inMipOffset % srcBlockSize;

                            srcSlice = inMipOffset / srcSliceSize;
                            inMipOffset = inMipOffset % srcSliceSize;

                            srcGob = inMipOffset / (64U * 8U);

                            srcImageOffset.x = static_cast<i32>(srcBlock * 64);
                            srcImageOffset.y = static_cast<i32>((srcRob * other.mipLayouts[srcLevel].blockHeight) + (srcGob * 8U));
                            srcImageOffset.z = static_cast<i32>((srcMob * other.mipLayouts[srcLevel].blockDepth) + srcSlice);
                        }

                        inMipOffset = 0;
                    }

                    for (; dstMob < dstMobCount && srcMob < srcMobCount; ) {
                        for (; dstRob < dstRobCount && srcRob < srcRobCount; ) {
                            for (; dstBlock < dstAlignedRobWidthBlocks && srcBlock < srcAlignedRobWidthBlocks; ) {
                                for (; dstSlice < mipLayouts[dstLevel].blockDepth && srcSlice < other.mipLayouts[srcLevel].blockDepth;) {
                                    for (; dstGob < mipLayouts[dstLevel].blockHeight && srcGob < other.mipLayouts[srcLevel].blockHeight; ++dstGob, dstImageOffset.y += 8, ++srcGob, srcImageOffset.y += 8) {
                                        if (dstImageOffset.z < dstImageExtent.depth && srcImageOffset.z < srcImageExtent.depth) {
                                            if (dstBlock < dstRobWidthBlocks && srcBlock < srcRobWidthBlocks && dstImageOffset.x < dstImageExtent.width && srcImageOffset.x < srcImageExtent.width) {
                                                if (dstImageOffset.y < dstImageExtent.height && srcImageOffset.y < srcImageExtent.height) {
                                                    texture::Dimensions minDim{std::min(std::min(srcImageExtent.width, static_cast<u32>(srcImageOffset.x) + 64U) - static_cast<u32>(srcImageOffset.x), std::min(dstImageExtent.width, static_cast<u32>(dstImageOffset.x) + 64U) - static_cast<u32>(dstImageOffset.x)),
                                                                               std::min(std::min(srcImageExtent.height, static_cast<u32>(srcImageOffset.y) + srcGobExtent.height) - static_cast<u32>(srcImageOffset.y), std::min(dstImageExtent.height, static_cast<u32>(dstImageOffset.y) + dstGobExtent.height) - static_cast<u32>(dstImageOffset.y)), 1U};

                                                    results.toStaging.emplace_back(vk::BufferImageCopy{
                                                        .imageSubresource = {
                                                            .aspectMask = otherFormat->vkAspect,
                                                            .layerCount = 1,
                                                            .mipLevel = srcLevel,
                                                            .baseArrayLayer = srcLayer
                                                        },
                                                        .imageExtent = {minDim.width / otherFormat->bpb, minDim.height, minDim.depth},
                                                        .bufferOffset = results.stagingBufferSize,
                                                        .imageOffset = {srcImageOffset.x / otherFormat->bpb, srcImageOffset.y, srcImageOffset.z}
                                                    });
                                                    results.fromStaging.emplace_back(vk::BufferImageCopy{
                                                        .imageSubresource = {
                                                            .aspectMask = format->vkAspect,
                                                            .layerCount = 1,
                                                            .mipLevel = dstLevel,
                                                            .baseArrayLayer = dstLayer
                                                        },
                                                        .imageExtent = {minDim.width / format->bpb, minDim.height, minDim.depth},
                                                        .bufferOffset = results.stagingBufferSize,
                                                        .imageOffset = {dstImageOffset.x / format->bpb, dstImageOffset.y, dstImageOffset.z}
                                                    });

                                                    results.stagingBufferSize += 64U * 8U;
                                                }
                                            }
                                        }
                                    }

                                    if (dstGob == mipLayouts[dstLevel].blockHeight) {
                                        dstImageOffset.y -= static_cast<i32>(mipLayouts[dstLevel].blockHeight) * 8;
                                        dstImageOffset.z += 1;
                                        dstGob = 0;
                                        ++dstSlice;
                                    }
                                    if (srcGob == other.mipLayouts[srcLevel].blockHeight) {
                                        srcImageOffset.y -= static_cast<i32>(other.mipLayouts[srcLevel].blockHeight) * 8;
                                        srcImageOffset.z += 1;
                                        srcGob = 0;
                                        ++srcSlice;
                                    }
                                }

                                if (dstSlice == mipLayouts[dstLevel].blockDepth) {
                                    dstImageOffset.z -= static_cast<i32>(mipLayouts[dstLevel].blockDepth);
                                    dstSlice = 0;
                                    dstImageOffset.x += 64;
                                    ++dstBlock;
                                }
                                if (srcSlice == other.mipLayouts[srcLevel].blockDepth) {
                                    srcImageOffset.z -= static_cast<i32>(other.mipLayouts[srcLevel].blockDepth);
                                    srcSlice = 0;
                                    srcImageOffset.x += 64;
                                    ++srcBlock;
                                }
                            }

                            if (dstBlock == dstAlignedRobWidthBlocks) {
                                dstImageOffset.x = 0;
                                dstBlock = 0;
                                dstImageOffset.y += static_cast<i32>(mipLayouts[dstLevel].blockHeight) * 8;
                                ++dstRob;
                            }
                            if (srcBlock == srcAlignedRobWidthBlocks) {
                                srcImageOffset.x = 0;
                                srcBlock = 0;
                                srcImageOffset.y += static_cast<i32>(other.mipLayouts[srcLevel].blockHeight) * 8;
                                ++srcRob;
                            }
                        }

                        if (dstRob == dstRobCount) {
                            dstImageOffset.y = 0;
                            dstRob = 0;
                            dstImageOffset.z += static_cast<i32>(mipLayouts[dstLevel].blockDepth);
                            ++dstMob;
                        }
                        if (srcRob == srcRobCount) {
                            srcImageOffset.y = 0;
                            srcRob = 0;
                            srcImageOffset.z += static_cast<i32>(other.mipLayouts[srcLevel].blockDepth);
                            ++srcMob;
                        }
                    }

                    if (dstMob == dstMobCount) {
                        dstImageOffset = vk::Offset3D{};
                        dstMob = 0;
                        ++dstLevel;
                    }
                    if (srcMob == srcMobCount) {
                        srcImageOffset = vk::Offset3D{};
                        srcMob = 0;
                        ++srcLevel;
                    }
                }

                if (dstLevel == levelCount) {
                    dstLevel = 0;
                    ++dstLayer;
                }
                if (srcLevel == other.levelCount) {
                    srcLevel = 0;
                    ++srcLayer;
                }
            }

            return {std::move(results)};
        } else {
            LOGV("Src: {}x{}x{} format: {}, levels: {}, layers: {}, address: {}-{}, pitch: {}", other.mipLayouts[0].dimensions.width / otherFormat->bpb, other.mipLayouts[0].dimensions.height, other.mipLayouts[0].dimensions.depth, vk::to_string(otherFormat->vkFormat), other.levelCount, other.layerCount, fmt::ptr(other.mappings.front().data()), fmt::ptr(other.mappings.front().end().base()), other.tileConfig.pitch);
            LOGV("Dst: {}x{}x{} format: {}, levels: {}, layers: {}, address: {}-{}, pitch: {}", mipLayouts[0].dimensions.width / format->bpb, mipLayouts[0].dimensions.height, mipLayouts[0].dimensions.depth, vk::to_string(format->vkFormat), levelCount, layerCount, fmt::ptr(mappings.front().data()), fmt::ptr(mappings.front().end().base()), tileConfig.pitch);
            LOGW("Pitch overlaps are not implemented");
            return std::nullopt;
        }
    }

    bool GuestTexture::Contains(const texture::Mappings &otherMappings) {
        if (mappings.size() == 1 && otherMappings.size() == 1) [[likely]]
            return mappings.front().contains(otherMappings.front());
        if (mappings.size() < otherMappings.size())
            return false;

        for (auto it{mappings.begin()}; std::distance(it, mappings.end()) >= otherMappings.size(); ++it) {
            if ((it->valid() && it->data() <= otherMappings.front().data() && it->end().base() == otherMappings.front().end().base()) || (!it->valid() && !otherMappings.front().valid() && it->size() >= otherMappings.front().size())) {
                auto mappingIt{std::next(otherMappings.begin())};
                auto targetIt{std::next(it)};
                for (; mappingIt != std::prev(otherMappings.end()); ++mappingIt, ++targetIt) {
                    if (mappingIt->data() != targetIt->data() || mappingIt->end().base() != targetIt->end().base())
                        break;
                }
                if (mappingIt != std::prev(otherMappings.end()))
                    continue;

                if ((++mappingIt)->data() == (++targetIt)->data() && mappingIt->size() <= targetIt->size())
                    return true;
            }
        }

        return false;
    }

    u32 GuestTexture::OffsetFrom(const texture::Mappings &otherMappings) {
        if (otherMappings.front().data() == mappings.front().data())
            return 0U;

        u32 offset{};
        if (otherMappings.front().valid() && mappings.front().valid()) {
            auto it{mappings.begin()};
            while (it != mappings.end() && !it->contains(otherMappings.front().data())) {
                offset += it->size();
                ++it;
            }
            if (it != mappings.end())
                offset += otherMappings.front().data() - it->data();
            else
                return UINT32_MAX;
        } else {
            return UINT32_MAX;
        }

        return offset;
    }
}
