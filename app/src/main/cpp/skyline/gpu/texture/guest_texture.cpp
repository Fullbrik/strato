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

        if (levelCount == other.levelCount) {
            if ((util::AlignUp(mipLayouts[0].dimensions.width, 64) != util::AlignUp(other.mipLayouts[0].dimensions.width, 64) ||
                util::AlignUp(mipLayouts[0].dimensions.height, 8 * tileConfig.blockHeight) != util::AlignUp(other.mipLayouts[0].dimensions.height, 8 * tileConfig.blockHeight))
                && levelCount > 1) {
                return CalculateReinterpretCopy(other, format, otherFormat);
            } else {
                u32 level{};
                for (auto &mip : mipLayouts) {
                    texture::Dimensions minDim{std::min(mip.dimensions.width, other.mipLayouts[level].dimensions.width), std::min(mip.dimensions.height, other.mipLayouts[level].dimensions.height), std::min(mip.dimensions.depth, other.mipLayouts[level].dimensions.depth)};

                    result.toStaging.push_back(vk::BufferImageCopy{
                        .imageSubresource = {
                            .layerCount = std::min(layerCount, other.layerCount),
                            .aspectMask = format->Aspect({vk::ComponentSwizzle::eR}),
                            .mipLevel = level
                        },
                        .bufferOffset = result.stagingBufferSize,
                        .imageExtent = format->GetDimensionsFromBytes(minDim)
                    });
                    result.fromStaging.push_back(vk::BufferImageCopy{
                        .imageSubresource = {
                            .layerCount = std::min(layerCount, other.layerCount),
                            .aspectMask = otherFormat->Aspect({vk::ComponentSwizzle::eR}),
                            .mipLevel = level
                        },
                        .bufferOffset = result.stagingBufferSize,
                        .imageExtent = otherFormat->GetDimensionsFromBytes(minDim)
                    });
                    result.stagingBufferSize += minDim.width * minDim.height * minDim.depth;
                    ++level;
                }

                return {std::move(result)};
            }
        } else {
            if (layerCount == 1 && other.layerCount == 1) {
                if ((util::AlignUp(mipLayouts[0].dimensions.width, 64) != util::AlignUp(other.mipLayouts[0].dimensions.width, 64) ||
                    util::AlignUp(mipLayouts[0].dimensions.height, 8 * tileConfig.blockHeight) != util::AlignUp(other.mipLayouts[0].dimensions.height, 8 * tileConfig.blockHeight))
                    && levelCount > 1)
                    return CalculateReinterpretCopy(other, format, otherFormat);

                u32 minlevel{std::min(levelCount, other.levelCount)};
                for (u32 level{}; level < minlevel; ++level) {
                    texture::Dimensions minDim{std::min(mipLayouts[level].dimensions.width, other.mipLayouts[level].dimensions.width), std::min(mipLayouts[level].dimensions.height, other.mipLayouts[level].dimensions.height), std::min(mipLayouts[level].dimensions.depth, other.mipLayouts[level].dimensions.depth)};

                    result.toStaging.push_back(vk::BufferImageCopy{
                        .imageSubresource = {
                            .layerCount = 1,
                            .aspectMask = format->Aspect({vk::ComponentSwizzle::eR}),
                            .mipLevel = level
                        },
                        .bufferOffset = result.stagingBufferSize,
                        .imageExtent = format->GetDimensionsFromBytes(minDim)
                    });
                    result.fromStaging.push_back(vk::BufferImageCopy{
                        .imageSubresource = {
                            .layerCount = 1,
                            .aspectMask = otherFormat->Aspect({vk::ComponentSwizzle::eR}),
                            .mipLevel = level
                        },
                        .bufferOffset = result.stagingBufferSize,
                        .imageExtent = otherFormat->GetDimensionsFromBytes(minDim)
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
                ++level;
                levelOffset += mipLayouts[level].blockLinearSize;
            }

            if (offset - layerOffset != levelOffset)
                return std::nullopt; // The offset is not aligned to the start of a level

            if (util::AlignUp(mipLayouts[level].dimensions.width, 64) != util::AlignUp(pTextureSizes.width, 64) || util::AlignUp(mipLayouts[level].dimensions.height, 8 * tileConfig.blockHeight) != util::AlignUp(pTextureSizes.height, 8 * tileConfig.blockHeight))
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

    // TODO: Finish and fixup this
    std::optional<texture::TextureCopies> GuestTexture::CalculateCopy(GuestTexture &other, texture::Format format, texture::Format otherFormat) {
        u32 offset{OffsetFrom(other.mappings)}, otherOffset{other.OffsetFrom(mappings)};

        if (tileConfig.mode != other.tileConfig.mode) [[unlikely]] {
            LOGW("Overlaps of different tile modes are not supported!");
            return std::nullopt;
        }

        if (format->IsCompressed()) {
            LOGW("Src format is compressed (not supported)");
            return std::nullopt;
        } else if (format->vkAspect == (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)) {
            LOGW("Src format is depth stencil (not implemented)");
            return std::nullopt;
        }

        if (otherFormat->IsCompressed()) {
            LOGW("Dst format is compressed (not supported)");
            return std::nullopt;
        } else if (otherFormat->vkAspect == (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)) {
            LOGW("Dst format is depth stencil (not implemented)");
            return std::nullopt;
        }

        if (offset == 0) {
            return CalculateOffsetlessOverlap(other, format, otherFormat);
        } else {
            if (format->IsCompressed() || otherFormat->IsCompressed())
                return std::nullopt;

            bool flip{offset == UINT32_MAX};
            u32 &actualOffset{flip ? otherOffset : offset};
            GuestTexture &largerGuest{flip ? other : *this};
            GuestTexture &smallerGuest{flip ? *this : other};

            if (largerGuest.size <= actualOffset)
                return std::nullopt;

            return CalculateReinterpretCopy(other, format, otherFormat);
        }
    }

    std::optional<texture::TextureCopies> GuestTexture::CalculateReinterpretCopy(GuestTexture &other, texture::Format format, texture::Format otherFormat) {
        u32 thisOffset{OffsetFrom(other.mappings)}, otherOffsetFromThis{other.OffsetFrom(mappings)};
        bool flip{thisOffset == UINT32_MAX};

        texture::TextureCopies results{};

        u32 layer{}, level{};
        u32 otherLayer{}, otherLevel{};

        if (flip)
            otherLayer = otherOffsetFromThis / other.layerStride;
        else
            layer = thisOffset / layerStride;

        if (tileConfig.mode == texture::TileMode::Block) {
            LOGV("Src: {}x{}x{} format: {}, levels: {}, layers: {}, address: 0x{}-0x{}, blockHeight: {}, blockDepth: {}", other.mipLayouts[0].dimensions.width / otherFormat->bpb, other.mipLayouts[0].dimensions.height, other.mipLayouts[0].dimensions.depth, vk::to_string(otherFormat->vkFormat), other.levelCount, other.layerCount, fmt::ptr(other.mappings.front().data()), fmt::ptr(other.mappings.front().end().base()), other.mipLayouts[0].blockHeight, other.mipLayouts[0].blockDepth);
            LOGV("Dst: {}x{}x{} format: {}, levels: {}, layers: {}, address: 0x{}-0x{}, blockHeight: {}, blockDepth: {}", mipLayouts[0].dimensions.width / format->bpb, mipLayouts[0].dimensions.height, mipLayouts[0].dimensions.depth, vk::to_string(format->vkFormat), levelCount, layerCount, fmt::ptr(mappings.front().data()), fmt::ptr(mappings.front().end().base()), mipLayouts[0].blockHeight, mipLayouts[0].blockDepth);

            ///!< FIXME: Figure out a more optimised (and less ugly) approach to this
            u32 inMipOffset{};
            if (flip) {
                u32 layerOffset{otherLayer * other.layerStride}, levelOffset{};

                while (otherLevel < other.levelCount && (layerOffset + levelOffset) < otherOffsetFromThis) {
                    ++otherLevel;
                    levelOffset += other.mipLayouts[otherLevel].blockLinearSize;
                }

                if ((otherOffsetFromThis - layerOffset) != levelOffset) {
                    inMipOffset = static_cast<u32>(other.mipLayouts[otherLevel].blockLinearSize) - ((layerOffset + levelOffset) - otherOffsetFromThis);
                    --otherLevel;
                }
            } else {
                u32 layerOffset{layer * layerStride}, levelOffset{};

                while (level < levelCount && (layerOffset + levelOffset) < thisOffset) {
                    ++level;
                    levelOffset += mipLayouts[level].blockLinearSize;
                }

                if ((thisOffset - layerOffset) != levelOffset) {
                    inMipOffset = static_cast<u32>(mipLayouts[level].blockLinearSize) - ((layerOffset + levelOffset) - thisOffset);
                    --level;
                }
            }


            u32 sliceSize, blockSize;
            u32 currentGob{}, currentSlice{}, currentBlock{}, currentRob{}, currentMob{};
            u32 robWidthBlocks, alignedRobWidthBlocks, robCount, mobCount;
            u32 blockCount;
            u32 offsetInBlocks, offsetInSlices, offsetInGobs;
            vk::Offset3D imageOffset{};
            vk::Extent3D gobExtent{64U / format->bpb, 8, 1};

            u32 otherSliceSize, otherBlockSize;
            u32 otherCurrentGob{}, otherCurrentSlice{}, otherCurrentBlock{}, otherCurrentRob{}, otherCurrentMob{};
            u32 otherRobWidthBlocks, otherAlignedRobWidthBlocks, otherRobCount, otherMobCount;
            u32 otherBlockCount;
            vk::Offset3D otherImageOffset{};
            vk::Extent3D otherGobExtent{64U / otherFormat->bpb, 8, 1};

            for (; layer < layerCount && otherLayer < other.layerCount;) {
                for (; level < levelCount && otherLevel < other.levelCount;) {
                    vk::Extent3D imageExtent{mipLayouts[level].dimensions.width / format->bpb, mipLayouts[level].dimensions.height, 1};

                    sliceSize = 64U * 8U * static_cast<u32>(mipLayouts[level].blockHeight);
                    blockSize = sliceSize * static_cast<u32>(mipLayouts[level].blockDepth);

                    robWidthBlocks = util::DivideCeil(mipLayouts[level].dimensions.width, 64U);
                    alignedRobWidthBlocks = util::AlignUp(robWidthBlocks, tileConfig.sparseBlockWidth);
                    robCount = util::DivideCeil(mipLayouts[level].dimensions.height, 8U * static_cast<u32>(mipLayouts[level].blockHeight));
                    mobCount = util::DivideCeil(mipLayouts[level].dimensions.depth, static_cast<u32>(mipLayouts[level].blockDepth));

                    blockCount = robWidthBlocks * robCount * mobCount;

                    vk::Extent3D otherImageExtent{other.mipLayouts[otherLevel].dimensions.width / format->bpb, other.mipLayouts[otherLevel].dimensions.height, 1};

                    otherSliceSize = 64U * 8U * static_cast<u32>(other.mipLayouts[otherLevel].blockHeight);
                    otherBlockSize = otherSliceSize * static_cast<u32>(other.mipLayouts[otherLevel].blockDepth);

                    otherRobWidthBlocks = util::DivideCeil(other.mipLayouts[otherLevel].dimensions.width, 64U);
                    otherAlignedRobWidthBlocks = util::AlignUp(otherRobWidthBlocks, other.tileConfig.sparseBlockWidth);
                    otherRobCount = util::DivideCeil(other.mipLayouts[otherLevel].dimensions.height, 8U * static_cast<u32>(other.mipLayouts[otherLevel].blockHeight));
                    otherMobCount = util::DivideCeil(other.mipLayouts[otherLevel].dimensions.depth, static_cast<u32>(other.mipLayouts[otherLevel].blockDepth));

                    otherBlockCount = otherRobWidthBlocks * otherRobCount * otherMobCount;

                    //if (inMipOffset) {
                    //    currentMob = inMipOffset / (blockSize * alignedRobWidthBlocks * robCount);
                    //    currentRob = (inMipOffset / (blockSize * alignedRobWidthBlocks)) % robCount * alignedRobWidthBlocks;
                    //    offsetInBlocks = inMipOffset / blockSize;
                    //    offsetInSlices = (inMipOffset % blockSize) / sliceSize;
//
                    //    inMipOffset = 0;
                    //}

                    for (; currentMob < mobCount && otherCurrentMob < otherMobCount; ) {
                        for (; currentRob < robCount && otherCurrentRob < otherRobCount; ) {
                            for (; currentBlock < alignedRobWidthBlocks && otherCurrentBlock < otherAlignedRobWidthBlocks; ) {
                                for (; currentSlice < mipLayouts[level].blockDepth && otherCurrentSlice < other.mipLayouts[otherLevel].blockDepth;) {
                                    for (; currentGob < mipLayouts[level].blockHeight && otherCurrentGob < other.mipLayouts[otherLevel].blockHeight; ++currentGob, imageOffset.y += 8, ++otherCurrentGob, otherImageOffset.y += 8) {
                                        if (currentBlock < robWidthBlocks && otherCurrentBlock < otherRobWidthBlocks && imageOffset.x < mipLayouts[level].dimensions.width && otherImageOffset.x < other.mipLayouts[otherLevel].dimensions.width) {
                                            if (imageOffset.y < mipLayouts[level].dimensions.height && otherImageOffset.y < other.mipLayouts[otherLevel].dimensions.height) {
                                                if (imageOffset.z < mipLayouts[level].dimensions.depth && otherImageOffset.z < other.mipLayouts[otherLevel].dimensions.depth) {
                                                    results.toStaging.emplace_back(vk::BufferImageCopy{
                                                        .imageSubresource = {
                                                            .aspectMask = otherFormat->vkAspect,
                                                            .layerCount = 1,
                                                            .mipLevel = otherLevel,
                                                            .baseArrayLayer = otherLayer
                                                        },
                                                        .imageExtent = {std::min(otherImageExtent.width, static_cast<u32>(otherImageOffset.x) + otherGobExtent.width) - static_cast<u32>(otherImageOffset.x),
                                                                        std::min(otherImageExtent.height, static_cast<u32>(otherImageOffset.y) + otherGobExtent.height) - static_cast<u32>(otherImageOffset.y),
                                                                        gobExtent.depth},
                                                        .bufferOffset = results.stagingBufferSize,
                                                        .imageOffset = otherImageOffset
                                                    });
                                                    results.fromStaging.emplace_back(vk::BufferImageCopy{
                                                        .imageSubresource = {
                                                            .aspectMask = format->vkAspect,
                                                            .layerCount = 1,
                                                            .mipLevel = level,
                                                            .baseArrayLayer = layer
                                                        },
                                                        .imageExtent = {std::min(imageExtent.width, static_cast<u32>(imageOffset.x) + gobExtent.width) - static_cast<u32>(imageOffset.x),
                                                                        std::min(imageExtent.height, static_cast<u32>(imageOffset.y) + gobExtent.height) - static_cast<u32>(imageOffset.y),
                                                                        gobExtent.depth},
                                                        .bufferOffset = results.stagingBufferSize,
                                                        .imageOffset = imageOffset
                                                    });

                                                    results.stagingBufferSize += 64 * 8;
                                                }
                                            }
                                        }
                                    }

                                    if (currentGob == mipLayouts[level].blockHeight) {
                                        imageOffset.y -= static_cast<i32>(mipLayouts[level].blockHeight) * 8;
                                        imageOffset.z += 1;
                                        currentGob = 0;
                                        ++currentSlice;
                                    }
                                    if (otherCurrentGob == other.mipLayouts[otherLevel].blockHeight) {
                                        otherImageOffset.y -= static_cast<i32>(other.mipLayouts[otherLevel].blockHeight) * 8;
                                        otherImageOffset.z += 1;
                                        otherCurrentGob = 0;
                                        ++otherCurrentSlice;
                                    }
                                }

                                if (currentSlice == mipLayouts[level].blockDepth) {
                                    imageOffset.z -= static_cast<i32>(mipLayouts[level].blockDepth);
                                    currentSlice = 0;
                                    imageOffset.x += static_cast<i32>(gobExtent.width);
                                    ++currentBlock;
                                }
                                if (otherCurrentSlice == other.mipLayouts[otherLevel].blockDepth) {
                                    otherImageOffset.z -= static_cast<i32>(other.mipLayouts[otherLevel].blockDepth);
                                    otherCurrentSlice = 0;
                                    otherImageOffset.x += static_cast<i32>(otherGobExtent.width);
                                    ++otherCurrentBlock;
                                }
                            }

                            if (currentBlock == alignedRobWidthBlocks) {
                                imageOffset.x = 0;
                                currentBlock = 0;
                                imageOffset.y += static_cast<i32>(mipLayouts[level].blockHeight) * 8;
                                ++currentRob;
                            }
                            if (otherCurrentBlock == otherAlignedRobWidthBlocks) {
                                otherImageOffset.x = 0;
                                otherCurrentBlock = 0;
                                otherImageOffset.y += static_cast<i32>(other.mipLayouts[otherLevel].blockHeight) * 8;
                                ++otherCurrentRob;
                            }
                        }

                        if (currentRob == robCount) {
                            imageOffset.y = 0;
                            currentRob = 0;
                            imageOffset.z += static_cast<i32>(mipLayouts[level].blockDepth);
                            ++currentMob;
                        }
                        if (otherCurrentRob == otherRobCount) {
                            otherImageOffset.y = 0;
                            otherCurrentRob = 0;
                            otherImageOffset.z += static_cast<i32>(other.mipLayouts[otherLevel].blockDepth);
                            ++otherCurrentMob;
                        }
                    }

                    if (currentMob == mobCount) {
                        imageOffset = vk::Offset3D{};
                        currentMob = 0;
                        ++level;
                    }
                    if (otherCurrentMob == otherMobCount) {
                        otherImageOffset = vk::Offset3D{};
                        otherCurrentMob = 0;
                        ++otherLevel;
                    }
                }

                if (level == levelCount) {
                    level = 0;
                    ++layer;
                }
                if (otherLevel == other.levelCount) {
                    otherLevel = 0;
                    ++otherLayer;
                }
            }

            return {std::move(results)};
        } else {
            LOGW("Src: {}x{}x{} format: {}, levels: {}, layers: {}, address: {}-{}, pitch: {}", other.mipLayouts[0].dimensions.width / otherFormat->bpb, other.mipLayouts[0].dimensions.height, other.mipLayouts[0].dimensions.depth, vk::to_string(otherFormat->vkFormat), other.levelCount, other.layerCount, fmt::ptr(other.mappings.front().data()), fmt::ptr(other.mappings.front().end().base()), other.tileConfig.pitch);
            LOGW("Dst: {}x{}x{} format: {}, levels: {}, layers: {}, address: {}-{}, pitch: {}", mipLayouts[0].dimensions.width / format->bpb, mipLayouts[0].dimensions.height, mipLayouts[0].dimensions.depth, vk::to_string(format->vkFormat), levelCount, layerCount, fmt::ptr(mappings.front().data()), fmt::ptr(mappings.front().end().base()), tileConfig.pitch);
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
        if (otherMappings.front().data() < mappings.front().data())
            return UINT32_MAX;

        u32 offset{};
        if (otherMappings.front().valid()) {
            auto it{mappings.begin()};
            while (!it->contains(otherMappings.front().data())) {
                offset += it->size();
                ++it;
            }
            offset += otherMappings.front().data() - it->data();
        }

        return offset;
    }
}
