// SPDX-License-Identifier: MPL-2.0
// Copyright © 2023 Skyline Team and Contributors (https://github.com/skyline-emu/)

#pragma once

#include <common/base.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include "common.h"

namespace skyline::gpu {
    namespace texture {
        using Mappings = boost::container::small_vector<span<u8>, 3>;
        struct TextureCopies {
            u32 stagingBufferSize;
            std::vector<vk::BufferImageCopy> toStaging;
            std::vector<vk::BufferImageCopy> fromStaging;
        };

        /**
         * @brief The layout of a texture in GPU memory
         * @note Refer to Chapter 20.1 of the Tegra X1 TRM for information
         */
        enum class TileMode {
            Linear, //!< All pixels are arranged linearly
            Pitch,  //!< All pixels are arranged linearly but rows aligned to the pitch
            Block,  //!< All pixels are arranged into blocks and swizzled in a Z-order curve to optimize for spacial locality
        };

        /**
         * @brief The parameters of the tiling mode, covered in Table 76 in the Tegra X1 TRM
         */
        struct TileConfig {
            TileMode mode;
            union {
                struct {
                    u8 blockHeight;        //!< The height of the blocks in GOBs
                    u8 blockDepth;         //!< The depth of the blocks in GOBs
                    u8 sparseBlockWidth{1};   //!< A multiple to align the amount of blocks to on the X axis
                };
                u32 pitch; //!< The pitch of the texture in bytes
            };

            constexpr bool operator==(const TileConfig &other) const {
                if (mode == other.mode) {
                    switch (mode) {
                        case TileMode::Linear:
                            return true;
                        case TileMode::Pitch:
                            return pitch == other.pitch;
                        case TileMode::Block:
                            return blockHeight == other.blockHeight && blockDepth == other.blockDepth && sparseBlockWidth == other.sparseBlockWidth;
                    }
                }

                return false;
            }
        };

        /**
         * @brief A description of a single mipmapped level of a block-linear surface
         */
        struct MipLevelLayout {
            Dimensions dimensions; //!< The dimensions of the mipmapped level, these are exact dimensions and not aligned to a GOB
            size_t linearSize; //!< The size of a linear image with this mipmapped level in bytes
            size_t blockLinearSize; //!< The size of a blocklinear image with this mipmapped level in bytes
            size_t blockHeight, blockDepth; //!< The block height and block depth set for the level

            constexpr MipLevelLayout(Dimensions dimensions, size_t linearSize, size_t blockLinearSize, size_t blockHeight, size_t blockDepth) : dimensions{dimensions}, linearSize{linearSize}, blockLinearSize{blockLinearSize}, blockHeight{blockHeight}, blockDepth{blockDepth} {}
        };

        u32 CalculateLayerStride(texture::Dimensions dimensions, texture::Format format, texture::TileConfig tileConfig, u32 levelCount, u32 layerCount);

        size_t CalculateLinearLayerStride(const std::vector<texture::MipLevelLayout> &mipLayouts);
    }

    /**
     * @brief Contains common information for all `HostTexture`s of a certain `Texture` from guest
     */
    struct GuestTexture {
      private:
        std::optional<texture::TextureCopies> CalculateOffsetlessOverlap(GuestTexture &other, texture::Format format, texture::Format otherFormat);

      public:
        texture::Mappings mappings; //!< Spans to CPU memory for the underlying data backing this texture
        texture::TileConfig tileConfig;
        u32 levelCount; //!< The total amount of mip levels in the parent texture
        u32 layerCount; //!< The total amount of array layers in the parent texture
        u32 layerStride; //!< The stride between layers in bytes, this may not match the calculated stride due to external alignment requirements
        u32 layerSize; //!< The size of a single layer in bytes
        u32 size; //!< The size of the texture in bytes, same as layer stride * layer count

        std::vector<texture::MipLevelLayout> mipLayouts; //!< The layout of each mip level in the guest texture, the dimensions are in bytes and this. Do not use for any copies as they won't match due to rounding differences.
        size_t linearLayerStride; //!< The stride of a single layer given linear tiling using the guest format
        size_t linearSize; //!< The size of the texture given linear tiling, same as linear layer stride * layer count

        GuestTexture(texture::Mappings mappings, texture::Dimensions sampleDimensions, texture::Format format, texture::TileConfig tileConfig, u32 levelCount, u32 layerCount, u32 layerStride);

        /**
         * @brief Calculates the subresource range of a bit-compatible resource starting at the given offset, if there are irreconcilable differences then std::nullopt is returned
         * @note The returned aspect mask is always the same as the aspect mask of the supplied format, it is not affected by the guest texture's format
         */
        std::optional<vk::ImageSubresourceRange> CalculateSubresource(texture::TileConfig pTileConfig, u32 offset, texture::Dimensions pTextureSizes, u32 pLevelCount, u32 pLayerCount, u32 pLayerStride, vk::ImageAspectFlags aspectMask);

        std::optional<texture::TextureCopies> CalculateCopy(GuestTexture &other, texture::Format format, texture::Format otherFormat);

        std::optional<texture::TextureCopies> CalculateReinterpretCopy(GuestTexture &other, texture::Format format, texture::Format otherFormat);

        /**
         * @brief Returns true if `otherMappings` is completely inside `mappings` false otherwise
         */
        bool Contains(const texture::Mappings &otherMappings);

        /**
         * @returns The amount of bytes between the start of `otherMappings` and `mappings`. Returns U32_MAX if the start of `otherMappings` is before `mappings`
         */
        u32 OffsetFrom(const texture::Mappings &otherMappings);
    };
}
