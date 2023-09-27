// SPDX-License-Identifier: MPL-2.0
// Copyright © 2021 Skyline Team and Contributors (https://github.com/skyline-emu/)
// Copyright © 2018-2020 fincs (https://github.com/devkitPro/deko3d)

#pragma once

#include <bit>
#include <common/base.h>

namespace skyline::gpu::interconnect {
    #pragma pack(push, 1)

    /**
     * @brief The Texture Image Control is a descriptor used to configure the texture unit in Maxwell GPUs
     * @url https://github.com/NVIDIA/open-gpu-doc/blob/master/classes/3d/clb197tex.h
     * @url https://github.com/envytools/envytools/blob/master/rnndb/graph/gm200_texture.xml
     * @url https://github.com/devkitPro/deko3d/blob/00c12d1f4809014f1cc22719dd2e3476735eec64/source/maxwell/texture_image_control_block.h
     * @note Any members with underscore number suffixes represent a bitfield range of a value that member represents
     * @note Any enumerations that have numerical enumerants are prefixed with 'e'
     */
    struct TextureImageControl {
        /**
         * @note An underscore may be used to describe a different block in a format
         */
        enum class ImageFormat : u8 {
            Invalid = 0x00,
            R32G32B32A32 = 0x01,
            R32G32B32 = 0x02,
            R16G16B16A16 = 0x03,
            R32G32 = 0x04,
            R32B24G8 = 0x05,
            Etc2RGB = 0x06,
            X8B8G8R8 = 0x07,
            A8B8G8R8 = 0x08,
            A2B10G10R10 = 0x09,
            Etc2RGBPta = 0x0A,
            Etc2RGBA = 0x0B,
            R16G16 = 0x0C,
            G8R24 = 0x0D,
            G24R8 = 0x0E,
            R32 = 0x0F,
            Bc6HSfloat = 0x10,
            Bc6HUfloat = 0x11,
            A4B4G4R4 = 0x12,
            A5B5G5R1 = 0x13,
            A1B5G5R5 = 0x14,
            B5G6R5 = 0x15,
            B6G5R5 = 0x16,
            Bc7 = 0x17,
            G8R8 = 0x18,
            Eac = 0x19,
            EacX2 = 0x1A,
            R16 = 0x1B,
            Y8Video = 0x1C,
            R8 = 0x1D,
            G4R4 = 0x1E,
            R1 = 0x1F,
            E5B9G9R9 = 0x20,
            B10G11R11 = 0x21,
            G8B8G8R8 = 0x22,
            B8G8R8G8 = 0x23,
            Bc1 = 0x24,
            Bc2 = 0x25,
            Bc3 = 0x26,
            Bc4 = 0x27,
            Bc5 = 0x28,
            D24S8 = 0x29,
            X8D24 = 0x2A,
            S8D24 = 0x2B,
            X4V4D24_Cov4R4V = 0x2C,
            X4V4D24_Cov8R8V = 0x2D,
            V8D24_Cov4R12V = 0x2E,
            D32 = 0x2F,
            D32X24S8 = 0x30,
            X8D24_X20V4S8_Cov4R4V = 0x31,
            X8D24_X20V4S8_Cov8R8V = 0x32,
            D32_X20V4X8_Cov4R4V = 0x33,
            D32_X20V4X8_Cov8R8V = 0x34,
            D32_X20V4S8_Cov4R4V = 0x35,
            D32_X20V4S8_Cov8R8V = 0x36,
            X8D24_X16V8S8_Cov4R12V = 0x37,
            D32_X16V8X8_Cov4R12V = 0x38,
            D32_X16V8S8_Cov4R12V = 0x39,
            D16 = 0x3A,
            V8D24_Cov8R24V = 0x3B,
            X8D24_X16V8S8_Cov8R24V = 0x3C,
            D32_X16V8X8_Cov8R24V = 0x3D,
            D32_X16V8S8_Cov8R24V = 0x3E,
            Astc4x4 = 0x40,
            Astc5x5 = 0x41,
            Astc6x6 = 0x42,
            Astc8x8 = 0x44,
            Astc10x10 = 0x45,
            Astc12x12 = 0x46,
            Astc5x4 = 0x50,
            Astc6x5 = 0x51,
            Astc8x6 = 0x52,
            Astc10x8 = 0x53,
            Astc12x10 = 0x54,
            Astc8x5 = 0x55,
            Astc10x5 = 0x56,
            Astc10x6 = 0x57,
        };

        enum class ImageComponent : u8 {
            Snorm = 1,
            Unorm = 2,
            Sint = 3,
            Uint = 4,
            SnormForceFp16 = 5,
            UnormForceFp16 = 6,
            Float = 7,
        };

        enum class ImageSwizzle : u8 {
            Zero = 0,
            R = 2,
            G = 3,
            B = 4,
            A = 5,
            OneInt = 6,
            OneFloat = 7,
        };

        enum class HeaderType : u8 {
            Buffer1D = 0,
            PitchColorKey = 1,
            Pitch = 2,
            BlockLinear = 3,
            BlockLinearColorKey = 4,
        };

        enum class TextureType : u8 {
            e1D = 0,
            e2D = 1,
            e3D = 2,
            eCube = 3,
            e1DArray = 4,
            e2DArray = 5,
            e1DBuffer = 6,
            e2DNoMipmap = 7,
            eCubeArray = 8,
        };

        enum class MsaaMode : u8 {
            e1x1 = 0,
            e2x1 = 1,
            e2x2 = 2,
            e4x2 = 3,
            e4x2D3D = 4,
            e2x1D3D = 5,
            e4x4 = 6,
            e2x2Vc4 = 8,
            e2x2Vc12 = 9,
            e4x2Vc8 = 10,
            e4x2Vc24 = 11,
        };

        enum class LodQuality : u8 {
            Low = 0,
            High = 1,
        };

        enum class SectorPromotion : u8 {
            None = 0,
            To2V = 1,
            To2H = 2,
            To4 = 3,
        };

        enum class BorderSize : u8 {
            One = 0,
            Two = 1,
            Four = 2,
            Eight = 3,
            SamplerColor = 7,
        };

        enum class AnisotropySpreadModifier : u8 {
            None = 0,
            One = 1,
            Two = 2,
            Sqrt = 3,
        };

        enum class AnisotropySpread : u8 {
            Half = 0,
            One = 1,
            Two = 2,
            Max = 3,
        };

        enum class MaxAnisotropy : u8 {
            e1to1 = 0,
            e2to1 = 1,
            e4to1 = 2,
            e6to1 = 3,
            e8to1 = 4,
            e10to1 = 5,
            e12to1 = 6,
            e16to1 = 7,
        };

        // 0x00
        struct FormatWord {
            static constexpr u32 FormatColorComponentPadMask{(1U << 31) | 0b111'111'111'111'1111111U}; //!< Mask for the format, component and pad fields

            ImageFormat format : 7;
            ImageComponent componentR : 3;
            ImageComponent componentG : 3;
            ImageComponent componentB : 3;
            ImageComponent componentA : 3;
            ImageSwizzle swizzleX : 3;
            ImageSwizzle swizzleY : 3;
            ImageSwizzle swizzleZ : 3;
            ImageSwizzle swizzleW : 3;
            bool packComponents : 1;

            constexpr bool operator==(const FormatWord &) const = default;

            constexpr u32 Raw() const {
                if (std::is_constant_evaluated()) {
                    u32 raw{packComponents};
                    raw <<= 3;
                    raw |= static_cast<u32>(swizzleW);
                    raw <<= 3;
                    raw |= static_cast<u32>(swizzleZ);
                    raw <<= 3;
                    raw |= static_cast<u32>(swizzleY);
                    raw <<= 3;
                    raw |= static_cast<u32>(swizzleX);
                    raw <<= 3;
                    raw |= static_cast<u32>(componentA);
                    raw <<= 3;
                    raw |= static_cast<u32>(componentB);
                    raw <<= 3;
                    raw |= static_cast<u32>(componentG);
                    raw <<= 3;
                    raw |= static_cast<u32>(componentR);
                    raw <<= 7;
                    raw |= static_cast<u32>(format);
                    return raw;
                } else {
                    return util::BitCast<u32>(*this);
                }
            }
        } formatWord;

        // 0x04
        union Address {
            //!< Pitch specific
            struct {
                u8 reserved1A : 5;
                u32 address31To5 : 27;
            };
            //!< Blocklinear specific
            struct {
                u8 reserved1Y : 5;
                u8 gobDepthOffset : 2; //!< The offset of the texture in slices
                u8 reserved1X : 2;
                u32 address31To9 : 23;
            };

            u32 raw;

            constexpr bool operator==(const Address &other) const {
                return raw == other.raw;
            };
        } lowAddress;

        // 0x08
        u16 address47To32;
        u8 addressReserved : 5;
        TextureImageControl::HeaderType headerType : 3;
        u8 reservedHeaderVersion : 1;
        u8 resourceViewCoherencyHash : 4;
        u8 reserved2A : 3;

        // 0x0C
        union TileConfig {
            //!< Pitch specific
            u16 pitch20To5;
            //!< Blocklinear specific
            struct {
                u8 gobsPerBlockWidthLog2 : 3;
                u8 gobsPerBlockHeightLog2 : 3;
                u8 gobsPerBlockDepthLog2 : 3;
                u8 reserved3Y : 1;
                u8 tileWidthInGobsLog2 : 3;
                bool gob3D : 1;
                u8 reserved3Z : 2;
            };

            u16 raw;

            constexpr bool operator==(const TileConfig &other) const {
                return raw == other.raw;
            };
        } tileConfig;
        bool lodAnisoQuality2 : 1;
        TextureImageControl::LodQuality lodAnisoQualityLod : 1;
        TextureImageControl::LodQuality lodIsoQualityLod : 1;
        TextureImageControl::AnisotropySpreadModifier anisotropyCoarseSpreadModifier : 2;
        u8 anisoSpreadScale : 5;
        bool useHeaderOptControl : 1;
        bool depthTexture : 1;
        u8 maxMipLevel : 4;

        // 0x10
        u16 widthMinusOne;
        u8 reserved4A : 3;
        u8 anisotropySpreadMaxLog2 : 3;
        bool srgbConversion : 1;
        TextureImageControl::TextureType textureType : 4;
        TextureImageControl::SectorPromotion sectorPromotion : 2;
        TextureImageControl::BorderSize borderSize : 3;

        // 0x14
        u16 heightMinusOne;
        u16 depthMinusOne : 14;
        u8 reserved5A : 1;
        bool normalizedCordinates : 1;

        // 0x18
        u8 reserved6Y : 1;
        u8 trilinearOpt : 5;
        u16 mipLodBias : 13;
        u8 anisotropyBias : 4;
        AnisotropySpread anisotropyFineSpreadFunc : 2;
        AnisotropySpread anisotropyCoarseSpreadFunc : 2;
        MaxAnisotropy maxAnisotropy : 3;
        AnisotropySpreadModifier anisotropyFineSpreadModifier : 2;

        // 0x1C
        u8 viewMinMipLevel : 4;
        u8 viewMaxMipLevel : 4;
        MsaaMode msaaMode : 4;
        u16 minLodClamp : 12;
        u8 reserved7Y;

        constexpr bool operator==(const TextureImageControl &) const = default;

        constexpr u64 Iova() const {
            switch (headerType) {
                case TextureImageControl::HeaderType::Pitch:
                    return (static_cast<u64>(address47To32) << 32) | (static_cast<u64>(lowAddress.address31To5) << 5);
                case TextureImageControl::HeaderType::BlockLinear:
                    return (static_cast<u64>(address47To32) << 32) | (static_cast<u64>(lowAddress.address31To9) << 9);
                default:
                    return 0UL;
            }
        }
    };
    static_assert(sizeof(TextureImageControl) == 0x20);

    #pragma pack(pop)
}
