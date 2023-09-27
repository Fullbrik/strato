// SPDX-License-Identifier: MPL-2.0
// Copyright © 2022 Skyline Team and Contributors (https://github.com/skyline-emu/)

#include <soc/gm20b/channel.h>
#include <soc/gm20b/gmmu.h>
#include <gpu/texture_manager.h>
#include <gpu/texture/formats.h>
#include <vulkan/vulkan_enums.hpp>
#include "gpu/texture/guest_texture.h"
#include "textures.h"

namespace skyline::gpu::interconnect {
    void TexturePoolState::EngineRegisters::DirtyBind(DirtyManager &manager, dirty::Handle handle) const {
        manager.Bind(handle, texHeaderPool);
    }

    TexturePoolState::TexturePoolState(dirty::Handle dirtyHandle, DirtyManager &manager, const EngineRegisters &engine) : engine{manager, dirtyHandle, engine} {}

    void TexturePoolState::Flush(InterconnectContext &ctx) {
        auto mapping{ctx.channelCtx.asCtx->gmmu.LookupBlock(engine->texHeaderPool.offset)};

        textureHeaders = mapping.first.subspan(mapping.second).cast<TextureImageControl>().first(engine->texHeaderPool.maximumIndex + 1);
    }

    void TexturePoolState::PurgeCaches() {
        textureHeaders = span<TextureImageControl>{};
    }

    Textures::Textures(DirtyManager &manager, const TexturePoolState::EngineRegisters &engine) : texturePool{manager, engine} {}

    void Textures::MarkAllDirty() {
        texturePool.MarkDirty(true);
    }

    static texture::Format ConvertTicFormat(TextureImageControl::FormatWord format, bool srgb) {
        using TIC = TextureImageControl;
        #define TIC_FORMAT(fmt, compR, compG, compB, compA, srgb) \
                TIC::FormatWord{ .format = TIC::ImageFormat::fmt,          \
                                 .componentR = TIC::ImageComponent::compR, \
                                 .componentG = TIC::ImageComponent::compG, \
                                 .componentB = TIC::ImageComponent::compB, \
                                 .componentA = TIC::ImageComponent::compA, \
                                 .packComponents = srgb }.Raw() // Reuse _pad_ to store if the texture is sRGB

        // For formats where all components are of the same type
        #define TIC_FORMAT_ST(format, component) \
                TIC_FORMAT(format, component, component, component, component, false)

        #define TIC_FORMAT_ST_SRGB(format, component) \
                TIC_FORMAT(format, component, component, component, component, true)

        #define TIC_FORMAT_CASE(ticFormat, skFormat, componentR, componentG, componentB, componentA)  \
                case TIC_FORMAT(ticFormat, componentR, componentG, componentB, componentA, false): \
                    return format::skFormat

        #define TIC_FORMAT_CASE_ST(ticFormat, skFormat, component)  \
                case TIC_FORMAT_ST(ticFormat, component): \
                    return format::skFormat ## component

        #define TIC_FORMAT_CASE_ST_SRGB(ticFormat, skFormat, component)  \
                case TIC_FORMAT_ST_SRGB(ticFormat, component): \
                    return format::skFormat ## Srgb

        #define TIC_FORMAT_CASE_NORM(ticFormat, skFormat)  \
                TIC_FORMAT_CASE_ST(ticFormat, skFormat, Unorm); \
                TIC_FORMAT_CASE_ST(ticFormat, skFormat, Snorm)

        #define TIC_FORMAT_CASE_INT(ticFormat, skFormat)  \
                TIC_FORMAT_CASE_ST(ticFormat, skFormat, Uint); \
                TIC_FORMAT_CASE_ST(ticFormat, skFormat, Sint)

        #define TIC_FORMAT_CASE_NORM_INT(ticFormat, skFormat) \
                TIC_FORMAT_CASE_NORM(ticFormat, skFormat); \
                TIC_FORMAT_CASE_INT(ticFormat, skFormat)

        #define TIC_FORMAT_CASE_NORM_INT_FLOAT(ticFormat, skFormat) \
                TIC_FORMAT_CASE_NORM_INT(ticFormat, skFormat); \
                TIC_FORMAT_CASE_ST(ticFormat, skFormat, Float)

        #define TIC_FORMAT_CASE_INT_FLOAT(ticFormat, skFormat) \
                TIC_FORMAT_CASE_INT(ticFormat, skFormat); \
                TIC_FORMAT_CASE_ST(ticFormat, skFormat, Float)

        // Ignore the swizzle components of the format word
        // FIXME: Don't do this
        format.packComponents = srgb; // Reuse the packComponents field to store the srgb flag
        switch ((format.Raw() & TextureImageControl::FormatWord::FormatColorComponentPadMask)) {
            TIC_FORMAT_CASE_NORM_INT(R8, R8);

            TIC_FORMAT_CASE_NORM_INT_FLOAT(R16, R16);
            TIC_FORMAT_CASE_ST(D16, D16, Unorm);
            TIC_FORMAT_CASE_NORM_INT(G8R8, R8G8);
            TIC_FORMAT_CASE_ST(B5G6R5, B5G6R5, Unorm);
            TIC_FORMAT_CASE_ST(A4B4G4R4, R4G4B4A4, Unorm);
            TIC_FORMAT_CASE_ST(A1B5G5R5, A1B5G5R5, Unorm);

            TIC_FORMAT_CASE_INT_FLOAT(R32, R32);
            TIC_FORMAT_CASE_ST(D32, D32, Float);
            TIC_FORMAT_CASE_NORM_INT_FLOAT(R16G16, R16G16);
            TIC_FORMAT_CASE(G24R8, S8UintD24Unorm, Uint, Unorm, Unorm, Unorm);
            TIC_FORMAT_CASE(D24S8, S8UintD24Unorm, Uint, Unorm, Uint, Uint);
            TIC_FORMAT_CASE(D24S8, S8UintD24Unorm, Uint, Unorm, Unorm, Unorm);
            TIC_FORMAT_CASE(S8D24, D24UnormS8Uint, Unorm, Uint, Uint, Uint);

            TIC_FORMAT_CASE_ST(B10G11R11, B10G11R11, Float);
            TIC_FORMAT_CASE_NORM_INT(A8B8G8R8, R8G8B8A8);
            TIC_FORMAT_CASE_ST_SRGB(A8B8G8R8, R8G8B8A8, Unorm);
            TIC_FORMAT_CASE_NORM_INT(A2B10G10R10, A2B10G10R10);
            TIC_FORMAT_CASE_ST(E5B9G9R9, E5B9G9R9, Float);

            TIC_FORMAT_CASE_ST(Bc1, BC1, Unorm);
            TIC_FORMAT_CASE_ST_SRGB(Bc1, BC1, Unorm);
            TIC_FORMAT_CASE_NORM(Bc4, BC4);
            TIC_FORMAT_CASE_INT_FLOAT(R32G32, R32G32);
            TIC_FORMAT_CASE(D32X24S8, D32FloatS8Uint, Float, Uint, Uint, Unorm);
            TIC_FORMAT_CASE(D32X24S8, D32FloatS8Uint, Float, Uint, Unorm, Unorm);
            TIC_FORMAT_CASE(R32B24G8, D32FloatS8Uint, Float, Uint, Unorm, Unorm);

            TIC_FORMAT_CASE_NORM_INT_FLOAT(R16G16B16A16, R16G16B16A16);

            TIC_FORMAT_CASE_ST(Astc4x4, Astc4x4, Unorm);
            TIC_FORMAT_CASE_ST_SRGB(Astc4x4, Astc4x4, Unorm);
            TIC_FORMAT_CASE_ST(Astc5x4, Astc5x4, Unorm);
            TIC_FORMAT_CASE_ST_SRGB(Astc5x4, Astc5x4, Unorm);
            TIC_FORMAT_CASE_ST(Astc5x5, Astc5x5, Unorm);
            TIC_FORMAT_CASE_ST_SRGB(Astc5x5, Astc5x5, Unorm);
            TIC_FORMAT_CASE_ST(Astc6x5, Astc6x5, Unorm);
            TIC_FORMAT_CASE_ST_SRGB(Astc6x5, Astc6x5, Unorm);
            TIC_FORMAT_CASE_ST(Astc6x6, Astc6x6, Unorm);
            TIC_FORMAT_CASE_ST_SRGB(Astc6x6, Astc6x6, Unorm);
            TIC_FORMAT_CASE_ST(Astc8x5, Astc8x5, Unorm);
            TIC_FORMAT_CASE_ST_SRGB(Astc8x5, Astc8x5, Unorm);
            TIC_FORMAT_CASE_ST(Astc8x6, Astc8x6, Unorm);
            TIC_FORMAT_CASE_ST_SRGB(Astc8x6, Astc8x6, Unorm);
            TIC_FORMAT_CASE_ST(Astc8x8, Astc8x8, Unorm);
            TIC_FORMAT_CASE_ST_SRGB(Astc8x8, Astc8x8, Unorm);
            TIC_FORMAT_CASE_ST(Astc10x5, Astc10x5, Unorm);
            TIC_FORMAT_CASE_ST_SRGB(Astc10x5, Astc10x5, Unorm);
            TIC_FORMAT_CASE_ST(Astc10x6, Astc10x6, Unorm);
            TIC_FORMAT_CASE_ST_SRGB(Astc10x6, Astc10x6, Unorm);
            TIC_FORMAT_CASE_ST(Astc10x8, Astc10x8, Unorm);
            TIC_FORMAT_CASE_ST_SRGB(Astc10x8, Astc10x8, Unorm);
            TIC_FORMAT_CASE_ST(Astc10x10, Astc10x10, Unorm);
            TIC_FORMAT_CASE_ST_SRGB(Astc10x10, Astc10x10, Unorm);
            TIC_FORMAT_CASE_ST(Astc12x10, Astc12x10, Unorm);
            TIC_FORMAT_CASE_ST_SRGB(Astc12x10, Astc12x10, Unorm);
            TIC_FORMAT_CASE_ST(Astc12x12, Astc12x12, Unorm);
	        TIC_FORMAT_CASE_ST_SRGB(Astc12x12, Astc12x12, Unorm);

            TIC_FORMAT_CASE_ST(Bc2, BC2, Unorm);
            TIC_FORMAT_CASE_ST_SRGB(Bc2, BC2, Unorm);
            TIC_FORMAT_CASE_ST(Bc3, BC3, Unorm);
            TIC_FORMAT_CASE_ST_SRGB(Bc3, BC3, Unorm);
            TIC_FORMAT_CASE_NORM(Bc5, BC5);
            TIC_FORMAT_CASE(Bc6HUfloat, Bc6HUfloat, Float, Float, Float, Float);
            TIC_FORMAT_CASE(Bc6HSfloat, Bc6HSfloat, Float, Float, Float, Float);
            TIC_FORMAT_CASE_ST(Bc7, BC7, Unorm);
            TIC_FORMAT_CASE_ST_SRGB(Bc7, BC7, Unorm);

            TIC_FORMAT_CASE_INT_FLOAT(R32G32B32A32, R32G32B32A32);

            default:
                if (format.Raw())
                    LOGE("Cannot translate TIC format: 0x{:X}", static_cast<u32>(format.Raw()));
                return {};
        }

        #undef TIC_FORMAT
        #undef TIC_FORMAT_ST
        #undef TIC_FORMAT_CASE
        #undef TIC_FORMAT_CASE_ST
        #undef TIC_FORMAT_CASE_NORM
        #undef TIC_FORMAT_CASE_INT
        #undef TIC_FORMAT_CASE_NORM_INT
        #undef TIC_FORMAT_CASE_NORM_INT_FLOAT
    }

    static vk::ComponentMapping ConvertTicSwizzleMapping(TextureImageControl::FormatWord format, vk::ComponentMapping swizzleMapping) {
        auto convertComponentSwizzle{[swizzleMapping](TextureImageControl::ImageSwizzle swizzle) {
            switch (swizzle) {
                case TextureImageControl::ImageSwizzle::R:
                    return swizzleMapping.r;
                case TextureImageControl::ImageSwizzle::G:
                    return swizzleMapping.g;
                case TextureImageControl::ImageSwizzle::B:
                    return swizzleMapping.b;
                case TextureImageControl::ImageSwizzle::A:
                    return swizzleMapping.a;
                case TextureImageControl::ImageSwizzle::Zero:
                    return vk::ComponentSwizzle::eZero;
                case TextureImageControl::ImageSwizzle::OneFloat:
                case TextureImageControl::ImageSwizzle::OneInt:
                    return vk::ComponentSwizzle::eOne;
                default:
                    LOGE("Invalid swizzle: {:X}", static_cast<u32>(swizzle));
                    return vk::ComponentSwizzle::eZero;
            }
        }};

        return vk::ComponentMapping{
            .r = convertComponentSwizzle(format.swizzleX),
            .g = convertComponentSwizzle(format.swizzleY),
            .b = convertComponentSwizzle(format.swizzleZ),
            .a = convertComponentSwizzle(format.swizzleW)
        };
    }

    HostTextureView *Textures::GetTexture(InterconnectContext &ctx, u32 index, Shader::TextureType shaderType, bool isStorage, Shader::ImageFormat overrideFormat) {
        auto textureHeaders{texturePool.UpdateGet(ctx).textureHeaders};
        if (textureHeaderCache.size() != textureHeaders.size()) {
            textureHeaderCache.resize(textureHeaders.size());
            std::fill(textureHeaderCache.begin(), textureHeaderCache.end(), CacheEntry{});
        } else if (textureHeaders.size() > index && textureHeaderCache[index].view && !isStorage) {
            auto &cached{textureHeaderCache[index]};
            if (cached.sequenceNumber == ctx.channelCtx.channelSequenceNumber)
                return cached.view;

            if (cached.tic == textureHeaders[index] && !cached.view->stale) {
                cached.sequenceNumber = ctx.channelCtx.channelSequenceNumber;

                return cached.view;
            }
        }
        if (index >= textureHeaders.size()) [[unlikely]]
            return nullptr;

        TextureImageControl &textureHeader{textureHeaders[index]};
        auto &texture{textureHeaderStore[textureHeader]};

        if (!texture || texture->stale || isStorage) {
            texture::Format format;
            if (isStorage && overrideFormat != Shader::ImageFormat::Typeless) {
                switch (overrideFormat) {
                    case Shader::ImageFormat::R8_UINT:
                        format = format::R8Uint;
                        break;
                    case Shader::ImageFormat::R8_SINT:
                        format = format::R8Sint;
                        break;
                    case Shader::ImageFormat::R16_UINT:
                        format = format::R16Uint;
                        break;
                    case Shader::ImageFormat::R16_SINT:
                        format = format::R16Sint;
                        break;
                    case Shader::ImageFormat::R32_UINT:
                        format = format::R32Uint;
                        break;
                    case Shader::ImageFormat::R32G32_UINT:
                        format = format::R32G32Uint;
                        break;
                    case Shader::ImageFormat::R32G32B32A32_UINT:
                        format = format::R32G32B32A32Uint;
                        break;
                    [[unlikely]] default:
                        LOGE("Invalid storage image format: 0x{:X}", static_cast<u32>(overrideFormat));
                }
            } else {
                format = ConvertTicFormat(textureHeader.formatWord, textureHeader.srgbConversion);
            }
            if (!format)
                return nullptr;

            constexpr size_t CubeFaceCount{6};

            texture::Dimensions imageDimensions{static_cast<u32>(textureHeader.widthMinusOne + 1), static_cast<u32>(textureHeader.heightMinusOne + 1)};

            texture::TileConfig tileConfig{};

            switch (textureHeader.headerType) {
                case TextureImageControl::HeaderType::BlockLinear:
                    if (textureHeader.lowAddress.gobDepthOffset != 0)
                        LOGW("Gob Depth offsets are not supported! (0x{:X})", static_cast<u8>(textureHeader.lowAddress.gobDepthOffset));

                    tileConfig = {
                        .mode = texture::TileMode::Block,
                        .blockHeight = static_cast<u8>(1U << textureHeader.tileConfig.gobsPerBlockHeightLog2),
                        .blockDepth = static_cast<u8>(1U << textureHeader.tileConfig.gobsPerBlockDepthLog2),
                        .sparseBlockWidth = static_cast<u8>(1U << textureHeader.tileConfig.tileWidthInGobsLog2)
                    };
                    break;
                case TextureImageControl::HeaderType::Pitch:
                    tileConfig = {
                        .mode = texture::TileMode::Pitch,
                        .pitch = static_cast<u32>(textureHeader.tileConfig.pitch20To5 << 5U)
                    };
                    break;
                [[unlikely]] default:
                    LOGE("Unsupported or invalid TIC header type 0x{:X}!", static_cast<u32>(textureHeader.headerType));
                    return nullptr;
            }

            u16 layerCount{1};
            u32 levelCount{static_cast<u32>(textureHeader.maxMipLevel + 1)}, viewMipBase{textureHeader.viewMinMipLevel}, viewMipCount{static_cast<u32>(textureHeader.viewMaxMipLevel - textureHeader.viewMinMipLevel) + 1};

            vk::ImageViewType viewType;
            switch (textureHeader.textureType) {
                case TextureImageControl::TextureType::e1D:
                    viewType = shaderType == Shader::TextureType::ColorArray1D ? vk::ImageViewType::e1DArray : vk::ImageViewType::e1D;
                    imageDimensions.height = 1;
                    break;
                case TextureImageControl::TextureType::e1DArray:
                    viewType = vk::ImageViewType::e1DArray;
                    imageDimensions.height = 1;
                    layerCount = textureHeader.depthMinusOne + 1;
                    break;
                case TextureImageControl::TextureType::e1DBuffer:
                    throw exception("1D Buffers are not supported");
                case TextureImageControl::TextureType::e2DNoMipmap:
                    levelCount = 1;
                    viewMipBase = 0;
                    viewMipCount = 1;
                case TextureImageControl::TextureType::e2D:
                    viewType = shaderType == Shader::TextureType::ColorArray2D ? vk::ImageViewType::e2DArray : vk::ImageViewType::e2D;
                    break;
                case TextureImageControl::TextureType::e2DArray:
                    viewType = vk::ImageViewType::e2DArray;
                    layerCount = textureHeader.depthMinusOne + 1;
                    break;
                case TextureImageControl::TextureType::e3D:
                    viewType = vk::ImageViewType::e3D;
                    imageDimensions.depth = textureHeader.depthMinusOne + 1;
                    break;
                case TextureImageControl::TextureType::eCube:
                    viewType = shaderType == Shader::TextureType::ColorArrayCube ? vk::ImageViewType::eCubeArray : vk::ImageViewType::eCube;
                    layerCount = CubeFaceCount;
                    break;
                case TextureImageControl::TextureType::eCubeArray:
                    viewType = vk::ImageViewType::eCubeArray;
                    layerCount = (textureHeader.depthMinusOne + 1) * CubeFaceCount;
                    break;
                [[unlikely]] default:
                    LOGE("Invalid TIC texture type: 0x{:X}", static_cast<u32>(textureHeader.textureType));
                    return nullptr;
            }

            vk::SampleCountFlagBits sampleCount{vk::SampleCountFlagBits::e1};
            texture::Dimensions sampleDimensions{imageDimensions};
            // TODO: Support MSAA rendering, this code is pointless without it
            //switch (textureHeader.msaaMode) {
            //    case TextureImageControl::MsaaMode::e1x1:
            //        sampleCount = vk::SampleCountFlagBits::e1;
            //        break;
            //    case TextureImageControl::MsaaMode::e2x1:
            //    case TextureImageControl::MsaaMode::e2x1D3D:
            //        sampleCount = vk::SampleCountFlagBits::e2;
            //        sampleDimensions.width *= 2;
            //        break;
            //    case TextureImageControl::MsaaMode::e2x2:
            //    case TextureImageControl::MsaaMode::e2x2Vc4:
            //    case TextureImageControl::MsaaMode::e2x2Vc12:
            //        sampleCount = vk::SampleCountFlagBits::e4;
            //        sampleDimensions.width *= 2;
            //        sampleDimensions.height *= 2;
            //        break;
            //    case TextureImageControl::MsaaMode::e4x2:
            //    case TextureImageControl::MsaaMode::e4x2D3D:
            //    case TextureImageControl::MsaaMode::e4x2Vc8:
            //    case TextureImageControl::MsaaMode::e4x2Vc24:
            //        sampleCount = vk::SampleCountFlagBits::e8;
            //        sampleDimensions.width *= 4;
            //        sampleDimensions.height *= 2;
            //        break;
            //    case TextureImageControl::MsaaMode::e4x4:
            //        sampleCount = vk::SampleCountFlagBits::e16;
            //        sampleDimensions.width *= 4;
            //        sampleDimensions.height *= 4;
            //        break;
            //    default:
            //        throw exception("Invalid MSAA mode: {}", static_cast<u32>(textureHeader.msaaMode));
            //}

            u32 layerStride{texture::CalculateLayerStride(sampleDimensions, format, tileConfig, levelCount, layerCount)};
            texture::Mappings mappings{ctx.channelCtx.asCtx->gmmu.TranslateRange(textureHeader.Iova(), layerStride * layerCount)};
            if (mappings.empty() || std::none_of(mappings.begin(), mappings.end(), [](const auto &mapping) { return mapping.valid(); })) [[unlikely]] {
                LOGW("Unmapped texture in TIC pool: 0x{:X}", textureHeader.Iova());
                return nullptr;
            }

            //!< Image views from storage images can only be created with the identity swizzle
            auto swizzle{isStorage ? vk::ComponentMapping{} : ConvertTicSwizzleMapping(textureHeader.formatWord, format->swizzleMapping)};

            texture = ctx.gpu.texture.FindOrCreate({
                .tag = ctx.executor.executionTag,
                .mappings = std::move(mappings),
                .sampleDimensions = sampleDimensions,
                .imageDimensions = imageDimensions,
                .sampleCount = sampleCount,
                .tileConfig = tileConfig,
                .layerCount = layerCount,
                .levelCount = static_cast<u16>(levelCount),
                .layerStride = layerStride,
                .viewFormat = format,
                .viewAspect = format->Aspect(swizzle),
                .viewType = viewType,
                .viewComponents = swizzle,
                .viewMipBase = static_cast<u16>(viewMipBase),
                .viewMipCount = static_cast<u16>(viewMipCount),
                .extraUsageFlags = isStorage ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlags{}
            });
        }

        //!< Don't attempt to cache storage images
        if (!isStorage)
            textureHeaderCache[index] = {textureHeader, texture, ctx.channelCtx.channelSequenceNumber};
        return texture;
    }

    Shader::TextureType Textures::GetTextureType(InterconnectContext &ctx, u32 index) {
        auto textureHeaders{texturePool.UpdateGet(ctx).textureHeaders};
        switch (textureHeaders[index].textureType) {
            case TextureImageControl::TextureType::e1D:
                return Shader::TextureType::Color1D;
            case TextureImageControl::TextureType::e1DArray:
                return Shader::TextureType::ColorArray1D;
            case TextureImageControl::TextureType::e1DBuffer:
                return Shader::TextureType::Buffer;
            case TextureImageControl::TextureType::e2DNoMipmap:
            case TextureImageControl::TextureType::e2D:
                return Shader::TextureType::Color2D;
            case TextureImageControl::TextureType::e2DArray:
                return Shader::TextureType::ColorArray2D;
            case TextureImageControl::TextureType::e3D:
                return Shader::TextureType::Color3D;
            case TextureImageControl::TextureType::eCube:
                return Shader::TextureType::ColorCube;
            case TextureImageControl::TextureType::eCubeArray:
                return Shader::TextureType::ColorArrayCube;
            [[unlikely]] default:
                LOGE("Invalid texture type: 0x{:X}", static_cast<u32>(textureHeaders[index].textureType));
                return {};
        }
    }
}
