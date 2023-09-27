// SPDX-License-Identifier: MPL-2.0
// Copyright © 2021 Skyline Team and Contributors (https://github.com/skyline-emu/)

#pragma once

#include "guest_texture.h"

namespace skyline::gpu::texture {
    /**
     * @return The size of a layer of the specified non-mipmapped block-slinear surface in bytes
     */
    size_t GetBlockLinearLayerSize(Dimensions dimensions,
                                   size_t formatBlockWidth, size_t formatBlockHeight, size_t formatBpb,
                                   size_t gobBlockHeight, size_t gobBlockDepth, size_t sparseBlockWidth);

    /**
     * @param isMultiLayer If the texture has more than one layer, a multi-layer texture requires alignment to a block at layer end
     * @return The size of a layer of the specified block-linear surface in bytes
     */
    size_t GetBlockLinearLayerSize(Dimensions dimensions,
                                   size_t formatBlockHeight, size_t formatBlockWidth, size_t formatBpb,
                                   size_t gobBlockHeight, size_t gobBlockDepth, size_t sparseBlockWidth,
                                   size_t levelCount, bool isMultiLayer);

    /**
     * @return A vector of metadata about every mipmapped level of the supplied block-linear surface
     */
    std::vector<MipLevelLayout> CalculateMipLayout(Dimensions dimensions,
                                                   size_t formatBlockHeight, size_t formatBlockWidth, size_t formatBpb,
                                                   size_t gobBlockHeight, size_t gobBlockDepth, size_t sparseBlockWidth,
                                                   size_t levelCount);

    /**
     * @brief Copies the contents of a blocklinear texture to a linear output buffer
     */
    void CopyBlockLinearToLinear(Dimensions dimensions,
                                 size_t formatBlockWidth, size_t formatBlockHeight, size_t formatBpb,
                                 size_t gobBlockHeight, size_t gobBlockDepth, size_t sparseBlockWidth,
                                 u8 * __restrict blockLinear, u8 * __restrict linear);

    /**
     * @brief Copies the contents of a blocklinear texture to a pitch texture
     */
    void CopyBlockLinearToPitch(Dimensions dimensions,
                                size_t formatBlockWidth, size_t formatBlockHeight, size_t formatBpb, u32 pitchAmount,
                                size_t gobBlockHeight, size_t gobBlockDepth, size_t sparseBlockWidth,
                                u8 * __restrict blockLinear, u8 * __restrict pitch);

    /**
     * @brief Copies the contents of a part of a blocklinear texture to a pitch texture
     */
    void CopyBlockLinearToPitchSubrect(Dimensions pitchDimensions, Dimensions blockLinearDimensions,
                                       size_t formatBlockWidth, size_t formatBlockHeight, size_t formatBpb, u32 pitchAmount,
                                       size_t gobBlockHeight, size_t gobBlockDepth,
                                       u8 * __restrict blockLinear, u8 * __restrict pitch,
                                       u32 originX, u32 originY);

    /**
     * @brief Copies the contents of a linear buffer to a blocklinear texture
     */
    void CopyLinearToBlockLinear(Dimensions dimensions,
                                size_t formatBlockWidth, size_t formatBlockHeight, size_t formatBpb,
                                size_t gobBlockHeight, size_t gobBlockDepth, size_t sparseBlockWidth,
                                u8 * __restrict linear, u8 * __restrict blockLinear);

    /**
     * @brief Copies the contents of a pitch texture to a blocklinear texture
     */
    void CopyPitchToBlockLinear(Dimensions dimensions,
                                 size_t formatBlockWidth, size_t formatBlockHeight, size_t formatBpb, u32 pitchAmount,
                                 size_t gobBlockHeight, size_t gobBlockDepth, size_t sparseBlockWidth,
                                 u8 * __restrict pitch, u8 * __restrict blockLinear);

    /**
     * @brief Copies the contents of a linear texture to a part of a blocklinear texture
     */
    void CopyLinearToBlockLinearSubrect(Dimensions linearDimensions, Dimensions blockLinearDimensions,
                                       size_t formatBlockWidth, size_t formatBlockHeight, size_t formatBpb,
                                       size_t gobBlockHeight, size_t gobBlockDepth,
                                       u8 * __restrict linear, u8 * __restrict blockLinear,
                                        u32 originX, u32 originY);

    /**
     * @brief Copies the contents of a pitch texture to a part of a blocklinear texture
     */
    void CopyPitchToBlockLinearSubrect(Dimensions pitchDimensions, Dimensions blockLinearDimensions,
                                 size_t formatBlockWidth, size_t formatBlockHeight, size_t formatBpb, u32 pitchAmount,
                                 size_t gobBlockHeight, size_t gobBlockDepth,
                                 u8 * __restrict pitch, u8 * __restrict blockLinear,
                                 u32 originX, u32 originY);

    /**
     * @brief Copies the contents of a pitch-linear guest texture to a linear output buffer
     * @note This does not support 3D textures
     */
    void CopyPitchLinearToLinear(const GuestTexture &guest, u8 * __restrict guestInput, u8 * __restrict linearOutput);

    /**
     * @brief Copies the contents of a linear buffer to a pitch-linear guest texture
     * @note This does not support 3D textures
     */
    void CopyLinearToPitchLinear(const GuestTexture &guest, u8 * __restrict linearInput, u8 * __restrict guestOutput);
}
