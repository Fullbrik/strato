// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2023 Strato Team and Contributors (https://github.com/strato-emu/)

#include "gpu.h"
#include "usage_tracker.h"
#include "interconnect/command_executor.h"
#include <utility>
#include <vulkan/vulkan_funcs.hpp>

namespace skyline::gpu {
    TextureUsageTracker::TextureUsageTracker(GPU &gpu) : gpu{gpu}, infoMap{} {}

    TextureUsageTracker::TextureHandle TextureUsageTracker::AddTexture(Texture &texture) {
        std::unique_lock lock{mutex};
        return infoMap.Insert(span<span<u8>>{texture.guest.mappings}, TextureUsageInfo{ .texture = &texture });
    }

    void TextureUsageTracker::MarkClean(TextureHandle texture) {
        std::unique_lock lock{mutex};
        ++lastSequence;

        texture->value.dirtyTexture = nullptr;
        texture->value.sequence = lastSequence;
    }

    bool TextureUsageTracker::ShouldSyncGuest(TextureHandle texture) {
        std::unique_lock lock{mutex};

        std::vector<TextureMap::Interval> intervals{};
        for (const auto &mapping : texture->value.texture->guest.mappings)
            intervals.emplace_back(mapping.data(), mapping.end().base());

        auto overlaps{infoMap.GetRange(intervals)};

        std::erase_if(overlaps, [&texture](const auto &overlap) {
            return !overlap.get().dirtyTexture || overlap.get().sequence <= texture->value.sequence || overlap.get().texture == texture->value.texture || overlap.get().dirtyTexture->needsDecompression;
        });

        if (overlaps.empty())
            return true;
        else
            return false;
    }

    bool TextureUsageTracker::ShouldSyncHost(TextureHandle texture) {
        std::unique_lock lock{mutex};

        std::vector<TextureMap::Interval> intervals{};
        for (const auto &mapping : texture->value.texture->guest.mappings)
            intervals.emplace_back(mapping.data(), mapping.end().base());

        auto overlaps{infoMap.GetRange(intervals)};

        std::erase_if(overlaps, [&texture](const auto &overlap) {
            return !overlap.get().dirtyTexture || overlap.get().sequence <= texture->value.sequence || overlap.get().texture == texture->value.texture || overlap.get().dirtyTexture->needsDecompression || overlap.get().dirtyTexture->dirtyState != HostTexture::DirtyState::HostDirty;
        });

        if (overlaps.empty())
            return false;
        else
            return true;
    }

    void TextureUsageTracker::RemoveTexture(TextureHandle texture) {
        std::unique_lock lock{mutex};
        infoMap.Remove(texture);
    }

    bool TextureUsageTracker::RequestSync(interconnect::CommandExecutor &executor, TextureHandle texture, HostTexture *toSync, bool markDirty) {
        std::unique_lock lock{mutex};
        ++lastSequence;

        std::vector<TextureMap::Interval> intervals{};
        for (const auto &mapping : toSync->guest.mappings)
            intervals.emplace_back(mapping.data(), mapping.end().base());

        auto overlaps{infoMap.GetRange(intervals)};

        std::erase_if(overlaps, [&texture](const auto &overlap) {
            return !overlap.get().dirtyTexture || overlap.get().sequence <= texture->value.sequence || overlap.get().texture == texture->value.texture || overlap.get().dirtyTexture->needsDecompression || overlap.get().dirtyTexture->dirtyState != HostTexture::DirtyState::HostDirty;
        });

        texture->value.sequence = lastSequence;

        if (overlaps.empty()) {
            if (markDirty)
                texture->value.dirtyTexture = toSync;
            return false;
        }

        std::sort(overlaps.begin(), overlaps.end(), [](const auto &overlap, const auto &overlap2) {
            return overlap.get().sequence < overlap2.get().sequence;
        });

        std::vector<std::optional<texture::TextureCopies>> bufferImageCopies{};

        for (auto &overlap : overlaps)
            bufferImageCopies.emplace_back(texture->value.texture->guest.CalculateCopy(overlap.get().texture->guest, toSync->format, overlap.get().dirtyTexture->format));

        u32 i{};
        for (auto &overlap : overlaps) {
            if (auto &textureCopy{bufferImageCopies[i]}) {
                executor.AttachTexture(overlap.get().texture->shared_from_this());

                if (textureCopy->stagingBufferSize) {
                    std::unique_lock lock2{overlap.get().texture->accessMutex};

                    auto stagingBuffer{gpu.memory.AllocateStagingBuffer(textureCopy->stagingBufferSize)};
                    executor.cycle->AttachObject(stagingBuffer->shared_from_this());

                    executor.AddOutsideRpCommand([stagingBuffer = stagingBuffer->shared_from_this(), overlap = overlap.get(), toSync, overlapTrackingInfo = overlap.get().dirtyTexture->trackingInfo, trackingInfo = toSync->trackingInfo, bufferImageCopy = *textureCopy](vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &cycle, GPU &) {
                        if (!(trackingInfo.waitedStages & vk::PipelineStageFlagBits::eTransfer))
                            overlap.dirtyTexture->AccessForTransfer(commandBuffer, overlapTrackingInfo, false);
                        toSync->AccessForTransfer(commandBuffer, trackingInfo, true);

                        commandBuffer.copyImageToBuffer(overlap.dirtyTexture->GetImage(), overlap.dirtyTexture->GetLayout(), stagingBuffer->vkBuffer, vk::ArrayProxy<vk::BufferImageCopy>{static_cast<u32>(bufferImageCopy.toStaging.size()), const_cast<vk::BufferImageCopy *>(bufferImageCopy.toStaging.data())});
                        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, vk::BufferMemoryBarrier{
                            .buffer = stagingBuffer->vkBuffer,
                            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                            .dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .size = stagingBuffer->size()
                        }, {});
                        commandBuffer.copyBufferToImage(stagingBuffer->vkBuffer, toSync->GetImage(), toSync->GetLayout(), vk::ArrayProxy<vk::BufferImageCopy>{static_cast<u32>(bufferImageCopy.fromStaging.size()), const_cast<vk::BufferImageCopy *>(bufferImageCopy.fromStaging.data())});
                    });

                    overlap.get().dirtyTexture->trackingInfo.waitedStages |= vk::PipelineStageFlagBits::eTransfer;

                    toSync->trackingInfo.lastUsedStage = vk::PipelineStageFlagBits::eTransfer;
                    toSync->trackingInfo.lastUsedAccessFlag = vk::AccessFlagBits::eTransferWrite;
                    toSync->trackingInfo.waitedStages = {};
                }
            }

            ++i;
        }

        for (auto &host : texture->value.texture->hosts)
            host.dirtyState = HostTexture::DirtyState::OtherHostDirty;

        toSync->dirtyState = HostTexture::DirtyState::HostDirty;
        texture->value.dirtyTexture = toSync;

        return true;
    }
}
