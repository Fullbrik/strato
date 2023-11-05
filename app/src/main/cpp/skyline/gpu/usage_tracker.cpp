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
            return !overlap.get().dirtyTexture || overlap.get().sequence < texture->value.sequence || overlap.get().texture == texture->value.texture || overlap.get().dirtyTexture->needsDecompression;
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
            return !overlap.get().dirtyTexture || overlap.get().sequence <= texture->value.sequence || overlap.get().dirtyTexture->needsDecompression;
        });

        if (overlaps.empty())
            return false;
        else
            return true;
    }

    TextureUsageTracker::Overlaps TextureUsageTracker::GetOverlaps(TextureHandle texture) {
        std::unique_lock lock{mutex};

        std::vector<TextureMap::Interval> intervals{};
        for (const auto &mapping : texture->value.texture->guest.mappings)
            intervals.emplace_back(mapping.data(), mapping.end().base());

        auto overlaps{infoMap.GetRange(intervals)};

        std::erase_if(overlaps, [&texture](const auto &overlap) {
            return !overlap.get().dirtyTexture || overlap.get().sequence <= texture->value.sequence || overlap.get().dirtyTexture->needsDecompression;
        });

        return overlaps;
    }

    void TextureUsageTracker::RemoveTexture(TextureHandle texture) {
        std::unique_lock lock{mutex};
        infoMap.Remove(texture);
    }

    void TextureUsageTracker::RequestSync(interconnect::CommandExecutor &executor, TextureHandle texture, HostTexture *toSync, const TextureSyncRequestArgs &args, bool createTransferPass) {
        std::unique_lock syncLock{mutex};
        if (incrementSequence) [[likely]]
            ++lastSequence;

        std::vector<TextureMap::Interval> intervals{};
        for (const auto &mapping : toSync->guest.mappings)
            intervals.emplace_back(mapping.data(), mapping.end().base());

        auto overlaps{infoMap.GetRange(intervals)};

        std::erase_if(overlaps, [&texture](const auto &overlap) {
            return !overlap.get().dirtyTexture || overlap.get().sequence <= texture->value.sequence || overlap.get().dirtyTexture->needsDecompression || overlap.get().dirtyTexture->dirtyState != HostTexture::DirtyState::HostDirty;
        });

        texture->value.sequence = lastSequence;

        if (overlaps.empty()) {
            if (args.isWritten)
                texture->value.dirtyTexture = toSync;
            return;
        }

        std::sort(overlaps.begin(), overlaps.end(), [](const auto &overlap, const auto &overlap2) {
            return overlap.get().sequence < overlap2.get().sequence;
        });

        std::vector<std::optional<texture::TextureCopies>> bufferImageCopies{};

        for (auto &overlap : overlaps)
            bufferImageCopies.emplace_back(texture->value.texture->guest.CalculateCopy(overlap.get().texture->guest, toSync->format, overlap.get().dirtyTexture->format));

        TextureSyncRequestArgs transferArgs{
            .isReadInTP = false,
            .isRead = false,
            .isWritten = false,
            .usedStage = vk::PipelineStageFlagBits::eTransfer,
            .usedFlags = vk::AccessFlagBits::eTransferWrite
        };

        for (u32 i{}; i < overlaps.size();) {
            if (bufferImageCopies[i].has_value()) {
                ++i;
            } else {
                bufferImageCopies.erase(bufferImageCopies.begin() + i);
                overlaps.erase(overlaps.begin() + i);
            }
        }

        u32 i{};
        for (auto &overlap : overlaps) {
            if (auto &textureCopy{*bufferImageCopies[i]}; textureCopy.stagingBufferSize) {
                executor.AttachTexture(overlap.get().texture->shared_from_this());

                std::unique_lock textureLock{overlap.get().texture->accessMutex};

                if (createTransferPass) {
                    executor.CreateTransferPass();
                    createTransferPass = false;
                } else {
                    executor.AddTransferSubpass();
                }

                auto stagingBuffer{gpu.memory.AllocateStagingBuffer(textureCopy.stagingBufferSize)};
                executor.cycle->AttachObject(stagingBuffer->shared_from_this());

                executor.AddTextureBarrier(*overlap.get().dirtyTexture, {
                    .isReadInTP = true,
                    .isRead = false,
                    .isWritten = false,
                    .usedStage = vk::PipelineStageFlagBits::eTransfer,
                    .usedFlags = vk::AccessFlagBits::eTransferRead
                });

                const TextureSyncRequestArgs &syncArgs{i == (overlaps.size() - 1) ? args : transferArgs};

                executor.AddStagedTextureTransferCommand(*toSync, syncArgs, *stagingBuffer, [overlap = overlap.get(), stagingBuffer = stagingBuffer->shared_from_this(), textureCopy](const vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &, GPU &){
                    commandBuffer.copyImageToBuffer(overlap.dirtyTexture->GetImage(), overlap.dirtyTexture->GetLayout(), stagingBuffer->vkBuffer, textureCopy.toStaging);
                }, [toSync, stagingBuffer = stagingBuffer->shared_from_this(), textureCopy](const vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &, GPU &){
                    commandBuffer.copyBufferToImage(stagingBuffer->vkBuffer, toSync->GetImage(), toSync->GetLayout(), textureCopy.fromStaging);
                });
            }

            ++i;
        }

        for (auto &host : texture->value.texture->hosts)
            host.dirtyState = HostTexture::DirtyState::OtherHostDirty;

        toSync->dirtyState = HostTexture::DirtyState::HostDirty;

        if (args.isWritten) {
            toSync->trackingInfo.lastUsedStage = args.usedStage;
            toSync->trackingInfo.lastUsedAccessFlag = args.usedFlags;

            toSync->trackingInfo.waitedStages = {};
        } else {
            toSync->trackingInfo.lastUsedStage = vk::PipelineStageFlagBits::eTransfer;
            toSync->trackingInfo.lastUsedAccessFlag = vk::AccessFlagBits::eTransferWrite;

            toSync->trackingInfo.waitedStages = args.usedStage;
        }

        texture->value.dirtyTexture = toSync;
    }
}
