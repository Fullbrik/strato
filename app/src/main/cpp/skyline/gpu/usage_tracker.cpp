// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2023 Strato Team and Contributors (https://github.com/strato-emu/)

#include "gpu.h"
#include "usage_tracker.h"
#include "interconnect/command_executor.h"
#include <utility>
#include <vulkan/vulkan_funcs.hpp>

namespace skyline::gpu {
    void TextureUsageTracker::FilterOverlaps(std::vector<std::reference_wrapper<TextureUsageInfo>> &intervals, boost::container::small_vector<span<u8>, 3> &onArea) {

    }

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

        bool usedfastPath{true};
        std::vector<texture::TextureCopies> bufferImageCopies{};

        for (auto &overlap : overlaps) {
            if (auto imageCopy{texture->value.texture->guest.CalculateCopy(overlap.get().texture->guest, toSync->format, overlap.get().dirtyTexture->format)}) {
                bufferImageCopies.emplace_back(*imageCopy);
            } else {
                usedfastPath = false;
                break;
            }
        }

        ++overallCount;

        if (usedfastPath) {
            ++fastPathCount;

            u32 i{};
            for (auto &overlap : overlaps) {
                executor.AttachTexture(overlap.get().texture->shared_from_this());

                auto &textureCopy{bufferImageCopies[i]};

                if (textureCopy.stagingBufferSize) {
                    std::unique_lock lock2{overlap.get().texture->accessMutex};

                    auto stagingBuffer{gpu.memory.AllocateStagingBuffer(textureCopy.stagingBufferSize)};
                    executor.cycle->AttachObject(stagingBuffer->shared_from_this());

                    executor.AddOutsideRpCommand([stagingBuffer = stagingBuffer->shared_from_this(), overlap = overlap.get(), toSync, overlapTrackingInfo = overlap.get().dirtyTexture->trackingInfo, trackingInfo = toSync->trackingInfo, bufferImageCopy = textureCopy](vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &cycle, GPU &) {
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

                ++i;
            }

            for (auto &host : texture->value.texture->hosts)
                host.dirtyState = HostTexture::DirtyState::OtherHostDirty;

            toSync->dirtyState = HostTexture::DirtyState::HostDirty;
            texture->value.dirtyTexture = toSync;

            //Logger::Info("(TEST) fast: {}, overall: {}, {}%", fastPathCount, overallCount, (static_cast<double>(fastPathCount) / static_cast<double>(overallCount)) * 100.0);

            return true;
        }

        return true;

        std::erase_if(overlaps, [](const auto &overlap) {
            return overlap.get().dirtyTexture->preExecDirtyState != HostTexture::DirtyState::HostDirty;
        });

        if (overlaps.empty()) {
            if (markDirty)
                texture->value.dirtyTexture = toSync;
            return false;
        }

        FilterOverlaps(overlaps, texture->value.texture->guest.mappings);

        if (texture->value.dirtyTexture && texture->value.dirtyTexture->dirtyState == HostTexture::DirtyState::HostDirty)
            overlaps.emplace(overlaps.begin(), std::ref(texture->value));

        // TODO: Decide which approach is faster
        #if 1

        std::vector<vk::ImageMemoryBarrier> imageBarriers;
        imageBarriers.reserve(overlaps.size());
        std::vector<vk::BufferMemoryBarrier> bufferBarriers;
        bufferBarriers.reserve(overlaps.size());
        std::vector<std::shared_ptr<memory::StagingBuffer>> stagingBuffers{};
        stagingBuffers.reserve(overlaps.size());

        auto cycle{gpu.scheduler.Submit([&](vk::raii::CommandBuffer &commandBuffer) {
            vk::PipelineStageFlags srcStages{};

            for (auto &overlap : overlaps) {
                std::unique_lock lock2{overlap.get().texture->accessMutex};

                for (auto &host : overlap.get().texture->hosts)
                    host.dirtyState = HostTexture::DirtyState::GuestDirty;

                if (!overlap.get().texture->syncStagingBuffer)
                    overlap.get().texture->syncStagingBuffer = gpu.memory.AllocateStagingBuffer(overlap.get().dirtyTexture->copySize);

                stagingBuffers.push_back(overlap.get().texture->syncStagingBuffer);

                srcStages |= overlap.get().dirtyTexture->trackingInfo.lastUsedStage;
                imageBarriers.emplace_back(vk::ImageMemoryBarrier{
                    .image = overlap.get().dirtyTexture->GetImage(),
                    .srcAccessMask = overlap.get().dirtyTexture->trackingInfo.lastUsedAccessFlag,
                    .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                    .oldLayout = overlap.get().dirtyTexture->GetLayout(),
                    .newLayout = overlap.get().dirtyTexture->GetLayout(),
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .subresourceRange = {
                        .aspectMask = overlap.get().dirtyTexture->format->vkAspect,
                        .levelCount = overlap.get().texture->guest.levelCount,
                        .layerCount = overlap.get().texture->guest.layerCount,
                    }
                });

                bufferBarriers.emplace_back(vk::BufferMemoryBarrier{
                    .buffer = overlap.get().texture->syncStagingBuffer->vkBuffer,
                    .srcAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                    .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .size = overlap.get().texture->syncStagingBuffer->size()
                });
            }

            commandBuffer.pipelineBarrier(srcStages | vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, vk::ArrayProxy<vk::BufferMemoryBarrier>(static_cast<u32>(bufferBarriers.size()), bufferBarriers.data()), vk::ArrayProxy<vk::ImageMemoryBarrier>(static_cast<u32>(imageBarriers.size()), imageBarriers.data()));

            for (auto &overlap : overlaps)
                overlap.get().dirtyTexture->CopyIntoStagingBuffer(commandBuffer, overlap.get().texture->syncStagingBuffer);

            for (auto &bufferBarrier : bufferBarriers) {
                bufferBarrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
                bufferBarrier.dstAccessMask = vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eHostWrite;
            }

            commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, {}, vk::ArrayProxy<vk::BufferMemoryBarrier>(static_cast<u32>(bufferBarriers.size()), bufferBarriers.data()), {});
        })};

        for (auto &stagingBuffer : stagingBuffers)
            cycle->AttachObject(stagingBuffer);

        for (auto &overlap : overlaps)
            cycle->AttachObject(overlap.get().texture->shared_from_this());

        cycle->Wait();

        u32 i{};
        for (auto &overlap : overlaps) {
            overlap.get().dirtyTexture->CopyToGuest(stagingBuffers[i]->data());
            overlap.get().dirtyTexture = nullptr;

            //gpu.state.nce->DeleteTrap(*overlap.get().texture->trapHandle);

            ++i;
        }

        #else

        std::vector<std::shared_ptr<FenceCycle>> cycles{};
        cycles.reserve(overlaps.size());
        std::vector<std::shared_ptr<memory::StagingBuffer>> stagingBuffers{};
        stagingBuffers.reserve(overlaps.size());

        for (auto &overlap : overlaps) {
            std::unique_lock lock2{overlap.get().texture->accessMutex};

            for (auto &host : overlap.get().texture->hosts)
                host.dirtyState = HostTexture::DirtyState::GuestDirty;

            if (!overlap.get().texture->syncStagingBuffer)
                overlap.get().texture->syncStagingBuffer = gpu.memory.AllocateStagingBuffer(overlap.get().dirtyTexture->copySize);

            stagingBuffers.push_back(overlap.get().texture->syncStagingBuffer);

            cycles.emplace_back(gpu.scheduler.Submit([texture = overlap.get().texture, dirtyTexture = overlap.get().dirtyTexture, trackingInfo = overlap.get().dirtyTexture->trackingInfo](vk::raii::CommandBuffer &commandBuffer) {
                if (!(trackingInfo.waitedStages & vk::PipelineStageFlagBits::eTransfer))
                    dirtyTexture->AccessForTransfer(commandBuffer, trackingInfo, false);

                commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, vk::BufferMemoryBarrier{
                    .buffer = texture->syncStagingBuffer->vkBuffer,
                    .srcAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                    .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .size = texture->syncStagingBuffer->size()
                }, {});

                dirtyTexture->CopyIntoStagingBuffer(commandBuffer, texture->syncStagingBuffer);

                commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, {}, vk::BufferMemoryBarrier{
                    .buffer = texture->syncStagingBuffer->vkBuffer,
                    .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .dstAccessMask = vk::AccessFlagBits::eHostRead,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .size = texture->syncStagingBuffer->size()
                }, {});
            }));

            cycles.back()->AttachObjects(overlap.get().texture->shared_from_this(), overlap.get().texture->syncStagingBuffer->shared_from_this());
        }

        u32 i{};
        for (auto &cycle : cycles) {
            cycle->Wait();

            overlaps[i].get().dirtyTexture->CopyToGuest(stagingBuffers[i]->data());
            overlaps[i].get().dirtyTexture = nullptr;

            ++i;
        }

        #endif

        executor.AddOutsideRpCommand([texture = texture->value.texture, toSync](vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &cycle, GPU &) {
            texture->SynchronizeHostInline(commandBuffer, cycle, *toSync);
        });

        if (markDirty) {
            for (auto &host : texture->value.texture->hosts)
                host.dirtyState = HostTexture::DirtyState::OtherHostDirty;

            toSync->dirtyState = HostTexture::DirtyState::HostDirty;
            texture->value.dirtyTexture = toSync;
        } else {
            for (auto &host : texture->value.texture->hosts)
                host.dirtyState = HostTexture::DirtyState::GuestDirty;

            toSync->dirtyState = HostTexture::DirtyState::Clean;
            texture->value.dirtyTexture = nullptr;
        }

        //Logger::Info("(TEST) fast: {}, overall: {}, {}%", fastPathCount, overallCount, (static_cast<double>(fastPathCount) / static_cast<double>(overallCount)) * 100.0);

        return true;
    }
}
