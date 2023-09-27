// SPDX-License-Identifier: MPL-2.0
// Copyright © 2020 Skyline Team and Contributors (https://github.com/skyline-emu/)

#include <gpu.h>
#include <kernel/memory.h>
#include <kernel/types/KProcess.h>
#include <common/settings.h>
#include <mutex>
#include <vulkan/vulkan_enums.hpp>
#include "host_texture.h"
#include "texture.h"
#include "formats.h"
#include "host_compatibility.h"
#include <gpu/interconnect/command_executor.h>

namespace skyline::gpu {
    void Texture::SetupGuestMappings() {
        auto mappings{guest.mappings}; //!< We make a copy here on purpose

        if (mappings.size() == 1) {
            auto mapping{mappings.front()};
            u8 *alignedData{util::AlignDown(mapping.data(), constant::PageSize)};
            size_t alignedSize{static_cast<size_t>(util::AlignUp(mapping.end().base(), constant::PageSize) - alignedData)};

            mirror = gpu.state.process->memory.CreateMirror(span<u8>{alignedData, alignedSize});
            mirror = mirror.subspan(static_cast<size_t>(mapping.data() - alignedData), mapping.size());
        } else {
            std::vector<span<u8>> alignedMappings;

            const auto &frontMapping{mappings.front()};
            u8 *alignedData{util::AlignDown(frontMapping.data(), constant::PageSize)};
            alignedMappings.emplace_back(alignedData, frontMapping.end().base() - alignedData);

            size_t totalSize{frontMapping.size()};
            if (mappings.size() != 2)
                for (auto it{std::next(mappings.begin())}; it != std::prev(mappings.end()); ++it) {
                    size_t mappingSize{it->size()};
                    alignedMappings.emplace_back(it->data(), mappingSize);
                    totalSize += mappingSize;
                }

            const auto &backMapping{mappings.back()};
            totalSize += backMapping.size();
            alignedMappings.emplace_back(backMapping.data(), util::AlignUp(backMapping.size(), constant::PageSize));

            mirror = gpu.state.process->memory.CreateMirrors(alignedMappings);
            mirror = mirror.subspan(static_cast<size_t>(frontMapping.data() - alignedData), totalSize);
        }

        mappings.erase(std::remove_if(mappings.begin(), mappings.end(), [](const auto &mapping){
            return !mapping.valid();
        }), mappings.end());

        if (!mappings.empty()) {
            // We can't just capture `this` in the lambda since the lambda could exceed the lifetime of the texture
            std::weak_ptr<Texture> weakThis{weak_from_this()};
            trapHandle = gpu.state.nce->CreateTrap(mappings, [weakThis] {
                auto texture{weakThis.lock()};
                if (!texture)
                    return;

                std::unique_lock lock{texture->userMutex};
                std::unique_lock accessLock{texture->accessMutex};
                bool shouldWait{};
                for (auto &host : texture->hosts)
                    if (host.dirtyState == DirtyState::HostDirty)
                        shouldWait = true;

                if (shouldWait) {
                    // If this mutex would cause other callbacks to be blocked then we should block on this mutex in advance
                    std::shared_ptr<FenceCycle> waitCycle{};
                    do {
                        // We need to do a loop here since we can't wait with the texture locked but not doing so means that the texture could have it's cycle changed which we wouldn't wait on, loop until we are sure the cycle hasn't changed to avoid that
                        if (waitCycle) {
                            i64 startNs{texture->accumulatedGuestWaitCounter > SkipReadbackHackWaitCountThreshold ? util::GetTimeNs() : 0};
                            waitCycle->Wait();
                            if (startNs)
                                texture->accumulatedGuestWaitTime += std::chrono::nanoseconds(util::GetTimeNs() - startNs);

                            texture->accumulatedGuestWaitCounter++;
                        }

                        if (waitCycle && texture->cycle == waitCycle) {
                            texture->cycle = {};
                            waitCycle = {};
                        } else {
                            waitCycle = texture->cycle;
                        }
                    } while (waitCycle);
                }
            }, [weakThis] {
                TRACE_EVENT("gpu", "Texture::ReadTrap");

                auto texture{weakThis.lock()};
                if (!texture)
                    return true;

                std::unique_lock lock{texture->userMutex, std::try_to_lock};
                if (!lock)
                    return false;

                std::unique_lock stateLock{texture->accessMutex, std::try_to_lock};
                if (!stateLock)
                    return false;

                texture->SynchronizeGuest(false);
                return true;
            }, [weakThis] {
                TRACE_EVENT("gpu", "Texture::WriteTrap");

                auto texture{weakThis.lock()};
                if (!texture)
                    return true;

                std::unique_lock lock{texture->userMutex, std::try_to_lock};
                if (!lock)
                    return false;

                std::unique_lock stateLock{texture->accessMutex, std::try_to_lock};
                if (!stateLock)
                    return false;

                if (texture->accumulatedGuestWaitTime > SkipReadbackHackWaitTimeThreshold && *texture->gpu.state.settings->enableFastGpuReadbackHack) {
                    for (auto &host : texture->hosts) {
                        if (host.dirtyState == DirtyState::HostDirty)
                            host.dirtyState = DirtyState::Clean;
                    }
                    return true;
                }

                texture->SynchronizeGuest(true); // We need to assume the texture is dirty since we don't know what the guest is writing
                return true;
            });
        } else {
            LOGW("Completely unmapped texture!");
        }
    }

    Texture::Texture(GPU &gpu, TextureViewRequestInfo &info, bool mutableFormat) : gpu{gpu}, guest{info.mappings, info.sampleDimensions, info.viewFormat, info.tileConfig, info.levelCount, info.layerCount, info.layerStride}, mutableFormat{mutableFormat || !gpu.traits.quirks.vkImageMutableFormatCostly} {
        usageHandle = gpu.textureUsageTracker.AddTexture(*this);
    }

    Texture::~Texture() {
        std::scoped_lock destroyLock{userMutex, accessMutex};

        if (gpu.textureUsageTracker.ShouldSyncGuest(usageHandle))
            SynchronizeGuest(false);

        gpu.textureUsageTracker.RemoveTexture(usageHandle);
        if (trapHandle) [[likely]]
            gpu.state.nce->DeleteTrap(*trapHandle);
        if (mirror.valid()) [[likely]]
            munmap(util::AlignDown(mirror.data(), constant::PageSize), util::AlignUp<size_t>(static_cast<size_t>(mirror.end().base() - util::AlignDown(mirror.data(), constant::PageSize)), constant::PageSize));
    }

    void Texture::lock() {
        userMutex.lock();
    }

    bool Texture::LockWithTag(ContextTag pTag) {
        if (pTag == tag)
            return false;

        userMutex.lock();
        tag = pTag;
        return true;
    }

    void Texture::unlock() {
        tag = ContextTag{};
        userMutex.unlock();
    }

    bool Texture::try_lock() {
        if (userMutex.try_lock())
            return true;

        return false;
    }

    // TODO: Re-add adreno format aliasing (if it's even needed)
    HostTextureView *Texture::FindOrCreateView(TextureViewRequestInfo &info, vk::ImageSubresourceRange viewRange, u32 actualBaseMip) {
        std::unique_lock lock{accessMutex};

        auto createView{[this](HostTexture &host, texture::Format viewFormat, vk::ImageViewType viewType, vk::ImageSubresourceRange range, vk::ComponentMapping components) {
            vk::ImageViewCreateInfo createInfo{
                .image = host.backing.vkImage,
                .viewType = viewType,
                .format = viewFormat->vkFormat,
                .components = components,
                .subresourceRange = range
            };

            auto view{gpu.texture.viewAllocatorState.EmplaceUntracked<HostTextureView>(&host, viewType, viewFormat, components, range, vk::raii::ImageView{gpu.vkDevice, createInfo})};
            host.views.emplace_back(view);
            return view;
        }};

        vk::ImageType imageType{HostTexture::ConvertViewType(info.viewType, info.imageDimensions)};
        for (auto &host : hosts) {
            if (host.copyLayouts[actualBaseMip].dimensions == info.imageDimensions && host.imageType == imageType && host.sampleCount == info.sampleCount) {
                auto candidateFormat{info.viewFormat == host.guestFormat ? host.format : info.viewFormat}; // We want to use the texture's format if it isn't supplied or if the requested format matches the guest format then we want to use the host format just in case it's compressed

                if ((host.usage & info.extraUsageFlags) != info.extraUsageFlags)
                    continue; // If the host image lacks required usage flags then we can't use a view from it

                auto view{ranges::find_if(host.views, [&](HostTextureView *view) { return view->format == candidateFormat && view->type == info.viewType && view->range == viewRange && view->components == info.viewComponents; })};
                if (view != host.views.end())
                    return *view;

                if (host.needsDecompression)
                    // If the host texture needs decompression then we can't create a view with a different format, we depend on variants to handle that
                    continue;

                bool isViewFormatCompatible{texture::IsVulkanFormatCompatible(static_cast<VkFormat>(candidateFormat->vkFormat), static_cast<VkFormat>(host.format->vkFormat))};
                if (!isViewFormatCompatible)
                    continue; // If the view format isn't compatible then we can't create a view

                if (host.format == candidateFormat || host.flags & vk::ImageCreateFlagBits::eMutableFormat)
                    return createView(host, candidateFormat, info.viewType, viewRange, info.viewComponents);
                else
                    return nullptr; // We need to create a whole new texture if the texture doesn't support mutable formats
            }
        }

        if (actualBaseMip != 0)
            info.imageDimensions = info.viewFormat->GetDimensionsFromBytes(hosts.front().format->GetDimensionsInBytes(hosts.front().dimensions));

        auto &newHost = hosts.emplace_back(*this, info, imageType, mutableFormat);
        return createView(newHost, info.viewFormat, info.viewType, viewRange, info.viewComponents);
    }

    void Texture::WaitOnFence() {
        if (cycle) {
            TRACE_EVENT("gpu", "Texture::WaitOnFence");

            cycle->Wait();
            cycle = nullptr;
        }
    }

    void Texture::AttachCycle(const std::shared_ptr<FenceCycle> &lCycle) {
        if (lCycle.get() != cycle.get()) {
            lCycle->AttachObject(shared_from_this());
            lCycle->ChainCycle(cycle);
            cycle = lCycle;
        }
    }

    void Texture::MarkGpuDirty(UsageTracker &usageTracker) {
        for (auto mapping : guest.mappings)
            if (mapping.valid()) [[likely]]
                usageTracker.dirtyIntervals.Insert(mapping);
    }

    void Texture::SynchronizeHost(HostTexture &toSync, vk::PipelineStageFlags waitStage, vk::AccessFlags waitFlags) {
        TRACE_EVENT("gpu", "Texture::SynchronizeHost");

        HostTexture *toSyncFrom{};

        std::unique_lock lock{accessMutex};

        if (toSync.dirtyState == DirtyState::OtherHostDirty) {
            for (auto &host : hosts)
                if (host.dirtyState == DirtyState::HostDirty)
                    toSyncFrom = &host;

            toSync.dirtyState = DirtyState::HostDirty;
        } else {
            if (toSync.dirtyState != DirtyState::GuestDirty) {
                AttachCycle(gpu.scheduler.Submit([&](vk::raii::CommandBuffer &commandBuffer) {
                    toSync.AccessForTransfer(commandBuffer, toSync.trackingInfo, false);
                    toSync.trackingInfo.waitedStages |= vk::PipelineStageFlagBits::eTransfer;
                }));
                WaitOnFence();
                return;
            }

            toSync.dirtyState = DirtyState::Clean;

            gpu.state.nce->TrapRegions(*trapHandle, true); // Trap any future CPU reads (optionally) + writes to this texture
        }

        toSync.preExecDirtyState = toSync.dirtyState;

        if (toSyncFrom) {
            auto downloadStagingBuffer = gpu.memory.AllocateStagingBuffer(toSyncFrom->copySize);

            WaitOnFence();
            auto lCycle{gpu.scheduler.Submit([&](vk::raii::CommandBuffer &commandBuffer) {
                if (toSync.layout == vk::ImageLayout::eUndefined)
                    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eAllCommands, {}, {}, {}, vk::ImageMemoryBarrier{
                        .image = toSync.backing.vkImage,
                        .oldLayout = std::exchange(toSync.layout, vk::ImageLayout::eGeneral),
                        .newLayout = toSync.layout,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .subresourceRange = {
                            .aspectMask = toSync.format->vkAspect,
                            .levelCount = guest.levelCount,
                            .layerCount = guest.layerCount,
                        },
                    });
                else if (!(toSync.trackingInfo.waitedStages & vk::PipelineStageFlagBits::eTransfer)) {
                    toSync.AccessForTransfer(commandBuffer, toSync.trackingInfo, true);
                    toSync.trackingInfo.waitedStages |= vk::PipelineStageFlagBits::eTransfer;
                }

                if (!(toSyncFrom->trackingInfo.waitedStages & vk::PipelineStageFlagBits::eTransfer)) {
                    toSyncFrom->AccessForTransfer(commandBuffer, toSyncFrom->trackingInfo, false);
                    toSyncFrom->trackingInfo.waitedStages |= vk::PipelineStageFlagBits::eTransfer;
                }

                commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, vk::BufferMemoryBarrier{
                    .buffer = downloadStagingBuffer->vkBuffer,
                    .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .size = downloadStagingBuffer->size()
                }, {});
                toSyncFrom->CopyIntoStagingBuffer(commandBuffer, downloadStagingBuffer);
                commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, vk::BufferMemoryBarrier{
                    .buffer = downloadStagingBuffer->vkBuffer,
                    .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .size = downloadStagingBuffer->size()
                }, {});
                toSync.CopyFromStagingBuffer(commandBuffer, downloadStagingBuffer);
            })};
            toSyncFrom->trackingInfo.waitedStages |= vk::PipelineStageFlagBits::eTransfer;
            lCycle->Wait(); // We block till the copy is complete
        } else {
            auto stagingBuffer{toSync.SynchronizeHostImpl()};
            if (stagingBuffer) {
                WaitOnFence();
                auto lCycle{gpu.scheduler.Submit([&](vk::raii::CommandBuffer &commandBuffer) {
                    if (toSync.layout == vk::ImageLayout::eUndefined)
                        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eAllCommands, {}, {}, {}, vk::ImageMemoryBarrier{
                            .image = toSync.backing.vkImage,
                            .oldLayout = std::exchange(toSync.layout, vk::ImageLayout::eGeneral),
                            .newLayout = toSync.layout,
                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                            .subresourceRange = {
                                .aspectMask = toSync.format->vkAspect,
                                .levelCount = guest.levelCount,
                                .layerCount = guest.layerCount,
                            }
                        });
                    else if (!(toSync.trackingInfo.waitedStages & vk::PipelineStageFlagBits::eTransfer)) {
                        toSync.AccessForTransfer(commandBuffer, toSync.trackingInfo, true);
                        toSync.trackingInfo.waitedStages |= vk::PipelineStageFlagBits::eTransfer;
                    }

                    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, vk::BufferMemoryBarrier{
                        .buffer = stagingBuffer->vkBuffer,
                        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                        .dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .size = stagingBuffer->size()
                    }, {});
                    toSync.CopyFromStagingBuffer(commandBuffer, stagingBuffer);
                })};
                lCycle->Wait(); // We block till the copy is complete
            }
        }
    }

    void Texture::SynchronizeHostInline(const vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &pCycle, HostTexture &toSync) {
        TRACE_EVENT("gpu", "Texture::SynchronizeHostInline");

        // NOTE: dirtyState is managed by the caller
        auto stagingBuffer{toSync.SynchronizeHostImpl()};
        if (stagingBuffer) {
            pCycle->AttachObjects(stagingBuffer->shared_from_this());
            toSync.CopyFromStagingBuffer(commandBuffer, stagingBuffer);
        }
    }

    void Texture::SynchronizeGuest(bool isWritten) {
        TRACE_EVENT("gpu", "Texture::SynchronizeGuest");

        HostTexture *toSyncFrom{};
        for (auto &host : hosts) {
            if (host.dirtyState == DirtyState::HostDirty) {
                host.dirtyState = isWritten ? DirtyState::GuestDirty : DirtyState::Clean;
                toSyncFrom = &host;
            } else if ((host.dirtyState == DirtyState::Clean && isWritten) || host.dirtyState == DirtyState::OtherHostDirty) {
                host.dirtyState = DirtyState::GuestDirty;
            }
            host.preExecDirtyState = host.dirtyState;
        }

        if (!toSyncFrom) // If state is already CPU dirty/Clean we don't need to do anything
            return;

        if (toSyncFrom->GetLayout() == vk::ImageLayout::eUndefined || toSyncFrom->needsDecompression) // We cannot sync the contents of an undefined texture and we don't support recompression of a decompressed texture
            return;

        gpu.textureUsageTracker.MarkClean(usageHandle);

        if (toSyncFrom->tiling == vk::ImageTiling::eOptimal) {
            auto tempStagingBuffer = gpu.memory.AllocateStagingBuffer(toSyncFrom->copySize);

            WaitOnFence();
            auto lCycle{gpu.scheduler.Submit([&](vk::raii::CommandBuffer &commandBuffer) {
                if (!(toSyncFrom->trackingInfo.waitedStages & vk::PipelineStageFlagBits::eTransfer)) {
                    toSyncFrom->AccessForTransfer(commandBuffer, toSyncFrom->trackingInfo, false);
                    toSyncFrom->trackingInfo.waitedStages |= vk::PipelineStageFlagBits::eTransfer;
                }

                toSyncFrom->CopyIntoStagingBuffer(commandBuffer, tempStagingBuffer);
                commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, {}, vk::BufferMemoryBarrier{
                    .buffer = tempStagingBuffer->vkBuffer,
                    .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .dstAccessMask = vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eHostWrite,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .size = tempStagingBuffer->size()
                }, {});
            })};
            lCycle->Wait(); // We block till the copy is complete

            toSyncFrom->CopyToGuest(tempStagingBuffer->data());
        } else if (toSyncFrom->tiling == vk::ImageTiling::eLinear) {
            // We can optimize linear texture sync on a UMA by mapping the texture onto the CPU and copying directly from it rather than using a staging buffer
            WaitOnFence();
            toSyncFrom->CopyToGuest(toSyncFrom->backing.data());
        } else [[unlikely]] {
            throw exception("Host -> Guest synchronization of images tiled as '{}' isn't implemented", vk::to_string(toSyncFrom->tiling));
        }
    }

    void Texture::OnExecStart(interconnect::CommandExecutor &executor) {
        std::unique_lock lock{accessMutex};

        executor.InsertPreExecuteCommand([this](vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &pCycle, GPU &) {
            std::unique_lock lock{accessMutex};
            for (auto &host : hosts) {
                if (host.layout == vk::ImageLayout::eUndefined)
                    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eAllCommands, {}, {}, {}, vk::ImageMemoryBarrier{
                        .image = host.backing.vkImage,
                        .oldLayout = std::exchange(host.layout, vk::ImageLayout::eGeneral),
                        .newLayout = host.layout,
                        .dstAccessMask = vk::AccessFlagBits::eMemoryWrite,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .subresourceRange = {
                            .aspectMask = host.format->vkAspect,
                            .levelCount = guest.levelCount,
                            .layerCount = guest.layerCount,
                        },
                    });

                host.UpdateRenderPassUsage(0, texture::RenderPassUsage::None);
            }
        });
    }

    void Texture::OnExecEnd() {
        std::unique_lock lock{accessMutex};

        if (syncStagingBuffer)
            syncStagingBuffer = nullptr;

        for (auto &host : hosts)
            host.preExecDirtyState = host.dirtyState;

        unlock();
    }
}
