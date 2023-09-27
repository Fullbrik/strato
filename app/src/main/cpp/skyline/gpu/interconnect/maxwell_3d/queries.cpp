// SPDX-License-Identifier: MPL-2.0
// Copyright © 2023 Skyline Team and Contributors (https://github.com/skyline-emu/)

#include <gpu.h>
#include <soc/gm20b/channel.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include "queries.h"

namespace skyline::gpu::interconnect::maxwell3d {
    Queries::Counter::Counter(vk::raii::Device &device, vk::QueryType type) : pool{device, vk::QueryPoolCreateInfo{
        .queryType = type,
        .queryCount = Counter::QueryPoolSize
    }} {}

    void Queries::Counter::Reset(InterconnectContext &ctx) {
        usedQueryCount = ctx.executor.allocator->EmplaceUntracked<u32>();
        queries = ctx.executor.allocator->AllocateUntracked<Query>(Counter::QueryPoolSize);
        std::memset(queries.data(), 0, queries.size_bytes());
        recordedCopy = false;

        lastTag = ctx.executor.executionTag;
        lastRenderPassIndex = *ctx.executor.GetRenderPassIndex();

        ctx.executor.InsertPreRpCommand([&pool = pool, usedQueryCount = usedQueryCount](vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &, GPU &){
            commandBuffer.resetQueryPool(*pool, 0, *usedQueryCount);
        });
    }

    void Queries::Counter::Begin(InterconnectContext &ctx, bool atBegin) {
        if (atBegin) {
            ctx.executor.InsertRpBeginCommand([&pool = pool, queriesPtr = queries, usedQueryIndex = (*usedQueryCount) - 1](vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &, GPU &gpu) {
                if (queriesPtr[usedQueryIndex].view)
                    commandBuffer.beginQuery(*pool, usedQueryIndex, gpu.traits.supportsPreciseOcclusionQueries ? vk::QueryControlFlagBits::ePrecise : vk::QueryControlFlags{});
            });
        } else {
            ctx.executor.AddCommand([&pool = pool, queriesPtr = queries, usedQueryIndex = (*usedQueryCount) - 1](vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &, GPU &gpu) {
                if (queriesPtr[usedQueryIndex].view)
                    commandBuffer.beginQuery(*pool, usedQueryIndex, gpu.traits.supportsPreciseOcclusionQueries ? vk::QueryControlFlagBits::ePrecise : vk::QueryControlFlags{});
            });
        }
    }

    void Queries::Counter::Report(InterconnectContext &ctx, BufferView dst, std::optional<u64> timestamp) {
        if (lastTag != ctx.executor.executionTag || lastRenderPassIndex != ctx.executor.GetRenderPassIndex()) {
            Reset(ctx);

            *usedQueryCount = 1;
            Begin(ctx, true);
            End(ctx, true);
        }

        // Allocate memory for the timestamp in the megabuffer since updateBuffer can be expensive
        BufferBinding timestampBuffer{timestamp ? ctx.gpu.megaBufferAllocator.Push(ctx.executor.cycle, span<u64>(*timestamp).cast<u8>()) : BufferBinding{}};
        queries[*usedQueryCount - 1] = {false, dst, timestampBuffer};

        if (!recordedCopy) {
            ctx.executor.InsertPostRpCommand([&pool = pool, queriesPtr = queries, usedQueryCountPtr = usedQueryCount](vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &, GPU &gpu) {
                for (u32 i{}; i < *usedQueryCountPtr; i++) {
                    if (!queriesPtr[i].view)
                        continue;

                    auto dstBinding{queriesPtr[i].view.GetBinding(gpu)};
                    auto timestampSrcBinding{queriesPtr[i].timestampBinding};

                    commandBuffer.copyQueryPoolResults(*pool, i, 1, dstBinding.buffer, dstBinding.offset, 0, vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
                    if (timestampSrcBinding)
                        commandBuffer.copyBuffer(timestampSrcBinding.buffer, dstBinding.buffer, vk::BufferCopy {
                            .size = 8,
                            .srcOffset = timestampSrcBinding.offset,
                            .dstOffset = dstBinding.offset + 8
                        });
                }
            });
            recordedCopy = true;
        }
    }

    void Queries::Counter::Restart(InterconnectContext &ctx) {
        if (recordedCopy) {
            End(ctx, false);
            ++(*usedQueryCount);
            Begin(ctx, false);
            End(ctx, true);
        }
    }

    void Queries::Counter::End(InterconnectContext &ctx, bool atEnd) {
        if (atEnd) {
            ctx.executor.InsertRpEndCommand([&pool = pool, queriesPtr = queries, usedQueryIndex = (*usedQueryCount) - 1](vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &, GPU &gpu) {
                if (!queriesPtr[usedQueryIndex].alreadyEnded && queriesPtr[usedQueryIndex].view)
                    commandBuffer.endQuery(*pool, usedQueryIndex);
            });
        } else {
            queries[(*usedQueryCount) - 1].alreadyEnded = true;
            ctx.executor.AddCommand([&pool = pool, queriesPtr = queries, usedQueryIndex = (*usedQueryCount) - 1](vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &, GPU &gpu) {
                if (queriesPtr[usedQueryIndex].view)
                    commandBuffer.endQuery(*pool, usedQueryIndex);
            });
        }
    }

    Queries::Queries(GPU &gpu) : counters{{{gpu.vkDevice, vk::QueryType::eOcclusion}}} {}

    void Queries::Query(InterconnectContext &ctx, soc::gm20b::IOVA address, CounterType type, std::optional<u64> timestamp) {
        view.Update(ctx, address, timestamp ? 16 : 4);
        usedQueryAddresses.emplace(u64{address});
        ctx.executor.AttachBuffer(*view);
        view->GetBuffer()->MarkGpuDirty(ctx.executor.usageTracker);

        auto &counter{counters[static_cast<u32>(type)]};
        counter.Report(ctx, *view, timestamp);
    }

    void Queries::ResetCounter(InterconnectContext &ctx, CounterType type) {
        auto &counter{counters[static_cast<u32>(type)]};
        counter.Restart(ctx);
    }

    void Queries::PurgeCaches(InterconnectContext &ctx) {
        view.PurgeCaches();
        for (u32 i{}; i < static_cast<u32>(CounterType::MaxValue); i++)
            counters[i].Reset(ctx);
    }

    bool Queries::QueryPresentAtAddress(soc::gm20b::IOVA address) {
        return usedQueryAddresses.contains(u64{address});
    }
}
