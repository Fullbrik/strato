// SPDX-License-Identifier: MPL-2.0
// Copyright © 2021 Skyline Team and Contributors (https://github.com/skyline-emu/)

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <range/v3/view.hpp>
#include <adrenotools/driver.h>
#include <common/settings.h>
#include <loader/loader.h>
#include <gpu.h>
#include <dlfcn.h>
#include "command_executor.h"
#include "command_nodes.h"
#include "gpu/texture/texture.h"
#include <nce.h>
#include "gpu/usage_tracker.h"

namespace skyline::gpu::interconnect {
    static void RecordFullBarrier(vk::raii::CommandBuffer &commandBuffer) {
        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands, {}, vk::MemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
                .dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
            }, {}, {}
        );
    }

    CommandRecordThread::CommandRecordThread(const DeviceState &state)
        : state{state},
          incoming{1U << *state.settings->executorSlotCountScale},
          outgoing{1U << *state.settings->executorSlotCountScale},
          thread{&CommandRecordThread::Run, this} {}

    CommandRecordThread::Slot::ScopedBegin::ScopedBegin(CommandRecordThread::Slot &slot) : slot{slot} {}

    CommandRecordThread::Slot::ScopedBegin::~ScopedBegin() {
        slot.Begin();
    }

    static vk::raii::CommandBuffer AllocateRaiiCommandBuffer(GPU &gpu, vk::raii::CommandPool &pool) {
        return {gpu.vkDevice, (*gpu.vkDevice).allocateCommandBuffers(
                    {
                        .commandPool = *pool,
                        .level = vk::CommandBufferLevel::ePrimary,
                        .commandBufferCount = 1
                    }, *gpu.vkDevice.getDispatcher()).front(),
                *pool};
    }

    CommandRecordThread::Slot::Slot(GPU &gpu)
        : commandPool{gpu.vkDevice,
                      vk::CommandPoolCreateInfo{
                          .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer | vk::CommandPoolCreateFlagBits::eTransient,
                          .queueFamilyIndex = gpu.vkQueueFamilyIndex
                      }
          },
          commandBuffer{AllocateRaiiCommandBuffer(gpu, commandPool)},
          fence{gpu.vkDevice, vk::FenceCreateInfo{ .flags = vk::FenceCreateFlagBits::eSignaled }},
          semaphore{gpu.vkDevice, vk::SemaphoreCreateInfo{}},
          cycle{std::make_shared<FenceCycle>(gpu.vkDevice, *fence, *semaphore, true)},
          nodes{allocator},
          pendingRenderPassEndNodes{allocator},
          pendingPostRenderPassNodes{allocator} {
        Begin();
    }

    CommandRecordThread::Slot::Slot(Slot &&other) noexcept
        : commandPool{std::move(other.commandPool)},
          commandBuffer{std::move(other.commandBuffer)},
          fence{std::move(other.fence)},
          semaphore{std::move(other.semaphore)},
          cycle{std::move(other.cycle)},
          allocator{std::move(other.allocator)},
          nodes{std::move(other.nodes)},
          pendingRenderPassEndNodes{std::move(other.pendingRenderPassEndNodes)},
          pendingPostRenderPassNodes{std::move(other.pendingPostRenderPassNodes)},
          ready{other.ready} {}

    std::shared_ptr<FenceCycle> CommandRecordThread::Slot::Reset(GPU &gpu) {
        auto startTime{util::GetTimeNs()};

        cycle->Wait();
        cycle = std::make_shared<FenceCycle>(*cycle);
        if (util::GetTimeNs() - startTime > GrowThresholdNs)
            didWait = true;

        // Command buffer doesn't need to be reset since that's done implicitly by begin
        return cycle;
    }

    void CommandRecordThread::Slot::WaitReady() {
        std::unique_lock lock{beginLock};
        beginCondition.wait(lock, [this] { return ready; });
        cycle->AttachObject(std::make_shared<ScopedBegin>(*this));
    }

    void CommandRecordThread::Slot::Begin() {
        std::unique_lock lock{beginLock};
        commandBuffer.begin(vk::CommandBufferBeginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        });
        ready = true;
        beginCondition.notify_all();
    }

    void CommandRecordThread::ProcessSlot(Slot *slot) {
        TRACE_EVENT_FMT("gpu", "ProcessSlot: {}, execution: {}", fmt::ptr(slot), u64{slot->executionTag});
        auto &gpu{*state.gpu};

        vk::RenderPass lRenderPass;
        u32 subpassIndex;

        using namespace node;
        for (NodeVariant &node : slot->nodes) {
            #define NODE(name) [&](name& node) { TRACE_EVENT_INSTANT("gpu", #name); node(slot->commandBuffer, slot->cycle, gpu); }
            std::visit(VariantVisitor{
                NODE(FunctionNode),

                [&](CheckpointNode &node) {
                    RecordFullBarrier(slot->commandBuffer);

                    TRACE_EVENT_INSTANT("gpu", "CheckpointNode", "id", node.id, [&](perfetto::EventContext ctx) {
                        ctx.event()->add_flow_ids(node.id);
                    });

                    std::array<vk::BufferCopy, 1> copy{vk::BufferCopy{
                        .size = node.binding.size,
                        .srcOffset = node.binding.offset,
                        .dstOffset = 0,
                    }};

                    slot->commandBuffer.copyBuffer(node.binding.buffer, gpu.debugTracingBuffer.vkBuffer, copy);

                    RecordFullBarrier(slot->commandBuffer);
                },

                NODE(RenderPassNode),
                NODE(RenderPassEndNode),
                NODE(SyncNode)
            }, node);
            #undef NODE
        }

        slot->commandBuffer.end();
        slot->ready = false;

        gpu.scheduler.SubmitCommandBuffer(slot->commandBuffer, slot->cycle);

        slot->nodes.clear();
        slot->allocator.Reset();
    }

    void CommandRecordThread::Run() {
        auto &gpu{*state.gpu};

        RENDERDOC_API_1_4_2 *renderDocApi{};
        if (void *mod{dlopen("libVkLayer_GLES_RenderDoc.so", RTLD_NOW | RTLD_NOLOAD)}) {
            auto *pfnGetApi{reinterpret_cast<pRENDERDOC_GetAPI>(dlsym(mod, "RENDERDOC_GetAPI"))};
            if (int ret{pfnGetApi(eRENDERDOC_API_Version_1_4_2, (void **)&renderDocApi)}; ret != 1)
                LOGW("Failed to intialise RenderDoc API: {}", ret);
        }

        outgoing.Push(&slots.emplace_back(gpu));

        if (int result{pthread_setname_np(pthread_self(), "Sky-CmdRecord")})
            LOGW("Failed to set the thread name: {}", strerror(result));
        AsyncLogger::UpdateTag();

        try {
            signal::SetHostSignalHandler({SIGINT, SIGILL, SIGTRAP, SIGBUS, SIGFPE, SIGSEGV}, signal::ExceptionalSignalHandler);

            incoming.Process([this, renderDocApi, &gpu](Slot *slot) {
                idle = false;
                VkInstance instance{*gpu.vkInstance};
                if (renderDocApi && slot->capture)
                    renderDocApi->StartFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance), nullptr);

                ProcessSlot(slot);

                if (renderDocApi && slot->capture)
                    renderDocApi->EndFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(instance), nullptr);
                slot->capture = false;

                // NOTE: what
                if (slot->didWait && (slots.size() + 1) < (1U << *state.settings->executorSlotCountScale)) {
                    outgoing.Push(&slots.emplace_back(gpu));
                    outgoing.Push(&slots.emplace_back(gpu));
                    slot->didWait = false;
                }

                outgoing.Push(slot);
                idle = true;
            }, [] {});
        } catch (const signal::SignalException &e) {
            LOGE("{}\nStack Trace:{}", e.what(), state.loader->GetStackTrace(e.frames));
            if (state.process)
                state.process->Kill(false);
            else
                std::rethrow_exception(std::current_exception());
        } catch (const std::exception &e) {
            LOGE("{}", e.what());
            if (state.process)
                state.process->Kill(false);
            else
                std::rethrow_exception(std::current_exception());
        }
    }

    bool CommandRecordThread::IsIdle() const {
        return idle;
    }

    CommandRecordThread::Slot *CommandRecordThread::AcquireSlot() {
        auto startTime{util::GetTimeNs()};
        auto slot{outgoing.Pop()};
        if (util::GetTimeNs() - startTime > GrowThresholdNs)
            slot->didWait = true;

        return slot;
    }

    void CommandRecordThread::ReleaseSlot(Slot *slot) {
        incoming.Push(slot);
    }

    void ExecutionWaiterThread::Run() {
        signal::SetHostSignalHandler({SIGSEGV}, nce::NCE::HostSignalHandler); // We may access NCE trapped memory

        // Enable turbo clocks to begin with if requested
        if (*state.settings->forceMaxGpuClocks)
            adrenotools_set_turbo(true);

        while (true) {
            std::pair<std::shared_ptr<FenceCycle>, std::function<void()>> item{};
            {
                std::unique_lock lock{mutex};
                if (pendingSignalQueue.empty()) {
                    idle = true;

                    // Don't force turbo clocks when the GPU is idle
                    if (*state.settings->forceMaxGpuClocks)
                        adrenotools_set_turbo(false);

                    condition.wait(lock, [this] { return !pendingSignalQueue.empty(); });

                    // Once we have work to do, force turbo clocks is enabled
                    if (*state.settings->forceMaxGpuClocks)
                        adrenotools_set_turbo(true);

                    idle = false;
                }
                item = std::move(pendingSignalQueue.front());
                pendingSignalQueue.pop();
            }
            {
                TRACE_EVENT("gpu", "GPU");
                if (item.first)
                    item.first->Wait();
            }

            if (item.second)
                item.second();
        }
    }

    ExecutionWaiterThread::ExecutionWaiterThread(const DeviceState &state) : state{state}, thread{&ExecutionWaiterThread::Run, this} {}

    bool ExecutionWaiterThread::IsIdle() const {
        return idle;
    }

    void ExecutionWaiterThread::Queue(std::shared_ptr<FenceCycle> cycle, std::function<void()> &&callback) {
        {
            std::unique_lock lock{mutex};
            pendingSignalQueue.push({std::move(cycle), std::move(callback)});
        }
        condition.notify_all();
    }

    void CheckpointPollerThread::Run() {
        u32 prevCheckpoint{};
        for (size_t iteration{}; true; iteration++) {
            u32 curCheckpoint{state.gpu->debugTracingBuffer.as<u32>()};

            if ((iteration % 1024) == 0)
                LOGI("Current Checkpoint: {}", curCheckpoint);

            while (prevCheckpoint != curCheckpoint) {
                // Make sure to report an event for every checkpoint inbetween the previous and current values, to ensure the perfetto trace is consistent
                prevCheckpoint++;
                TRACE_EVENT_INSTANT("gpu", "Checkpoint", "id", prevCheckpoint, [&](perfetto::EventContext ctx) {
                    ctx.event()->add_terminating_flow_ids(prevCheckpoint);
                });
            }

            prevCheckpoint = curCheckpoint;
            std::this_thread::sleep_for(std::chrono::microseconds(5));
        }
    }

    CheckpointPollerThread::CheckpointPollerThread(const DeviceState &state) : state{state}, thread{&CheckpointPollerThread::Run, this} {}

    CommandExecutor::CommandExecutor(const DeviceState &state)
        : state{state},
          gpu{*state.gpu},
          recordThread{state},
          waiterThread{state},
          checkpointPollerThread{EnableGpuCheckpoints ? std::optional<CheckpointPollerThread>{state} : std::optional<CheckpointPollerThread>{}},
          tag{AllocateTag()} {
        RotateRecordSlot();
    }

    CommandExecutor::~CommandExecutor() {
        cycle->Cancel();
    }

    void CommandExecutor::RotateRecordSlot() {
        if (slot) {
            slot->capture = captureNextExecution;
            recordThread.ReleaseSlot(slot);
        }

        captureNextExecution = false;
        slot = recordThread.AcquireSlot();
        cycle = slot->Reset(gpu);
        slot->executionTag = executionTag;
        allocator = &slot->allocator;
    }

    bool CommandExecutor::CreateRenderPassWithAttachments(vk::Rect2D renderArea, span<std::pair<HostTextureView *, TextureSyncRequestArgs>> sampledImages, span<HostTextureView *> colorAttachments, TextureSyncRequestArgs colorAttachmentSync, std::pair<HostTextureView *, TextureSyncRequestArgs> depthStencilAttachment, vk::PipelineStageFlags srcStageMask, vk::PipelineStageFlags dstStageMask) {
        span<HostTextureView *> otherDSASpan{depthStencilAttachment.first ?depthStencilAttachment.first : span<HostTextureView *>{}};
        span<std::pair<HostTextureView *, TextureSyncRequestArgs>> depthStencilAttachmentSpan{depthStencilAttachment.first ? span<std::pair<HostTextureView *, TextureSyncRequestArgs>>(depthStencilAttachment) : span<std::pair<HostTextureView *, TextureSyncRequestArgs>>()};
        auto outputAttachmentViews{ranges::views::concat(colorAttachments, otherDSASpan)};

        bool newRenderPass{renderPass == nullptr || renderPass->renderArea != renderArea ||
            !ranges::all_of(outputAttachmentViews, [this](auto view) { return !view || view->hostTexture->ValidateRenderPassUsage(renderPassIndex, texture::RenderPassUsage::RenderTarget); }) ||
            !ranges::all_of(sampledImages, [this](auto view) { return !view.first || view.first->hostTexture->ValidateRenderPassUsage(renderPassIndex, texture::RenderPassUsage::Descriptor); })};

        if (!newRenderPass)
            // Try to bind the new attachments to the current render pass, we can avoid creating a new render pass if the attachments are compatible
            newRenderPass = !renderPass->BindAttachments(colorAttachments, depthStencilAttachment.first);

        if (newRenderPass) {
            // We need to create a render pass if one doesn't already exist or the current one isn't compatible
            FinishRenderPass();

            if (!transferPass.active)
                CreateTransferPass();

            for (auto &view : sampledImages)
                if (view.first)
                    view.first->RequestSync(*this, view.second);

            for (auto &view : colorAttachments) {
                if (view) {
                    view->RequestSync(*this, colorAttachmentSync);
                    view->hostTexture->usedInRP = true;
                }
            }

            if (depthStencilAttachment.first) {
                depthStencilAttachment.first->RequestSync(*this, depthStencilAttachment.second);
                depthStencilAttachment.first->hostTexture->usedInRP = true;
            }

            renderPass = &std::get<node::RenderPassNode>(slot->nodes.emplace_back(std::in_place_type_t<node::RenderPassNode>(), renderArea, colorAttachments, depthStencilAttachment.first));
            subpassCount = 1;
            renderPassIt = std::prev(slot->nodes.end());
        } else {
            transferPass.ignoreCompat = true;

            node::SyncNode *rpSyncNode{&std::get<node::SyncNode>(slot->nodes.emplace_back(std::in_place_type_t<node::SyncNode>()))};
            //rpSyncNode->deps = vk::DependencyFlagBits::eByRegion;

            for (auto &view : sampledImages)
                if (view.first)
                    view.first->RequestSync(*this, view.second);

            for (auto &view : colorAttachments)
                if (view)
                    view->RequestRPSync(*this, colorAttachmentSync, rpSyncNode);

            if (depthStencilAttachment.first)
                depthStencilAttachment.first->RequestRPSync(*this, depthStencilAttachment.second, rpSyncNode);

            transferPass.ignoreCompat = false;
        }

        renderPass->UpdateDependency(srcStageMask, dstStageMask);

        for (auto view : outputAttachmentViews)
            if (view)
                view->hostTexture->UpdateRenderPassUsage(renderPassIndex, texture::RenderPassUsage::RenderTarget);

        for (auto view : sampledImages)
            if (view.first)
                view.first->hostTexture->UpdateRenderPassUsage(renderPassIndex, texture::RenderPassUsage::Descriptor);

        return newRenderPass;
    }

    void CommandExecutor::CreateTransferPass() {
        FinishRenderPass();

        transferPass.active = true;

        transferPass.preStagingCopyNode = &std::get<node::SyncNode>(slot->nodes.emplace_back(std::in_place_type_t<node::SyncNode>()));
        transferPass.stagingCopyNode = &std::get<node::SyncNode>(slot->nodes.emplace_back(std::in_place_type_t<node::SyncNode>()));
        transferPass.toStagingIt = std::prev(slot->nodes.end());
        transferPass.postStagingCopyNode = &std::get<node::SyncNode>(slot->nodes.emplace_back(std::in_place_type_t<node::SyncNode>()));
        transferPass.fromStagingIt = std::prev(slot->nodes.end());

        for (const auto &attachedTexture : ranges::views::concat(attachedTextures, preserveAttachedTextures))
            for (auto &host : attachedTexture.texture->hosts)
                host.usedInTP = false;
    }

    void CommandExecutor::FinishRenderPass() {
        if (renderPass) {
            slot->nodes.splice(slot->nodes.end(), slot->pendingRenderPassEndNodes);
            slot->nodes.emplace_back(std::in_place_type_t<node::RenderPassEndNode>());
            slot->nodes.splice(slot->nodes.end(), slot->pendingPostRenderPassNodes);
            ++renderPassIndex;

            span<std::optional<node::RenderPassNode::Attachment>> depthStencilAttachmentSpan{renderPass->depthStencilAttachment};
            for (auto &attachment : ranges::views::concat(renderPass->colorAttachments, depthStencilAttachmentSpan))
                if (attachment && attachment->view)
                    attachment->view->hostTexture->usedInRP = false;

            renderPass = nullptr;
            subpassCount = 0;
        }
    }

    CommandExecutor::LockedTexture::LockedTexture(std::shared_ptr<Texture> texture) : texture{std::move(texture)} {}

    constexpr CommandExecutor::LockedTexture::LockedTexture(CommandExecutor::LockedTexture &&other) noexcept : texture{std::exchange(other.texture, nullptr)} {}

    bool CommandExecutor::AttachTextureView(HostTextureView *view) {
        bool didLock{view->LockWithTag(executionTag)};
        if (didLock) {
            // TODO: fixup remaining bugs with this and add better heuristics to avoid pauses
            // if (view->texture->FrequentlyLocked()) {
                attachedTextures.emplace_back(view->texture->shared_from_this());
                attachedTextures.back().texture->AttachCycle(cycle);
            // } else {
            //    preserveAttachedTextures.emplace_back(view->texture);
            // }
        }

        return didLock;
    }

    bool CommandExecutor::AttachTexture(std::shared_ptr<Texture> texture) {
        bool didLock{texture->LockWithTag(executionTag)};
        if (didLock) {
            // TODO: fixup remaining bugs with this and add better heuristics to avoid pauses
            // if (view->texture->FrequentlyLocked()) {
            attachedTextures.emplace_back(std::move(texture));
            attachedTextures.back().texture->AttachCycle(cycle);
            // } else {
            //    preserveAttachedTextures.emplace_back(view->texture);
            // }
        }

        return didLock;
    }

    CommandExecutor::LockedBuffer::LockedBuffer(std::shared_ptr<Buffer> buffer) : buffer{std::move(buffer)} {}

    constexpr CommandExecutor::LockedBuffer::LockedBuffer(CommandExecutor::LockedBuffer &&other) noexcept : buffer{std::exchange(other.buffer, nullptr)} {}

    constexpr Buffer *CommandExecutor::LockedBuffer::operator->() const {
        return buffer.get();
    }

    CommandExecutor::LockedBuffer::~LockedBuffer() {
        if (buffer)
            buffer->unlock();
    }

    void CommandExecutor::AttachBufferBase(std::shared_ptr<Buffer> buffer) {
        // TODO: fixup remaining bugs with this and add better heuristics to avoid pauses
        // if (buffer->FrequentlyLocked())
        attachedBuffers.emplace_back(std::move(buffer));
        // else
        //    preserveAttachedBuffers.emplace_back(std::move(buffer));
    }

    bool CommandExecutor::AttachBuffer(BufferView &view) {
        bool didLock{view.LockWithTag(tag)};
        if (didLock)
            AttachBufferBase(view.GetBuffer()->shared_from_this());

        return didLock;
    }

    void CommandExecutor::AttachLockedBufferView(BufferView &view, ContextLock<BufferView> &&lock) {
        if (lock.OwnsLock()) {
            // Transfer ownership to executor so that the resource will stay locked for the period it is used on the GPU
            AttachBufferBase(view.GetBuffer()->shared_from_this());
            lock.Release(); // The executor will handle unlocking the lock so it doesn't need to be handled here
        }
    }

    void CommandExecutor::AttachLockedBuffer(std::shared_ptr<Buffer> buffer, ContextLock<Buffer> &&lock) {
        if (lock.OwnsLock()) {
            AttachBufferBase(std::move(buffer));
            lock.Release(); // See AttachLockedBufferView(...)
        }
    }

    void CommandExecutor::AttachDependency(const std::shared_ptr<void> &dependency) {
        cycle->AttachObject(dependency);
    }

    void CommandExecutor::AddSubpass(ExecutorCommand function, vk::Rect2D renderArea, span<std::pair<HostTextureView *, TextureSyncRequestArgs>> sampledImages, span<HostTextureView *> colorAttachments, TextureSyncRequestArgs colorAttachmentSync, std::pair<HostTextureView *, TextureSyncRequestArgs> depthStencilAttachment, vk::PipelineStageFlags srcStageMask, vk::PipelineStageFlags dstStageMask) {
        bool newRenderpass{CreateRenderPassWithAttachments(renderArea, sampledImages, colorAttachments, colorAttachmentSync, depthStencilAttachment, srcStageMask, dstStageMask)};
        slot->nodes.emplace_back(std::in_place_type_t<node::FunctionNode>(), std::forward<decltype(function)>(function));

        if (slot->nodes.size() >= *state.settings->executorFlushThreshold && newRenderpass)
            Submit();
    }

    void CommandExecutor::AddOutsideRpCommand(ExecutorCommand function) {
        FinishRenderPass();

        slot->nodes.emplace_back(std::in_place_type_t<node::FunctionNode>(), std::forward<decltype(function)>(function));
    }

    void CommandExecutor::AddCommand(ExecutorCommand function) {
        slot->nodes.emplace_back(std::in_place_type_t<node::FunctionNode>(), std::forward<decltype(function)>(function));
    }

    void CommandExecutor::InsertPreExecuteCommand(ExecutorCommand function) {
        slot->nodes.emplace(slot->nodes.begin(), std::in_place_type_t<node::FunctionNode>(), std::forward<decltype(function)>(function));
    }

    void CommandExecutor::InsertPreRpCommand(ExecutorCommand function) {
        slot->nodes.emplace(renderPass ? renderPassIt : slot->nodes.end(), std::in_place_type_t<node::FunctionNode>(), std::forward<decltype(function)>(function));
    }

    void CommandExecutor::InsertRpBeginCommand(ExecutorCommand function) {
        slot->nodes.emplace(renderPass ? std::next(renderPassIt) : slot->nodes.end(), std::in_place_type_t<node::FunctionNode>(), std::forward<decltype(function)>(function));
    }

    void CommandExecutor::InsertPostRpCommand(ExecutorCommand function) {
        slot->pendingPostRenderPassNodes.emplace_back(std::in_place_type_t<node::FunctionNode>(), std::forward<decltype(function)>(function));
    }

    void CommandExecutor::InsertRpEndCommand(ExecutorCommand function) {
        slot->pendingRenderPassEndNodes.emplace_back(std::in_place_type_t<node::FunctionNode>(), std::forward<decltype(function)>(function));
    }

    void CommandExecutor::AddRPTextureBarrier(HostTexture &toWait, const TextureSyncRequestArgs &args, node::SyncNode *toWaitWith) {
        if (!args.isWritten && (toWait.trackingInfo.waitedStages & args.usedStage))
            return;

        vk::PipelineStageFlags srcStages{toWait.trackingInfo.lastUsedStage};
        if (args.isWritten)
            srcStages |= toWait.trackingInfo.waitedStages;

        toWaitWith->srcStages |= srcStages;

        toWaitWith->dstStages |= args.usedStage;

        toWaitWith->imageBarriers.emplace_back(vk::ImageMemoryBarrier{
            .image = toWait.GetImage(),
            .srcAccessMask = toWait.trackingInfo.lastUsedAccessFlag,
            .dstAccessMask = args.usedFlags,
            .oldLayout = toWait.GetLayout(),
            .newLayout = toWait.GetLayout(),
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .subresourceRange = {
                .aspectMask = toWait.format->vkAspect,
                .levelCount = toWait.texture.guest.levelCount,
                .layerCount = toWait.texture.guest.layerCount
            }
        });

        toWait.usedInTP = true;

        if (!toWaitWith->active)
            toWaitWith->active = true;

        renderPass->UpdateSelfDependency(srcStages, args.usedStage, toWait.trackingInfo.lastUsedAccessFlag, args.usedFlags);
    }

    void CommandExecutor::AddTextureBarrier(HostTexture &toWait, const TextureSyncRequestArgs &args) {
        if (!transferPass.ignoreCompat && (!transferPass.active || toWait.usedInTP))
            CreateTransferPass();

        transferPass.preStagingCopyNode->srcStages |= toWait.trackingInfo.lastUsedStage;
        if (args.isWritten)
            transferPass.preStagingCopyNode->srcStages |= toWait.trackingInfo.waitedStages;

        transferPass.preStagingCopyNode->dstStages |= args.usedStage;

        transferPass.preStagingCopyNode->imageBarriers.emplace_back(vk::ImageMemoryBarrier{
            .image = toWait.GetImage(),
            .srcAccessMask = toWait.trackingInfo.lastUsedAccessFlag,
            .dstAccessMask = args.usedFlags,
            .oldLayout = toWait.GetLayout(),
            .newLayout = toWait.GetLayout(),
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .subresourceRange = {
                .aspectMask = toWait.format->vkAspect,
                .levelCount = toWait.texture.guest.levelCount,
                .layerCount = toWait.texture.guest.layerCount
            }
        });

        if (!transferPass.preStagingCopyNode->active)
            transferPass.preStagingCopyNode->active = true;

        toWait.usedInTP = true;
    }

    void CommandExecutor::AddTextureTransferCommand(HostTexture &toWait, const TextureSyncRequestArgs &args, ExecutorCommand function) {
        if (!transferPass.ignoreCompat && (!transferPass.active || toWait.usedInTP))
            CreateTransferPass();

        transferPass.preStagingCopyNode->srcStages |= toWait.trackingInfo.lastUsedStage;
        if (args.isWritten)
            transferPass.preStagingCopyNode->srcStages |= toWait.trackingInfo.waitedStages;

        transferPass.preStagingCopyNode->dstStages |= vk::PipelineStageFlagBits::eTransfer;

        transferPass.preStagingCopyNode->imageBarriers.emplace_back(vk::ImageMemoryBarrier{
            .image = toWait.GetImage(),
            .srcAccessMask = toWait.trackingInfo.lastUsedAccessFlag,
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = toWait.GetLayout(),
            .newLayout = toWait.GetLayout(),
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .subresourceRange = {
                .aspectMask = toWait.format->vkAspect,
                .levelCount = toWait.texture.guest.levelCount,
                .layerCount = toWait.texture.guest.layerCount
            }
        });

        if (!transferPass.preStagingCopyNode->active)
            transferPass.preStagingCopyNode->active = true;

        transferPass.postStagingCopyNode->srcStages |= vk::PipelineStageFlagBits::eTransfer;

        transferPass.postStagingCopyNode->dstStages |= args.usedStage;

        transferPass.postStagingCopyNode->imageBarriers.emplace_back(vk::ImageMemoryBarrier{
            .image = toWait.GetImage(),
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = args.usedFlags,
            .oldLayout = toWait.GetLayout(),
            .newLayout = toWait.GetLayout(),
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .subresourceRange = {
                .aspectMask = toWait.format->vkAspect,
                .levelCount = toWait.texture.guest.levelCount,
                .layerCount = toWait.texture.guest.layerCount
            }
        });

        if (!transferPass.postStagingCopyNode->active)
            transferPass.postStagingCopyNode->active = true;

        slot->nodes.emplace(transferPass.fromStagingIt, std::in_place_type_t<node::FunctionNode>(), std::forward<decltype(function)>(function));

        toWait.usedInTP = true;
    }

    void CommandExecutor::AddStagedTextureTransferCommand(HostTexture &toWait, const TextureSyncRequestArgs &args, ExecutorCommand preStagingFunction, ExecutorCommand postStagingFunction) {
        if (!transferPass.ignoreCompat && (!transferPass.active || toWait.usedInTP))
            CreateTransferPass();

        transferPass.preStagingCopyNode->srcStages |= toWait.trackingInfo.lastUsedStage | vk::PipelineStageFlagBits::eTransfer;
        if (args.isWritten)
            transferPass.preStagingCopyNode->srcStages |= toWait.trackingInfo.waitedStages;
        transferPass.preStagingCopyNode->dstStages |= vk::PipelineStageFlagBits::eTransfer;

        transferPass.preStagingCopyNode->imageBarriers.emplace_back(vk::ImageMemoryBarrier{
            .image = toWait.GetImage(),
            .srcAccessMask = toWait.trackingInfo.lastUsedAccessFlag,
            .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            .oldLayout = toWait.GetLayout(),
            .newLayout = toWait.GetLayout(),
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .subresourceRange = {
                .aspectMask = toWait.format->vkAspect,
                .levelCount = toWait.texture.guest.levelCount,
                .layerCount = toWait.texture.guest.layerCount
            }
        });

        if (!transferPass.preStagingCopyNode->active)
            transferPass.preStagingCopyNode->active = true;

        transferPass.stagingCopyNode->srcStages |= vk::PipelineStageFlagBits::eTransfer;
        transferPass.stagingCopyNode->dstStages |= vk::PipelineStageFlagBits::eTransfer;

        transferPass.stagingCopyNode->bufferBarriers.emplace_back(vk::BufferMemoryBarrier{
            .buffer = toWait.texture.syncStagingBuffer->vkBuffer,
            .size = toWait.texture.syncStagingBuffer->size(),
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = vk::AccessFlagBits::eTransferRead,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED
        });

        if (!transferPass.stagingCopyNode->active)
            transferPass.stagingCopyNode->active = true;

        transferPass.postStagingCopyNode->srcStages |= vk::PipelineStageFlagBits::eTransfer;
        transferPass.postStagingCopyNode->dstStages |= args.usedStage;

        transferPass.postStagingCopyNode->imageBarriers.emplace_back(vk::ImageMemoryBarrier{
            .image = toWait.GetImage(),
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = args.usedFlags,
            .oldLayout = toWait.GetLayout(),
            .newLayout = toWait.GetLayout(),
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .subresourceRange = {
                .aspectMask = toWait.format->vkAspect,
                .levelCount = toWait.texture.guest.levelCount,
                .layerCount = toWait.texture.guest.layerCount
            }
        });

        if (!transferPass.postStagingCopyNode->active)
            transferPass.postStagingCopyNode->active = true;

        slot->nodes.emplace(transferPass.toStagingIt, std::in_place_type_t<node::FunctionNode>(), std::forward<decltype(preStagingFunction)>(preStagingFunction));
        slot->nodes.emplace(transferPass.fromStagingIt, std::in_place_type_t<node::FunctionNode>(), std::forward<decltype(postStagingFunction)>(postStagingFunction));

        toWait.usedInTP = true;
    }

    void CommandExecutor::AddFullBarrier() {
        AddOutsideRpCommand([](vk::raii::CommandBuffer &commandBuffer, const std::shared_ptr<FenceCycle> &, GPU &) {
            RecordFullBarrier(commandBuffer);
        });
    }

    void CommandExecutor::AddClearColorSubpass(vk::Rect2D renderArea, HostTextureView *attachment, const vk::ClearColorValue &value) {
        if (!renderPass || !renderPass->ClearColorAttachment(attachment, value, gpu)) {
            CreateRenderPassWithAttachments(renderArea, {}, attachment, {}, {nullptr, {}});
            renderPass->ClearColorAttachment(attachment, value, gpu);
        }
    }

    void CommandExecutor::AddClearDepthStencilSubpass(vk::Rect2D renderArea, HostTextureView *attachment, const vk::ClearDepthStencilValue &value) {
        if (!renderPass || !renderPass->ClearDepthStencilAttachment(attachment, value, gpu)) {
            CreateRenderPassWithAttachments(renderArea, {}, {}, {}, {attachment, {}});
            renderPass->ClearDepthStencilAttachment(attachment, value, gpu);
        }
    }

    void CommandExecutor::AddFlushCallback(std::function<void()> &&callback) {
        flushCallbacks.emplace_back(std::forward<decltype(callback)>(callback));
    }

    void CommandExecutor::AddPipelineChangeCallback(std::function<void()> &&callback) {
        pipelineChangeCallbacks.emplace_back(std::forward<decltype(callback)>(callback));
    }

    void CommandExecutor::NotifyPipelineChange() {
        for (auto &callback : pipelineChangeCallbacks)
            callback();
    }

    std::optional<u32> CommandExecutor::GetRenderPassIndex() {
        return renderPassIndex;
    }

    u32 CommandExecutor::AddCheckpointImpl(std::string_view annotation) {
        FinishRenderPass();

        slot->nodes.emplace_back(node::CheckpointNode{gpu.megaBufferAllocator.Push(cycle, span<u32>(&nextCheckpointId, 1).cast<u8>()), nextCheckpointId});

        TRACE_EVENT_INSTANT("gpu", "Mark Checkpoint", "id", nextCheckpointId, "annotation", [&annotation](perfetto::TracedValue context) {
            std::move(context).WriteString(annotation.data(), annotation.size());
        }, [&](perfetto::EventContext ctx) {
            ctx.event()->add_flow_ids(nextCheckpointId);
        });

        return nextCheckpointId++;
    }

    void CommandExecutor::SubmitInternal() {
        if (renderPass) {
            FinishRenderPass();
        } else {
            slot->nodes.splice(slot->nodes.end(), slot->pendingRenderPassEndNodes);
            slot->nodes.splice(slot->nodes.end(), slot->pendingPostRenderPassNodes);
        }
        transferPass.toStagingIt = {};
        transferPass.fromStagingIt = {};
        transferPass.active = false;

        slot->WaitReady();

        // We need this barrier here to ensure that resources are in the state we expect them to be in, we shouldn't overwrite resources while prior commands might still be using them or read from them while they might be modified by prior commands
        RecordFullBarrier(slot->commandBuffer);

        for (const auto &attachedTexture : ranges::views::concat(attachedTextures, preserveAttachedTextures)) {
            attachedTexture.texture->OnExecStart(*this);
            slot->nodes.emplace_back(std::in_place_type_t<node::FunctionNode>(), [attachedTexture = attachedTexture.texture](vk::raii::CommandBuffer &, const std::shared_ptr<FenceCycle> &, GPU &){attachedTexture->OnExecEnd();});

            attachedTexture.texture->AttachCycle(cycle);
        }

        for (const auto &attachedBuffer : ranges::views::concat(attachedBuffers, preserveAttachedBuffers)) {
            if (attachedBuffer->RequiresCycleAttach()) {
                attachedBuffer->SynchronizeHost(); // Synchronize attached buffers from the CPU without using a staging buffer
                cycle->AttachObject(attachedBuffer.buffer);
                attachedBuffer->UpdateCycle(cycle);
                attachedBuffer->AllowAllBackingWrites();
            }
        }

        RotateRecordSlot();
    }

    void CommandExecutor::ResetInternal() {
        attachedTextures.clear();
        attachedBuffers.clear();
        allocator->Reset();
        renderPassIndex = 0;
        usageTracker.sequencedIntervals.Clear();

        // Periodically clear preserve attachments just in case there are new waiters which would otherwise end up waiting forever
        if ((submissionNumber % (2U << *state.settings->executorSlotCountScale)) == 0) {
            preserveAttachedBuffers.clear();
            preserveAttachedTextures.clear();
        }
    }

    void CommandExecutor::Submit(std::function<void()> &&callback, bool wait, bool destroyTextures) {
        for (const auto &flushCallback : flushCallbacks)
            flushCallback();

        executionTag = AllocateTag();

        // Ensure all pushed callbacks wait for the submission to have finished GPU execution
        if (!slot->nodes.empty())
            waiterThread.Queue(cycle, {});

        if (*state.settings->useDirectMemoryImport) {
            // When DMI is in use, callbacks and deferred actions should be executed in sequence with the host GPU
            for (auto &actionCb : pendingDeferredActions)
                waiterThread.Queue(nullptr, std::move(actionCb));

            pendingDeferredActions.clear();

            if (callback)
                waiterThread.Queue(nullptr, std::move(callback));
        }

        if (!slot->nodes.empty()) {
            TRACE_EVENT("gpu", "CommandExecutor::Submit");
            SubmitInternal();
            submissionNumber++;
        }

        if (!*state.settings->useDirectMemoryImport) {
            // When DMI is not in use, execute callbacks immediately after submission
            for (auto &actionCb : pendingDeferredActions)
                actionCb();

            pendingDeferredActions.clear();

            if (callback)
                callback();
        }

        ResetInternal();

        if (destroyTextures)
            gpu.texture.DestroyStaleTextures();

        if (wait) {
            usageTracker.dirtyIntervals.Clear();

            std::condition_variable cv;
            std::mutex mutex;
            bool gpuDone{};

            waiterThread.Queue(nullptr, [&cv, &mutex, &gpuDone] {
                std::scoped_lock lock{mutex};
                gpuDone = true;
                cv.notify_one();
            });

            std::unique_lock lock{mutex};
            cv.wait(lock, [&gpuDone] { return gpuDone; });
        }
    }

    void CommandExecutor::AddDeferredAction(std::function<void()> &&callback) {
        pendingDeferredActions.emplace_back(std::move(callback));
    }

    void CommandExecutor::LockPreserve() {
        if (!preserveLocked) {
            preserveLocked = true;

            for (auto &buffer : preserveAttachedBuffers)
                buffer->LockWithTag(tag);

            //gpu.texture.mutex.lock(); // We need to lock the texture mutex to ensure that no other thread is currently modifying the texture state
            for (auto &texture : preserveAttachedTextures)
                texture.texture->LockWithTag(tag);
        }
    }

    void CommandExecutor::UnlockPreserve() {
        if (preserveLocked) {
            for (auto &buffer : preserveAttachedBuffers)
                buffer->unlock();

            for (auto &texture : preserveAttachedTextures)
                texture.texture->unlock();

            preserveLocked = false;
        }
    }
}
