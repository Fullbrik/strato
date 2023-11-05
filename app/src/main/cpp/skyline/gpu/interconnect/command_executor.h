// SPDX-License-Identifier: MPL-2.0
// Copyright © 2021 Skyline Team and Contributors (https://github.com/skyline-emu/)

#pragma once

#include <boost/container/stable_vector.hpp>
#include <renderdoc_app.h>
#include <common/linear_allocator.h>
#include <gpu/megabuffer.h>
#include "command_nodes.h"
#include "common/spin_lock.h"

namespace skyline::gpu {
    struct UsageTracker;
    struct TextureSyncRequestArgs;
    class HostTexture;
}

namespace skyline::gpu::interconnect {
    constexpr bool EnableGpuCheckpoints{false}; //!< Whether to enable GPU debugging checkpoints (WILL DECREASE PERF SIGNIFICANTLY)

    using ExecutorCommand = std::function<void(vk::raii::CommandBuffer &, const std::shared_ptr<FenceCycle> &, GPU &)> &&;

    /*
     * @brief Thread responsible for recording Vulkan commands from the execution nodes and submitting them
     */
    class CommandRecordThread {
      public:
        /**
         * @brief Single execution slot, buffered back and forth between the GPFIFO thread and the record thread
         */
        struct Slot {
            /**
             * @brief Helper to begin the slot command buffer on the cycle waiter thread
             */
            struct ScopedBegin {
                Slot &slot;

                ScopedBegin(Slot &slot);

                ~ScopedBegin();
            };

            vk::raii::CommandPool commandPool; //!< Use one command pool per slot since command buffers from different slots may be recorded into on multiple threads at the same time
            vk::raii::CommandBuffer commandBuffer;
            vk::raii::Fence fence;
            vk::raii::Semaphore semaphore;
            std::shared_ptr<FenceCycle> cycle;
            LinearAllocatorState<> allocator;
            std::list<node::NodeVariant, LinearAllocator<node::NodeVariant>> nodes;
            std::list<node::NodeVariant, LinearAllocator<node::NodeVariant>> pendingRenderPassEndNodes;
            std::list<node::NodeVariant, LinearAllocator<node::NodeVariant>> pendingPostRenderPassNodes;
            std::mutex beginLock;
            std::condition_variable beginCondition;
            ContextTag executionTag;
            bool ready{}; //!< If this slot's command buffer has had 'beginCommandBuffer' called and is ready to have commands recorded into it
            bool capture{}; //!< If this slot's Vulkan commands should be captured using the renderdoc API
            bool didWait{}; //!< If a wait of time longer than GrowThresholdNs occured when this slot was acquired

            Slot(GPU &gpu);

            Slot(Slot &&other) noexcept;

            /**
             * @brief Waits on the fence and resets the command buffer
             * @note A new fence cycle for the reset command buffer
             */
            std::shared_ptr<FenceCycle> Reset(GPU &gpu);

            /**
             * @brief Waits for the command buffer to be began so it can be recorded into
             */
            void WaitReady();

            void Begin();
        };

      private:
        static constexpr size_t GrowThresholdNs{constant::NsInMillisecond / 50}; //!< The wait time threshold at which the slot count will be increased
        const DeviceState &state;
        CircularQueue<Slot *> incoming; //!< Slots pending recording
        CircularQueue<Slot *> outgoing; //!< Slots that have been submitted, may still be active on the GPU
        std::list<Slot> slots;
        std::atomic<bool> idle;

        std::thread thread;

        void ProcessSlot(Slot *slot);

        void Run();

      public:
        CommandRecordThread(const DeviceState &state);

        bool IsIdle() const;

        /**
         * @return A free slot, `Reset` needs to be called before accessing it
         */
        Slot *AcquireSlot();

        /**
         * @brief Submit a slot to be recorded
         */
        void ReleaseSlot(Slot *slot);
    };

    /**
     * @brief Thread responsible for notifying the guest of the completion of GPU operations
     */
    class ExecutionWaiterThread {
      private:
        const DeviceState &state;
        std::thread thread;
        SpinLock mutex;
        std::condition_variable_any condition;
        std::queue<std::pair<std::shared_ptr<FenceCycle>, std::function<void()>>> pendingSignalQueue; //!< Queue of callbacks to be executed when their coressponding fence is signalled
        std::atomic<bool> idle{};

        void Run();

      public:
        ExecutionWaiterThread(const DeviceState &state);

        bool IsIdle() const;

        /**
         * @brief Queues `callback` to be executed when `cycle` is signalled, null values are valid for either, will null cycle representing an immediate callback (dep on previously queued cycles) and null callback representing a wait with no callback
         */
        void Queue(std::shared_ptr<FenceCycle> cycle, std::function<void()> &&callback);
    };

    /**
     * @brief Polls the debug buffer for checkpoint updates and reports them to perfetto
     */
    class [[maybe_unused]] CheckpointPollerThread {
      private:
        const DeviceState &state;
        std::thread thread;

        void Run();

      public:
        CheckpointPollerThread(const DeviceState &state);
    };

    /**
     * @brief Assembles a Vulkan command stream with various nodes and manages execution of the produced graph
     * @note This class is **NOT** thread-safe and should **ONLY** be utilized by a single thread
     */
    class CommandExecutor {
      private:
        const DeviceState &state;
        GPU &gpu;
        CommandRecordThread recordThread;
        CommandRecordThread::Slot *slot{};
        ExecutionWaiterThread waiterThread;
        [[maybe_unused]] std::optional<CheckpointPollerThread> checkpointPollerThread;
        node::RenderPassNode *renderPass{};

        std::list<node::NodeVariant, LinearAllocator<node::NodeVariant>>::iterator renderPassIt;
        size_t subpassCount{}; //!< The number of subpasses in the current render pass (TODO: remove)
        u32 renderPassIndex{};
        bool preserveLocked{};

        /**
         * @brief A wrapper of a Texture object that has been locked beforehand and must be unlocked afterwards
         */
        struct LockedTexture {
            std::shared_ptr<Texture> texture;

            explicit LockedTexture(std::shared_ptr<Texture> texture);

            LockedTexture(const LockedTexture &) = delete;

            constexpr LockedTexture(LockedTexture &&other) noexcept;

            ~LockedTexture();
        };

        std::vector<LockedTexture> preserveAttachedTextures;
        std::vector<LockedTexture> attachedTextures; //!< All textures that are attached to the current execution

        /**
         * @brief A wrapper of a Buffer object that has been locked beforehand and must be unlocked afterwards
         */
        struct LockedBuffer {
            std::shared_ptr<Buffer> buffer;

            LockedBuffer(std::shared_ptr<Buffer> buffer);

            LockedBuffer(const LockedBuffer &) = delete;

            constexpr LockedBuffer(LockedBuffer &&other) noexcept;

            constexpr Buffer *operator->() const;

            ~LockedBuffer();
        };

        std::vector<LockedBuffer> preserveAttachedBuffers;
        std::vector<LockedBuffer> attachedBuffers; //!< All textures that are attached to the current execution

        std::vector<std::function<void()>> flushCallbacks; //!< Set of persistent callbacks that will be called at the start of Execute in order to flush data required for recording
        std::vector<std::function<void()>> pipelineChangeCallbacks; //!< Set of persistent callbacks that will be called after any non-Maxwell 3D engine changes the active pipeline

        std::vector<std::function<void()>> pendingDeferredActions;

        u32 nextCheckpointId{}; //!< The ID of the next debug checkpoint to be allocated

        void RotateRecordSlot();

        /**
         * @brief Create a new render pass with the specified attachments or reuses the current render pass if compatible
         * @return If a new render pass was created or not
         */
        bool CreateRenderPassWithAttachments(vk::Rect2D renderArea, span<std::pair<HostTextureView *, TextureSyncRequestArgs>> sampledImages, span<HostTextureView *> colorAttachments, HostTextureView *depthStencilAttachment, vk::PipelineStageFlags srcStageMask = {}, vk::PipelineStageFlags dstStageMask = {});

        /**
         * @brief Execute all the nodes and submit the resulting command buffer to the GPU
         * @note It is the responsibility of the caller to handle resetting of command buffers, fence cycle and megabuffers
         */
        void SubmitInternal();

        /**
         * @brief Resets all the internal state, this must be called before starting a new submission as it clears everything from a past submission
         */
        void ResetInternal();

        void AttachBufferBase(std::shared_ptr<Buffer> buffer);

        /**
         * @brief Non-gated implementation of `AddCheckpoint`
         */
        u32 AddCheckpointImpl(std::string_view annotation);

      public:
        std::shared_ptr<FenceCycle> cycle; //!< The fence cycle that this command executor uses to wait for the GPU to finish executing commands
        LinearAllocatorState<> *allocator{};
        ContextTag tag; //!< The tag associated with this command executor, any tagged resource locking must utilize this tag
        size_t submissionNumber{};
        ContextTag executionTag{};
        bool captureNextExecution{};
        UsageTracker usageTracker{};
        struct TransferPass {
            node::SyncNode *preStagingCopyNode{}; //!< The first barrier of a transfer pass, if a transfer is done the barrier for the texture for previous reads/writes will be placed here, normal texture barriers will also be placed here
            node::SyncNode *stagingCopyNode{}; //!< The second barrier will make the results of copies into the staging buffer avalable to be copied out, unused otherwise
            std::list<node::NodeVariant, LinearAllocator<node::NodeVariant>>::iterator toStagingIt; //!< An iterator to the above node
            node::SyncNode *postStagingCopyNode{}; //!< The last barrier makes the results of transfers into the texture available for later commands
            std::list<node::NodeVariant, LinearAllocator<node::NodeVariant>>::iterator fromStagingIt; //!< An iterator to the above node
        } transferPass{};

        CommandExecutor(const DeviceState &state);

        ~CommandExecutor();

        /**
         * @brief Attach the lifetime of the texture to the command buffer
         * @return If this is the first usage of the backing of this resource within this execution
         * @note The supplied texture will be locked automatically until the command buffer is submitted and must **not** be locked by the caller
         */
        bool AttachTextureView(HostTextureView *view);

        /**
         * @brief Attach the lifetime of the texture to the command buffer
         * @return If this is the first usage of the backing of this resource within this execution
         * @note The supplied texture will be locked automatically until the command buffer is submitted and must **not** be locked by the caller
         */
        bool AttachTexture(std::shared_ptr<Texture> texture);

        /**
         * @brief Attach the lifetime of a buffer view to the command buffer
         * @return If this is the first usage of the backing of this resource within this execution
         * @note The supplied buffer will be locked automatically until the command buffer is submitted and must **not** be locked by the caller
         * @note This'll automatically handle syncing of the buffer in the most optimal way possible
         */
        bool AttachBuffer(BufferView &view);

        /**
         * @brief Attach the lifetime of a buffer view that's already locked to the command buffer
         * @note The supplied buffer **must** be locked with the executor's tag
         * @note There must be no other external locks on the buffer aside from the supplied lock
         * @note This'll automatically handle syncing of the buffer in the most optimal way possible
         */
        void AttachLockedBufferView(BufferView &view, ContextLock<BufferView> &&lock);

        /**
         * @brief Attach the lifetime of a buffer object that's already locked to the command buffer
         * @note The supplied buffer **must** be locked with the executor's tag
         * @note There must be no other external locks on the buffer aside from the supplied lock
         * @note This'll automatically handle syncing of the buffer in the most optimal way possible
         */
        void AttachLockedBuffer(std::shared_ptr<Buffer> buffer, ContextLock<Buffer> &&lock);

        /**
         * @brief Attach the lifetime of the fence cycle dependency to the command buffer
         */
        void AttachDependency(const std::shared_ptr<void> &dependency);

        void CreateTransferPass();

        /**
         * @brief Adds a command that needs to be executed inside a subpass configured with certain attachments
         * @note Any supplied texture should be attached prior and not undergo any persistent layout transitions till execution
         * @note Any texture views may be nullptr, in which case the texture will be ignored
         */
        void AddSubpass(ExecutorCommand function, vk::Rect2D renderArea, span<std::pair<HostTextureView *, TextureSyncRequestArgs>> sampledImages, span<HostTextureView *> colorAttachments, HostTextureView *depthStencilAttachment = {}, vk::PipelineStageFlags srcStageMask = {}, vk::PipelineStageFlags dstStageMask = {});

        void AddTransferSubpass();

        /**
         * @brief Adds a subpass that clears the entirety of the specified attachment with a color value, it may utilize VK_ATTACHMENT_LOAD_OP_CLEAR for a more efficient clear when possible
         * @note Any supplied texture should be attached prior and not undergo any persistent layout transitions till execution
         */
        void AddClearColorSubpass(vk::Rect2D renderArea, HostTextureView *attachment, const vk::ClearColorValue &value);

        /**
         * @brief Adds a subpass that clears the entirety of the specified attachment with a depth/stencil value, it may utilize VK_ATTACHMENT_LOAD_OP_CLEAR for a more efficient clear when possible
         * @note Any supplied texture should be attached prior and not undergo any persistent layout transitions till execution
         */
        void AddClearDepthStencilSubpass(vk::Rect2D renderArea, HostTextureView *attachment, const vk::ClearDepthStencilValue &value);

        /**
         * @brief Adds a command that needs to be executed outside the scope of a render pass
         */
        void AddOutsideRpCommand(ExecutorCommand function);

        /**
         * @brief Adds a command that can be executed inside or outside of an RP
         */
        void AddCommand(ExecutorCommand function);

        /**
         * @brief Inserts the input command into the node list at the beginning of the execution
         */
        void InsertPreExecuteCommand(ExecutorCommand function);

        /**
         * @brief Inserts the input command into the node list before the current RP begins (or immediately if not in an RP)
         */
        void InsertPreRpCommand(ExecutorCommand function);

        /**
         * @brief Inserts the input command into the node list after the beginning of the current RP (or immediately if not in an RP)
         */
        void InsertRpBeginCommand(ExecutorCommand function);

        /**
         * @brief Inserts the input command into the node list after the current RP (or execution) finishes
         */
        void InsertPostRpCommand(ExecutorCommand function);

        /**
         * @brief Inserts the input command into the node list right before the current RP (or execution) finishes
         */
        void InsertRpEndCommand(ExecutorCommand function);

        /**
         * @brief Ends a render pass if one is currently active and resets all corresponding state
         */
        void FinishRenderPass();

        void AddTextureBarrier(HostTexture &toWait, const TextureSyncRequestArgs &args);

        void AddTextureTransferCommand(HostTexture &toWait, const TextureSyncRequestArgs &args, ExecutorCommand function);

        void AddStagedTextureTransferCommand(HostTexture &toWait, const TextureSyncRequestArgs &args, const memory::StagingBuffer &stagingBuffer, ExecutorCommand preStagingFunction, ExecutorCommand postStagingFunction);

        /**
         * @brief Adds a full pipeline barrier to the command buffer
         */
        void AddFullBarrier();

        /**
         * @brief Adds a persistent callback that will be called at the start of Execute in order to flush data required for recording
         */
        void AddFlushCallback(std::function<void()> &&callback);

        /**
         * @brief Adds a persistent callback that will be called after any non-Maxwell 3D engine changes the active pipeline
         */
        void AddPipelineChangeCallback(std::function<void()> &&callback);

        /**
         * @brief Calls all registered pipeline change callbacks
         */
        void NotifyPipelineChange();

        std::optional<u32> GetRenderPassIndex();

        /**
         * @brief Records a checkpoint into the GPU command stream at the current
         * @param annotation A string annotation to display in perfetto for this checkpoint
         * @return The checkpoint ID
         */
        u32 AddCheckpoint(std::string_view annotation) {
            if constexpr (EnableGpuCheckpoints)
                return AddCheckpointImpl(annotation);
            else
                return 0;
        }

        /**
         * @brief Execute all the nodes and submit the resulting command buffer to the GPU
         * @param callback A function to call after command buffer submission
         * @param wait Whether to wait synchronously for GPU completion of the submit
         * @param destroyTextures Wheter to flush stale textures from the texture manager
         */
        void Submit(std::function<void()> &&callback = {}, bool wait = false, bool destroyTextures = true);

        /**
         * @brief Adds an action to be executed upon current cycle completion (if DMI is on, otherwise after submission)
         */
        void AddDeferredAction(std::function<void()> &&callback);

        /**
         * @brief Locks all preserve attached buffers/textures
         * @note This **MUST** be called before attaching any buffers/textures to an execution
         */
        void LockPreserve();

        /**
         * @brief Unlocks all preserve attached buffers/textures
         * @note This **MUST** be called when there is no GPU work left to be done to avoid deadlocks where the guest will try to lock a buffer/texture but the GPFIFO thread has no work so won't periodically unlock it
         */
        void UnlockPreserve();
    };
}
