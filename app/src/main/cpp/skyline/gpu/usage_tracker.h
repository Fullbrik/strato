// SPDX-License-Identifier: MPL-2.0
// Copyright © 2022 Skyline Team and Contributors (https://github.com/skyline-emu/)

#pragma once

#include <list>
#include <common/interval_list.h>
#include <common/interval_map.h>
#include <mutex>

namespace skyline::gpu {

    /**
     * @brief Tracks the usage of GPU memory and (only) buffers to allow for fine-grained flushing
     */
    struct UsageTracker {
        IntervalList<u8 *> dirtyIntervals; //!< Intervals of GPU-dirty contents that requires a flush before accessing
        IntervalList<u8 *> sequencedIntervals; //!< Intervals of GPFIFO-sequenced writes that occur within an execution
    };

    class GPU;
    class Texture;
    class HostTexture;
    struct TextureSyncRequestArgs;

    namespace interconnect {
        class CommandExecutor;
    }

    class TextureUsageTracker {
      private:
        GPU &gpu;
        struct TextureUsageInfo {
            Texture *texture;
            HostTexture *dirtyTexture{}; //!< Points to the HostTexture that contains the most up to date data. If guest contents are newer/the same then it's nullptr
            u64 sequence{}; //!< A counter that is used to determine which texture has the newest data, the higher the newer
        };

        std::recursive_mutex mutex{};

        using TextureMap = IntervalMap<u8 *, TextureUsageInfo>;
        TextureMap infoMap;
        bool incrementSequence{true};
        u64 lastSequence{};

      public:
        using TextureHandle = TextureMap::GroupHandle;
        using Overlaps = std::vector<std::reference_wrapper<TextureUsageInfo>>;

        TextureUsageTracker() = delete;
        TextureUsageTracker(GPU &gpu);

        ~TextureUsageTracker() = default;

        constexpr void EnableIncrementing(bool value) {
            ++lastSequence;
            incrementSequence = value;
        }

        TextureHandle AddTexture(Texture &texture);

        void MarkClean(TextureHandle texture);

        bool ShouldSyncGuest(TextureHandle texture);

        bool ShouldSyncHost(TextureHandle texture);

        Overlaps GetOverlaps(TextureHandle texture);

        void RemoveTexture(TextureHandle texture);

        void RequestSync(interconnect::CommandExecutor &executor, TextureHandle texture, HostTexture *toSync, const TextureSyncRequestArgs &args, bool createTransferPass);
    };
}
