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

    namespace interconnect {
        class CommandExecutor;
    }

    class TextureUsageTracker {
      private:
        GPU &gpu;
        struct TextureUsageInfo {
            Texture *texture;
            HostTexture *dirtyTexture{}; //!< If this isn't nullptr then it's a pointer to the HostTexture that must be flushed
            u64 sequence{}; //!< A counter to keep track of the order sync should take place in
        };

        std::recursive_mutex mutex{};

        using TextureMap = IntervalMap<u8 *, TextureUsageInfo>;
        TextureMap infoMap;
        u64 lastSequence{};
        u32 fastPathCount{};
        u32 overallCount{};

        void FilterOverlaps(std::vector<std::reference_wrapper<TextureUsageInfo>> &intervals, boost::container::small_vector<span<u8>, 3> &onArea);

      public:
        using TextureHandle = TextureMap::GroupHandle;

        TextureUsageTracker() = delete;
        TextureUsageTracker(GPU &gpu);

        ~TextureUsageTracker() = default;

        TextureHandle AddTexture(Texture &texture);

        void MarkClean(TextureHandle texture);

        bool ShouldSyncGuest(TextureHandle texture);

        bool ShouldSyncHost(TextureHandle texture);

        void RemoveTexture(TextureHandle texture);

        bool RequestSync(interconnect::CommandExecutor &executor, TextureHandle texture, HostTexture *toSync, bool markDirty);
    };
}
