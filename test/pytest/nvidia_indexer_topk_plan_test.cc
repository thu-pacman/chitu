// SPDX-FileCopyrightText: 2025 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0

#include "nvidia_indexer_topk_plan.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <iostream>

using namespace chitu::nvidia_topk_plan;

// Reference the pre-SM-scaling H20 worker caps, not the new multiplier table.
// Row-capped entries only admit P1, where the old grid was exactly rows.
int legacy_h20_cap(int rows, int width, bool prefill) {
    const int r = row_bucket(rows), c = width_bucket(width);
    if (!prefill)
        return r == 8 ? 312 : (r == 11 ? 624 : 156);
    if (r == 5 && c >= 9)
        return c == 11 ? 624 : 312;
    if (r == 6 && c >= 8)
        return c == 8 ? 624 : 312;
    return 156;
}

int main() {
    const int widths[] = {2048,   4096,    8192,   16384,  32768,
                          65536,  98304,   131072, 262144, 524288,
                          786432, 1048576, 2097152};
    const int sm_counts[] = {1, 7, 16, 78, 80, 114, 132, 144};
    int cases = 0;
    for (int rows = 1; rows <= 1025; ++rows) {
        for (int boundary : widths) {
            for (int offset : {-1, 0, 1}) {
                const int width = boundary + offset;
                if (width < 2048)
                    continue;
                for (bool prefill : {false, true}) {
                    const auto cfg = config(rows, width, prefill);
                    const int p = max_parts(rows, width, prefill);
                    assert(cfg.sm_multiplier == 0 || cfg.sm_multiplier == 2 ||
                           cfg.sm_multiplier == 4 || cfg.sm_multiplier == 8);
                    if (cfg.sm_multiplier == 0)
                        assert(p == 1);
                    const auto legacy =
                        p == 1 ? rows
                               : std::max(rows,
                                          std::min(rows * p,
                                                   legacy_h20_cap(rows, width,
                                                                  prefill)));
                    assert(grid_blocks(cfg, rows, p, 78) == uint32_t(legacy));
                    uint32_t previous = 0;
                    for (int sms : sm_counts) {
                        const auto blocks = grid_blocks(cfg, rows, p, sms);
                        const auto cap = cfg.sm_multiplier == 0
                                             ? rows
                                             : sms * cfg.sm_multiplier;
                        assert(blocks == uint32_t(std::max(
                                             rows, std::min(rows * p, cap))));
                        assert(blocks >= uint32_t(rows) &&
                               blocks <= uint32_t(rows * p));
                        assert(blocks >= previous);
                        previous = blocks;
                    }
                    ++cases;
                }
            }
        }
    }
    // Coefficients are queue size per SM, not resident-CTA guarantees.
    assert(grid_blocks({0, 2}, 32, 16, 78) == 156);
    assert(grid_blocks({0, 2}, 32, 16, 132) == 264);
    assert(grid_blocks({0, 4}, 32, 16, 132) == 512); // task-count limit
    assert(grid_blocks({0, 8}, 320, 16, 16) == 320); // row-count floor
    assert(grid_blocks({0, 0}, 96, 1, 132) == 96);
    // Multiplication stays 64-bit before the validated uint32 grid result.
    assert(grid_blocks({0, 8}, INT_MAX, 2, INT_MAX) == UINT32_MAX - 1);
    std::cout << "PASS " << cases
              << " shapes; H20 equivalence and 8 SM counts\n";
}
