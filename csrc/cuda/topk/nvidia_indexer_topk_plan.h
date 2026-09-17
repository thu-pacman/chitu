// SPDX-FileCopyrightText: 2025 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace chitu::nvidia_topk_plan {
// H20 (78 SM), FP32 K=2048. Decode fixes variant/grid by query rows;
// only P depends on live context. Prefill is eager and may select by width.
// Measured 2026-09-08: 322 shapes, 13,615 trials. The old inexact kernel
// is only a timing reference. Between samples use the next upper bucket;
// above 1M reuse the last P (exact but not autotuned). Grid targets scale
// with the launch device's SM count; 2/4/8 reproduce 156/312/624 on H20.
// Thread variants and P tables remain H20-tuned, not cross-device autotuning.
struct Config {
    int variant;
    // 0: one CTA per row; positive: total-grid target per device SM.
    int sm_multiplier;
};
constexpr Config decode_configs[] = {{0, 2}, {0, 2}, {0, 2}, {1, 2}, {0, 2},
                                     {0, 2}, {1, 2}, {1, 2}, {2, 4}, {2, 2},
                                     {2, 2}, {2, 8}, {2, 2}, {2, 2}, {2, 2}};
struct PrefillConfig {
    int variant;
    int parts;
    int sm_multiplier;
};
constexpr PrefillConfig prefill_configs[15][12] = {{{2, 1, 0},
                                                    {2, 1, 0},
                                                    {0, 1, 0},
                                                    {0, 1, 0},
                                                    {0, 1, 2},
                                                    {1, 8, 2},
                                                    {0, 8, 2},
                                                    {1, 8, 2},
                                                    {1, 8, 2},
                                                    {0, 16, 2},
                                                    {0, 16, 2},
                                                    {0, 16, 2}},
                                                   {{0, 1, 2},
                                                    {2, 1, 0},
                                                    {0, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0},
                                                    {1, 8, 2},
                                                    {1, 8, 2},
                                                    {1, 8, 2},
                                                    {0, 8, 2},
                                                    {0, 16, 2},
                                                    {0, 16, 2},
                                                    {0, 16, 2}},
                                                   {{0, 1, 2},
                                                    {2, 1, 0},
                                                    {1, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 2},
                                                    {1, 8, 2},
                                                    {0, 8, 2},
                                                    {1, 8, 2},
                                                    {0, 8, 2},
                                                    {0, 16, 2},
                                                    {0, 16, 2},
                                                    {0, 16, 2}},
                                                   {{0, 1, 2},
                                                    {1, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0},
                                                    {0, 1, 2},
                                                    {1, 8, 2},
                                                    {1, 8, 2},
                                                    {1, 8, 2},
                                                    {1, 8, 2},
                                                    {0, 8, 2},
                                                    {1, 8, 2},
                                                    {1, 8, 2}},
                                                   {{0, 1, 2},
                                                    {1, 1, 0},
                                                    {1, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 2},
                                                    {1, 4, 2},
                                                    {0, 4, 2},
                                                    {1, 4, 2},
                                                    {0, 4, 2},
                                                    {1, 4, 2},
                                                    {0, 8, 2},
                                                    {0, 8, 2}},
                                                   {{0, 1, 2},
                                                    {2, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 2},
                                                    {0, 2, 2},
                                                    {0, 2, 2},
                                                    {0, 2, 2},
                                                    {0, 4, 2},
                                                    {2, 16, 4},
                                                    {0, 16, 4},
                                                    {2, 16, 8}},
                                                   {{0, 1, 2},
                                                    {1, 1, 0},
                                                    {1, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 2},
                                                    {1, 1, 2},
                                                    {1, 1, 2},
                                                    {1, 1, 2},
                                                    {2, 8, 8},
                                                    {2, 4, 4},
                                                    {2, 4, 4},
                                                    {2, 4, 4}},
                                                   {{0, 1, 2},
                                                    {1, 1, 0},
                                                    {0, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0},
                                                    {0, 1, 2},
                                                    {1, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 2}},
                                                   {{0, 1, 2},
                                                    {2, 1, 0},
                                                    {2, 1, 2},
                                                    {0, 1, 0},
                                                    {0, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0}},
                                                   {{0, 1, 2},
                                                    {2, 1, 0},
                                                    {2, 1, 2},
                                                    {0, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0}},
                                                   {{0, 1, 2},
                                                    {2, 1, 0},
                                                    {2, 1, 2},
                                                    {2, 1, 2},
                                                    {0, 1, 0},
                                                    {1, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0}},
                                                   {{0, 1, 2},
                                                    {2, 1, 0},
                                                    {2, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0},
                                                    {0, 1, 0},
                                                    {0, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0}},
                                                   {{0, 1, 2},
                                                    {2, 1, 0},
                                                    {2, 1, 2},
                                                    {2, 1, 2},
                                                    {0, 1, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0}},
                                                   {{0, 1, 2},
                                                    {2, 1, 0},
                                                    {2, 1, 0},
                                                    {2, 1, 0},
                                                    {0, 1, 0},
                                                    {0, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0}},
                                                   {{0, 1, 2},
                                                    {2, 1, 0},
                                                    {2, 1, 0},
                                                    {2, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0},
                                                    {0, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0},
                                                    {1, 1, 0},
                                                    {0, 1, 0},
                                                    {1, 1, 0}}};
constexpr int host_decode_parts[15][12] = {
    {1, 1, 1, 1, 1, 8, 8, 8, 8, 16, 16, 16},
    {1, 1, 1, 1, 1, 8, 8, 8, 8, 16, 16, 16},
    {1, 1, 1, 1, 1, 8, 8, 8, 8, 16, 16, 16},
    {1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8},
    {1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 8, 8},
    {1, 1, 1, 1, 1, 2, 2, 2, 4, 16, 16, 16},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 8, 16},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 4, 4, 2, 2, 2, 2},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
// TU-local device constants avoid both per-thread local tables and RDC.
static __device__ __constant__ int device_decode_parts[15][12] = {
    {1, 1, 1, 1, 1, 8, 8, 8, 8, 16, 16, 16},
    {1, 1, 1, 1, 1, 8, 8, 8, 8, 16, 16, 16},
    {1, 1, 1, 1, 1, 8, 8, 8, 8, 16, 16, 16},
    {1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8},
    {1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 8, 8},
    {1, 1, 1, 1, 1, 2, 2, 2, 4, 16, 16, 16},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 8, 16},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 4, 4, 2, 2, 2, 2},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
static __device__ __constant__ int device_prefill_parts[15][12] = {
    {1, 1, 1, 1, 1, 8, 8, 8, 8, 16, 16, 16},
    {1, 1, 1, 1, 1, 8, 8, 8, 8, 16, 16, 16},
    {1, 1, 1, 1, 1, 8, 8, 8, 8, 16, 16, 16},
    {1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8},
    {1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 8, 8},
    {1, 1, 1, 1, 1, 2, 2, 2, 4, 16, 16, 16},
    {1, 1, 1, 1, 1, 1, 1, 1, 8, 4, 4, 4},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
__host__ __device__ inline int row_bucket(int rows) {
    if (rows <= 1)
        return 0;
    if (rows <= 2)
        return 1;
    if (rows <= 4)
        return 2;
    if (rows <= 8)
        return 3;
    if (rows <= 16)
        return 4;
    if (rows <= 32)
        return 5;
    if (rows <= 48)
        return 6;
    if (rows <= 64)
        return 7;
    if (rows <= 96)
        return 8;
    if (rows <= 128)
        return 9;
    if (rows <= 192)
        return 10;
    if (rows <= 256)
        return 11;
    if (rows <= 320)
        return 12;
    if (rows <= 512)
        return 13;
    return 14;
}
__host__ __device__ inline int width_bucket(int width) {
    if (width <= 2048)
        return 0;
    if (width <= 4096)
        return 1;
    if (width <= 8192)
        return 2;
    if (width <= 16384)
        return 3;
    if (width <= 32768)
        return 4;
    if (width <= 65536)
        return 5;
    if (width <= 98304)
        return 6;
    if (width <= 131072)
        return 7;
    if (width <= 262144)
        return 8;
    if (width <= 524288)
        return 9;
    if (width <= 786432)
        return 10;
    return 11;
}
__host__ __device__ inline int parts(int length, int rows, bool prefill) {
    if (length <= 2048)
        return 1;
    const int r = row_bucket(rows), c = width_bucket(length);
#ifdef __CUDA_ARCH__
    return prefill ? device_prefill_parts[r][c] : device_decode_parts[r][c];
#else
    return prefill ? prefill_configs[r][c].parts : host_decode_parts[r][c];
#endif
}
inline Config config(int rows, int width, bool prefill) {
    const int r = row_bucket(rows);
    if (prefill) {
        const auto c = prefill_configs[r][width_bucket(width)];
        return {c.variant, c.sm_multiplier};
    }
    return decode_configs[r];
}

// Caller validates rows * max_parts fits uint32 and num_sms is positive.
// MaxParts and shape are static during graph capture; live lengths/plan_parts
// never change the physical grid. Keep at least rows CTAs for the P1 branch.
inline uint32_t grid_blocks(Config cfg, int rows, int max_parts, int num_sms) {
    const uint64_t logical_tasks = static_cast<uint64_t>(rows) * max_parts;
    const uint64_t cap =
        cfg.sm_multiplier == 0
            ? static_cast<uint64_t>(rows)
            : static_cast<uint64_t>(num_sms) * cfg.sm_multiplier;
    const uint64_t workers = logical_tasks < cap ? logical_tasks : cap;
    return static_cast<uint32_t>(
        workers < static_cast<uint64_t>(rows) ? rows : workers);
}

// Plans need not be monotone in context length. Inspect every admitted bucket
// before proving that a graph's static shape can never require multiple CTAs.
inline int max_parts(int rows, int width, bool prefill) {
    if (rows <= 0 || width <= 2048)
        return 1;
    const int r = row_bucket(rows), last = width_bucket(width);
    int result = 1;
    for (int c = 0; c <= last; ++c) {
        const int p =
            prefill ? prefill_configs[r][c].parts : host_decode_parts[r][c];
        result = p > result ? p : result;
    }
    return result;
}
inline bool use_pure_p1(int rows, int width, bool prefill) {
    // H20 A/B: retain the 512-thread path for the 192/256-row buckets;
    // forcing 1024 threads regresses 8K there without a long-context gain.
    return !prefill && (rows <= 128 || (rows > 256 && rows <= 320)) &&
           max_parts(rows, width, false) == 1;
}
} // namespace chitu::nvidia_topk_plan
