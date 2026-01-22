#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# 自动检测 InfiniBand 配置并设置相关环境变量

detect_ib_cards() {
    # 检查 ibstat 命令是否可用
    if ! command -v ibstat &> /dev/null; then
        echo "Warning: ibstat command not available, skipping IB card detection" >&2
        return 1
    fi

    # 获取所有状态为 Active 的 IB 卡及其速率
    local ib_info=$(ibstat 2>/dev/null | awk '
        /^CA '\''/ { 
            ca_name = $2; 
            gsub(/'\''/, "", ca_name); 
        }
        /State:/ { 
            if ($2 == "Active") state[ca_name] = 1; 
        }
        /Rate:/ { 
            if (state[ca_name] == 1) {
                rate_str = $2;
                # 提取数字部分作为速率
                if (match(rate_str, /[0-9]+/)) {
                    rate[ca_name] = substr(rate_str, RSTART, RLENGTH);
                }
            }
        }
        END {
            for (ca in state) {
                if (state[ca] == 1) {
                    print ca, (rate[ca] ? rate[ca] : 0);
                }
            }
        }
    ')

    if [ -z "$ib_info" ]; then
        echo "Warning: No active IB cards found" >&2
        return 1
    fi

    # 找出最高速率
    local max_rate=$(echo "$ib_info" | awk '{print $2}' | sort -rn | head -n 1)

    # 筛选出速率最高的 IB 卡
    local ib_cards=$(echo "$ib_info" | awk -v max_rate="$max_rate" '$2 == max_rate {print $1}' | tr '\n' ',' | sed 's/,$//')

    if [ -z "$ib_cards" ]; then
        echo "Warning: Failed to get IB card list" >&2
        return 1
    fi

    echo "$ib_cards"
    return 0
}

detect_ib_network_interface() {
    # 检查 ibdev2netdev 命令是否可用
    if ! command -v ibdev2netdev &> /dev/null; then
        echo "Warning: ibdev2netdev command not available, skipping IB network interface detection" >&2
        return 1
    fi

    # 获取 IB 设备对应的网络接口
    local netdev_info=$(ibdev2netdev 2>/dev/null | grep "Up" | awk '{print $5}')

    if [ -z "$netdev_info" ]; then
        echo "Warning: No IB network interfaces in Up state found" >&2
        return 1
    fi

    # 优先选择 bond 接口
    local bond_iface=$(echo "$netdev_info" | grep -E '^bond' | head -n 1)
    if [ -n "$bond_iface" ]; then
        echo "$bond_iface"
        return 0
    fi

    # 如果没有 bond 接口，选择第一个可用接口
    local first_iface=$(echo "$netdev_info" | head -n 1)
    if [ -n "$first_iface" ]; then
        echo "$first_iface"
        return 0
    fi

    echo "Warning: Failed to get IB network interface" >&2
    return 1
}

# 自动配置 IB 相关环境变量
auto_configure_ib() {
    echo "Starting InfiniBand configuration auto-detection..." >&2

    # 检测 IB 卡
    local ib_cards=$(detect_ib_cards)
    if [ $? -eq 0 ] && [ -n "$ib_cards" ]; then
        export NCCL_IB_HCA="$ib_cards"
        export NVSHMEM_HCA_LIST="$ib_cards"
        echo "Detected IB cards: $ib_cards" >&2
        echo "Set NCCL_IB_HCA=$ib_cards" >&2
        echo "Set NVSHMEM_HCA_LIST=$ib_cards" >&2
    fi

    # 检测 IB 网络接口
    local ib_iface=$(detect_ib_network_interface)
    if [ $? -eq 0 ] && [ -n "$ib_iface" ]; then
        export GLOO_SOCKET_IFNAME="$ib_iface"
        export NCCL_SOCKET_IFNAME="$ib_iface"
        export HCCL_SOCKET_IFNAME="$ib_iface"
        export NVSHMEM_IB_DEVICE="$ib_iface"
        echo "Detected IB network interface: $ib_iface" >&2
        echo "Set GLOO_SOCKET_IFNAME=$ib_iface" >&2
        echo "Set NCCL_SOCKET_IFNAME=$ib_iface" >&2
        echo "Set HCCL_SOCKET_IFNAME=$ib_iface" >&2
        echo "Set NVSHMEM_IB_DEVICE=$ib_iface" >&2
    fi

    echo "InfiniBand configuration detection completed" >&2
}

# 如果直接执行此脚本，则运行自动配置
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    auto_configure_ib
fi
