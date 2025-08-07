/*
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

int get_physical_cpu_id_from_logical_cpu_id(
    const char *hardware_type /* one of: physical_package, die, core */,
    int logical_cpu_id);

int count_available_logical_cpus();

int count_available_physical_cpus();
