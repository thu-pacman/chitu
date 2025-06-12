#include <fstream>
#include <set>
#include <stdexcept>
#include <tuple>

#include <sched.h>

#include "affinity.h"

int get_physical_cpu_id_from_logical_cpu_id(
    const char *hardware_type /* one of: physical_package, die, core */,
    int logical_cpu_id) {
    // cat
    // /sys/devices/system/cpu/cpu{logical_cpu_id}/topology/{hardware_type}_id
    auto path = "/sys/devices/system/cpu/cpu" + std::to_string(logical_cpu_id) +
                "/topology/" + hardware_type + "_id";
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error(
            "Unable to query " + std::string(hardware_type) +
            " ID for logical CPU " + std::to_string(logical_cpu_id) + " from " +
            path);
    }
    int id;
    ifs >> id;
    return id;
}

int count_available_logical_cpus() {
    // We can't assume all CPUs on the system are available for us.
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask)) {
        throw std::runtime_error("Failed to get current thread affinity");
    }
    return CPU_COUNT(&mask);
}

int count_available_physical_cpus() {
    // We can't assume all CPUs on the system are available for us.
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask)) {
        throw std::runtime_error("Failed to get current thread affinity");
    }

    std::set<std::tuple<int /* physical package ID */, int /* die ID */,
                        int /* core ID */>>
        counted_physical_cores_;
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &mask)) {
            int physical_package_id =
                get_physical_cpu_id_from_logical_cpu_id("physical_package", i);
            int die_id = get_physical_cpu_id_from_logical_cpu_id("die", i);
            int core_id = get_physical_cpu_id_from_logical_cpu_id("core", i);
            auto core_tuple =
                std::make_tuple(physical_package_id, die_id, core_id);
            counted_physical_cores_.insert(core_tuple);
        }
    }
    return counted_physical_cores_.size();
}

int count_available_cpus(
    const std::string
        &thread_binding_policy /* one of: physical_core, logical_core */) {
    if (thread_binding_policy == "physical_core") {
        return count_available_physical_cpus();
    } else if (thread_binding_policy == "logical_core") {
        return count_available_logical_cpus();
    } else {
        throw std::invalid_argument("Invalid thread_binding_policy: " +
                                    thread_binding_policy);
    }
}
