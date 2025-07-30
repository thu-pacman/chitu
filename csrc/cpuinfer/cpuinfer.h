#ifndef CPU_INFER_HPP
#define CPU_INFER_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <set>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <vector>

#include <immintrin.h> // for _mm_pause on x86
#include <sched.h>

#ifdef USE_CUDA
#include "vendors/cuda.h"
#elif USE_MUSA
#include "vendors/musa.h"
#endif

#include "affinity.h"
#include "llama.cpp/ggml-impl.h"

inline bool parse_bind_physical_core(
    const std::string
        &thread_binding_policy /* one of: physical_core, logical_core */) {
    if (thread_binding_policy == "physical_core") {
        return true;
    } else if (thread_binding_policy == "logical_core") {
        return false;
    } else {
        throw std::invalid_argument("Invalid thread_binding_policy: " +
                                    thread_binding_policy);
    }
}

class CPUInfer {
  public:
    enum class WorkerState { Idle, Active, Terminate };

    struct alignas(64) WorkerContext {
        std::atomic_int task_counter;
        std::atomic<WorkerState> state;
        int task_end;
    };

  private:
    struct Platform {
#ifdef USE_CUDA
        using Stream = cudaStream_t;
#elif USE_MUSA
        using Stream = musaStream_t;
#endif
    };
    using GenericHostFn = void (*)(void *);

    // Core components
    std::atomic_bool sync_flag_;
    std::atomic_bool shutdown_flag_;
    std::vector<WorkerContext> workers_;
    std::vector<std::thread> worker_pool_;

    // Physical core management
    bool bind_to_physical_core_;
    std::mutex bind_core_mutex_;
    std::set<std::tuple<int /* physical package ID */, int /* die ID */,
                        int /* core ID */>>
        used_physical_cores_;
    std::unordered_set<int> used_logical_cores_;

    // Task management
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::queue<std::function<void()>> task_queue_;

    // Execution context
    std::thread dispatcher_;
    std::function<void(int)> compute_callback_;
    int active_workers_;

    void initialize_workers() {
        worker_pool_.reserve(workers_.size());
        for (size_t i = 1; i < workers_.size(); ++i) {
            worker_pool_.emplace_back(&CPUInfer::worker_routine, this, i);
        }
    }

    void shutdown_workers() {
        for (auto &ctx : workers_) {
            ctx.state.store(WorkerState::Terminate, std::memory_order_release);
        }
        for (auto &t : worker_pool_) {
            if (t.joinable())
                t.join();
        }
    }

    void worker_routine(int worker_id) {
        if (bind_to_physical_core_) {
            bind_cur_thread_to_first_available_physical_core();
        } else {
            bind_cur_thread_to_first_available_logical_core();
        }
        auto last_active = std::chrono::steady_clock::now();
        while (true) {
            switch (workers_[worker_id].state.load(std::memory_order_acquire)) {
            case WorkerState::Active:
                process_tasks(worker_id);
                last_active = std::chrono::steady_clock::now();
                break;
            case WorkerState::Terminate:
                return;
            case WorkerState::Idle:
                using namespace std::chrono_literals;
                constexpr auto IDLE_THRESHOLD = 50ms;
                constexpr auto SLEEP_DURATION = 1ms;
                if (std::chrono::steady_clock::now() - last_active >
                    IDLE_THRESHOLD) {
                    std::this_thread::sleep_for(SLEEP_DURATION);
                } else {
                    _mm_pause();
                }
                break;
            }
        }
    }

    void process_tasks(int worker_id) {
        auto &ctx = workers_[worker_id];

        while (true) {
            int task_id =
                ctx.task_counter.fetch_add(1, std::memory_order_acq_rel);
            if (task_id >= ctx.task_end) {
                break;
            }
            compute_callback_(task_id);
        }
        for (int t_offset = 1; t_offset < workers_.size(); t_offset++) {
            int t_i = (worker_id + t_offset) % workers_.size();
            if (workers_[t_i].state.load(std::memory_order_acquire) !=
                WorkerState::Active) {
                continue;
            }
            while (true) {
                int task_id = workers_[t_i].task_counter.fetch_add(
                    1, std::memory_order_acq_rel);
                if (task_id >= workers_[t_i].task_end) {
                    break;
                }
                compute_callback_(task_id);
            }
        }
        workers_[worker_id].state.store(WorkerState::Idle,
                                        std::memory_order_release);
    }

    int bind_cur_thread_to_first_available_physical_core() {
        std::lock_guard lock(bind_core_mutex_);

        // We can't assume all CPUs on the system are available for us. Query
        // the availability first.
        cpu_set_t old_mask;
        if (sched_getaffinity(0, sizeof(cpu_set_t), &old_mask)) {
            throw std::runtime_error("Failed to get current thread affinity");
        }
        for (int i = 0; i < CPU_SETSIZE; i++) {
            if (CPU_ISSET(i, &old_mask)) {
                // Find a physical core that is not already used
                int physical_package_id =
                    get_physical_cpu_id_from_logical_cpu_id("physical_package",
                                                            i);
                int die_id = get_physical_cpu_id_from_logical_cpu_id("die", i);
                int core_id =
                    get_physical_cpu_id_from_logical_cpu_id("core", i);
                auto core_tuple =
                    std::make_tuple(physical_package_id, die_id, core_id);
                if (used_physical_cores_.find(core_tuple) ==
                    used_physical_cores_.end()) {
                    used_physical_cores_.insert(core_tuple);

                    // Bind the current thread to this physical core
                    cpu_set_t new_mask;
                    CPU_ZERO(&new_mask);
                    CPU_SET(i, &new_mask);
                    if (sched_setaffinity(0, sizeof(cpu_set_t), &new_mask) !=
                        0) {
                        throw std::runtime_error(
                            "Failed to bind thread to physical core " +
                            std::to_string(i));
                    }
                    return i;
                }
            }
        }

        throw std::runtime_error(
            "No available physical cores found for "
            "binding the current thread. Please set the number of workers to "
            "be no greater than the number of physical cores.");
    }

    int bind_cur_thread_to_first_available_logical_core() {
        std::lock_guard lock(bind_core_mutex_);

        // We can't assume all CPUs on the system are available for us. Query
        // the availability first.
        cpu_set_t old_mask;
        if (sched_getaffinity(0, sizeof(cpu_set_t), &old_mask)) {
            throw std::runtime_error("Failed to get current thread affinity");
        }
        for (int i = 0; i < CPU_SETSIZE; i++) {
            if (CPU_ISSET(i, &old_mask)) {
                if (used_logical_cores_.find(i) == used_logical_cores_.end()) {
                    used_logical_cores_.insert(i);

                    // Bind the current thread to this logical core
                    cpu_set_t new_mask;
                    CPU_ZERO(&new_mask);
                    CPU_SET(i, &new_mask);
                    if (sched_setaffinity(0, sizeof(cpu_set_t), &new_mask) !=
                        0) {
                        throw std::runtime_error(
                            "Failed to bind thread to logical core " +
                            std::to_string(i));
                    }
                    return i;
                }
            }
        }

        throw std::runtime_error(
            "No available logical cores found for "
            "binding the current thread. Please set the number of workers to "
            "be no greater than the number of logical cores.");
    }

  public:
    explicit CPUInfer(
        const std::string
            &thread_binding_policy /* one of: physical_core, logical_core */)
        : CPUInfer(parse_bind_physical_core(thread_binding_policy)) {}

    explicit CPUInfer(bool bind_to_physical_core)
        : CPUInfer(bind_to_physical_core,
                   bind_to_physical_core ? count_available_physical_cpus()
                                         : count_available_logical_cpus()) {}

    explicit CPUInfer(bool bind_to_physical_core, int num_workers)
        : workers_(num_workers), bind_to_physical_core_(bind_to_physical_core),
          active_workers_(num_workers), sync_flag_(true),
          shutdown_flag_(false) {
        for (auto &ctx : workers_) {
            ctx.state.store(WorkerState::Idle, std::memory_order_relaxed);
        }

        for (int i = 0; i < (1 << 16); ++i) {
            ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(i);
        }

        dispatcher_ = std::thread(&CPUInfer::dispatch_tasks, this);
        initialize_workers();
    }

    ~CPUInfer() {
        shutdown_flag_.store(true, std::memory_order_release);
        queue_cv_.notify_all();

        if (dispatcher_.joinable())
            dispatcher_.join();
        shutdown_workers();
    }

    template <typename Fn, typename... Args>
    void enqueue(Fn &&fn, Args &&...args) {
        {
            std::lock_guard lock(queue_mutex_);
            task_queue_.emplace(std::bind(std::forward<Fn>(fn),
                                          std::forward<Args>(args)..., this));
            sync_flag_.store(false, std::memory_order_seq_cst);
        }
        queue_cv_.notify_one();
    }

    void sync() {
        while (!sync_flag_.load(std::memory_order_seq_cst)) {
            std::this_thread::yield();
        }
    }

    void parallel_for(int task_count, std::function<void(int)> compute_fn) {
        compute_callback_ = compute_fn;
        active_workers_ =
            std::min(workers_.size(), static_cast<size_t>(task_count));

        const int base_tasks = task_count / active_workers_;
        int remaining = task_count % active_workers_;

        workers_[0].task_counter.store(0, std::memory_order_relaxed);
        workers_[0].task_end = base_tasks + (remaining-- > 0);
        workers_[0].state.store(WorkerState::Active, std::memory_order_release);

        for (size_t i = 1; i < active_workers_; ++i) {
            workers_[i].task_counter.store(workers_[i - 1].task_end,
                                           std::memory_order_relaxed);
            workers_[i].task_end =
                workers_[i - 1].task_end + base_tasks + (remaining-- > 0);
            workers_[i].state.store(WorkerState::Active,
                                    std::memory_order_release);
        }

        process_tasks(0);
        for (size_t i = 1; i < active_workers_; ++i) {
            while (workers_[i].state.load(std::memory_order_acquire) ==
                   WorkerState::Active) {
                std::this_thread::yield();
            }
        }
    }

    void submit(std::pair<intptr_t, intptr_t> params) {
        void (*func)(void *) = (void (*)(void *))params.first;
        void *args = (void *)params.second;
        *((CPUInfer **)args) = this;
        func(args);
    }

    void submit_with_cuda_stream(intptr_t stream,
                                 std::pair<intptr_t, intptr_t> params) {
#ifdef USE_CUDA
        void (*func)(void *) = (void (*)(void *))params.first;
        void *args = (void *)params.second;
        *((CPUInfer **)args) = this;
        cudaLaunchHostFunc(reinterpret_cast<Platform::Stream>(stream),
                           (GenericHostFn)func, args);
#elif USE_MUSA
        // MUSA implementation
#endif
    }

    static void sync_(void *cpu_infer_ptr) {
        CPUInfer *cpuinfer = (CPUInfer *)cpu_infer_ptr;
        cpuinfer->sync();
    }

    void sync_with_cuda_stream(intptr_t stream) {
        cudaLaunchHostFunc(reinterpret_cast<Platform::Stream>(stream),
                           (GenericHostFn)&sync_, (void *)this);
    }

  private:
    void dispatch_tasks() {
        if (bind_to_physical_core_) {
            bind_cur_thread_to_first_available_physical_core();
        } else {
            bind_cur_thread_to_first_available_logical_core();
        }
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock lock(queue_mutex_);
                queue_cv_.wait(lock, [this] {
                    return !task_queue_.empty() ||
                           shutdown_flag_.load(std::memory_order_acquire);
                });

                if (shutdown_flag_ && task_queue_.empty())
                    return;

                task = std::move(task_queue_.front());
                task_queue_.pop();
            }

            task();

            {
                std::lock_guard lock(queue_mutex_);
                if (task_queue_.empty()) {
                    sync_flag_.store(true, std::memory_order_release);
                }
            }
        }
    }
};

#endif // CPU_INFER_HPP
