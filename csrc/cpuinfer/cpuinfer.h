#ifndef CPU_INFER_HPP
#define CPU_INFER_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#ifdef USE_CUDA
#include "vendors/cuda.h"
#elif USE_MUSA
#include "vendors/musa.h"
#endif

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

#include "llama.cpp/ggml-impl.h"

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

    // Task management
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::queue<std::function<void()>> task_queue_;

    // Execution context
    std::thread dispatcher_;
    std::function<void(int)> compute_callback_;
    int active_workers_;

    // Thread-local storage
#ifdef USE_NUMA
    static thread_local int numa_node_;
#endif
    static thread_local int worker_id_;

    // Platform-specific operations
    void bind_numa_node(int node) {
#ifdef USE_NUMA
        numa_node_ = node * numa_num_configured_nodes() / active_workers_;
        numa_bitmask *mask = numa_bitmask_alloc(numa_num_configured_nodes());
        numa_bitmask_setbit(mask, numa_node_);
        numa_bind(mask);
        numa_bitmask_free(mask);
#endif
    }

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
        worker_id_ = worker_id;
        bind_numa_node(worker_id);

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
                }
                break;
            }
        }
    }

    void process_tasks(int worker_id) {
        auto &ctx = workers_[worker_id];

        while (execute_task(ctx)) {
        }

        for (size_t offset = 1; offset < workers_.size(); ++offset) {
            int target = (worker_id + offset) % workers_.size();
            if (workers_[target].state.load(std::memory_order_acquire) !=
                WorkerState::Active)
                continue;

            while (execute_task(workers_[target])) {
            }
        }

        ctx.state.store(WorkerState::Idle, std::memory_order_release);
    }

    bool execute_task(WorkerContext &ctx) {
        const int task_id =
            ctx.task_counter.fetch_add(1, std::memory_order_acq_rel);
        if (task_id >= ctx.task_end)
            return false;

        compute_callback_(task_id);
        return true;
    }

  public:
    explicit CPUInfer(size_t num_workers)
        : workers_(num_workers), active_workers_(num_workers), sync_flag_(true),
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
            sync_flag_.store(false, std::memory_order_release);
        }
        queue_cv_.notify_one();
    }

    void sync() {
        while (!sync_flag_.load(std::memory_order_acquire)) {
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
                sync_flag_.store(task_queue_.empty(),
                                 std::memory_order_release);
            }
        }
    }
};

#endif // CPU_INFER_HPP