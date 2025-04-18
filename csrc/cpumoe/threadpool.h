#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

class ParallelExecutor {
  public:
#ifdef USE_NUMA
    static thread_local int numa_node;
#endif
    static thread_local int thread_local_id;

    explicit ParallelExecutor(int max_thread_num = 0) {
        max_thread_num_ = (max_thread_num > 0)
                              ? max_thread_num
                              : std::thread::hardware_concurrency() / 2;
        running_ = true;

        for (int i = 0; i < max_thread_num_; ++i) {
            threads_.emplace_back([this, i]() {
                thread_local_id = i;

#ifdef USE_NUMA
                if (numa_node == -1) {
                    numa_node =
                        i * numa_num_configured_nodes() / max_thread_num_;
                    struct bitmask *mask =
                        numa_bitmask_alloc(numa_num_configured_nodes());
                    numa_bitmask_setbit(mask, numa_node);
                    numa_bind(mask);
                    numa_bitmask_free(mask);
                }
#endif

                worker_loop(i);
            });
        }
    }

    ~ParallelExecutor() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            running_ = false;
        }
        cv_task_.notify_all();

        for (auto &thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    int get_thread_num() { return max_thread_num_; }

    void do_work_stealing_job(int task_num,
                              std::function<void(int)> compute_func) {
        if (task_num <= 0)
            return;

        auto task = std::make_shared<TaskContext>();
        task->task_count = task_num;
        task->next_task = 0;
        task->completed_tasks = 0;
        task->compute_func = compute_func;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            task_queue_.push(task);
        }
        cv_task_.notify_all();
    }

    void wait() {
        std::shared_ptr<TaskContext> task;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!task_queue_.empty()) {
                task = task_queue_.back();
            } else {
                return;
            }
        }

        if (task) {
            std::unique_lock<std::mutex> lock(task->mutex);
            task->cv.wait(lock, [&task] {
                return task->completed_tasks >= task->task_count;
            });
        }
    }

  private:
    struct TaskContext {
        int task_count;
        std::atomic<int> next_task{0};
        std::atomic<int> completed_tasks{0};
        std::function<void(int)> compute_func;
        std::mutex mutex;
        std::condition_variable cv;
    };

    void worker_loop(int thread_id) {
        while (true) {
            std::shared_ptr<TaskContext> task;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                while (running_ && task_queue_.empty()) {
                    cv_task_.wait(lock);
                }

                if (!running_)
                    break;

                if (!task_queue_.empty()) {
                    task = task_queue_.front();
                }
            }

            if (task) {
                while (true) {
                    int task_id = task->next_task.fetch_add(1);

                    if (task_id >= task->task_count) {
                        break;
                    }

                    if (task->compute_func) {
                        task->compute_func(task_id);
                    }

                    int completed = task->completed_tasks.fetch_add(1) + 1;

                    if (completed == task->task_count) {
                        {
                            std::lock_guard<std::mutex> lock(mutex_);
                            if (!task_queue_.empty() &&
                                task_queue_.front() == task) {
                                task_queue_.pop();
                            }
                        }

                        {
                            std::lock_guard<std::mutex> lock(task->mutex);
                            task->cv.notify_all();
                        }

                        break;
                    }
                }
            }
        }
    }

    int max_thread_num_;
    std::vector<std::thread> threads_;
    std::mutex mutex_;
    std::condition_variable cv_task_;
    bool running_;
    std::queue<std::shared_ptr<TaskContext>> task_queue_;
};