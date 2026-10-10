// LeanSift - fast SIFT feature extraction
// SPDX-License-Identifier: BSD-2-Clause

#include "lsift/executor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define LSIFT_PAUSE() _mm_pause()
#else
#define LSIFT_PAUSE() std::this_thread::yield()
#endif

namespace lsift {

void SerialExecutor::parallel_for(std::size_t n, std::size_t /*grain*/, const RangeFn& fn) {
  if (n > 0) fn(0, n);
}

namespace {
// Set while a thread runs ranges of a pool job, so nested parallel_for calls
// run serially instead of waiting on the pool they are part of.
thread_local bool tl_inside_job = false;

// How long a thread polls before blocking on a condition variable. The
// phases of the SIFT pipeline follow each other quickly; a blocked thread can
// take milliseconds to wake up, and an idle core first has to raise its clock
// again.
constexpr auto kSpinTime = std::chrono::milliseconds(10);

// Polls `pred` with a pause between checks for up to kSpinTime.
template <class Pred>
bool SpinUntil(Pred pred) {
  const auto deadline = std::chrono::steady_clock::now() + kSpinTime;
  for (;;) {
    for (int i = 0; i < 256; ++i) {
      if (pred()) return true;
      LSIFT_PAUSE();
    }
    if (std::chrono::steady_clock::now() > deadline) return pred();
  }
}
}  // namespace

// Protocol: the owner of a job publishes it by bumping `generation` with
// `open` set. A worker that sees a new generation registers in `active` and
// then checks that the job is still open before reading its fields. The owner
// waits until every item is `completed`, closes the job and waits for `active`
// to drop to zero; it never waits for workers that have not started, so a
// worker that wakes up late cannot delay the job.
struct ThreadPool::Impl {
  explicit Impl(unsigned num_threads) {
    for (unsigned i = 1; i < num_threads; ++i) workers.emplace_back([this] { worker_loop(); });
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stop.store(true);
      generation.fetch_add(1);
    }
    wake.notify_all();
    for (auto& t : workers) t.join();
  }

  void worker_loop() {
    std::uint64_t seen = 0;
    for (;;) {
      std::uint64_t gen = seen;
      SpinUntil([&] { return (gen = generation.load()) != seen; });
      if (gen == seen) {
        std::unique_lock<std::mutex> lock(mutex);
        wake.wait(lock, [&] { return generation.load() != seen; });
        gen = generation.load();
      }
      seen = gen;
      if (stop.load()) return;

      active.fetch_add(1);
      if (open.load() && generation.load() == seen) run_ranges();
      if (active.fetch_sub(1) == 1) notify_done();
    }
  }

  void notify_done() {
    std::lock_guard<std::mutex> lock(mutex);
    done.notify_all();
  }

  // Takes ranges from the shared counter until none are left.
  void run_ranges() {
    const bool was_inside = tl_inside_job;
    tl_inside_job = true;
    for (;;) {
      const std::size_t begin = next.fetch_add(grain, std::memory_order_relaxed);
      if (begin >= count) break;
      const std::size_t end = std::min(count, begin + grain);
      if (!failed.load(std::memory_order_relaxed)) {
        try {
          (*fn)(begin, end);
        } catch (...) {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (!error) error = std::current_exception();
          failed.store(true, std::memory_order_relaxed);
        }
      }
      if (completed.fetch_add(end - begin) + (end - begin) == count) notify_done();
    }
    tl_inside_job = was_inside;
  }

  template <class Pred>
  void wait_for(Pred pred) {
    if (!SpinUntil(pred)) {
      std::unique_lock<std::mutex> lock(mutex);
      done.wait(lock, pred);
    }
  }

  void parallel_for(std::size_t n, std::size_t g, const RangeFn& f) {
    if (n == 0) return;
    g = std::max<std::size_t>(g, 1);
    if (workers.empty() || n <= g || tl_inside_job) {
      f(0, n);
      return;
    }
    std::unique_lock<std::mutex> job(job_mutex, std::try_to_lock);
    if (!job.owns_lock()) {  // another thread is using the pool
      f(0, n);
      return;
    }

    fn = &f;
    count = n;
    grain = g;
    next.store(0);
    completed.store(0);
    failed.store(false);
    error = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex);
      open.store(true);
      generation.fetch_add(1);
    }
    wake.notify_all();

    run_ranges();
    wait_for([&] { return completed.load() == count; });
    open.store(false);
    wait_for([&] { return active.load() == 0; });

    fn = nullptr;
    if (error) std::rethrow_exception(error);
  }

  std::vector<std::thread> workers;
  std::mutex job_mutex;  // held by the thread that owns the current job
  std::mutex mutex;      // protects sleeping and waking
  std::condition_variable wake;
  std::condition_variable done;
  std::atomic<std::uint64_t> generation{0};
  std::atomic<bool> stop{false};
  std::atomic<bool> open{false};

  // The current job, written by its owner before `generation` is bumped.
  const RangeFn* fn = nullptr;
  std::size_t count = 0;
  std::size_t grain = 1;
  alignas(64) std::atomic<std::size_t> next{0};
  alignas(64) std::atomic<std::size_t> completed{0};
  alignas(64) std::atomic<int> active{0};
  std::atomic<bool> failed{false};
  std::mutex error_mutex;
  std::exception_ptr error;
};

ThreadPool::ThreadPool(unsigned num_threads) {
  if (num_threads == 0) num_threads = std::max(1u, std::thread::hardware_concurrency());
  impl_ = std::make_unique<Impl>(num_threads);
}

ThreadPool::~ThreadPool() = default;

void ThreadPool::parallel_for(std::size_t n, std::size_t grain, const RangeFn& fn) {
  impl_->parallel_for(n, grain, fn);
}

unsigned ThreadPool::concurrency() const noexcept {
  return static_cast<unsigned>(impl_->workers.size()) + 1;
}

Executor& default_executor() {
  static ThreadPool pool;
  return pool;
}

}  // namespace lsift
