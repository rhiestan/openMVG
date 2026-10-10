// LeanSift - fast SIFT feature extraction
// SPDX-License-Identifier: BSD-2-Clause

#pragma once

#include <cstddef>
#include <functional>
#include <memory>

#include "lsift/export.hpp"

namespace lsift {

/// Runs data-parallel loops for LeanSift.
///
/// Implement this interface to run LeanSift on an application's own threads,
/// for example a TBB arena or an existing job system. This avoids creating more
/// threads than cores when the application already processes several images
/// at the same time.
class LSIFT_EXPORT Executor {
public:
  using RangeFn = std::function<void(std::size_t begin, std::size_t end)>;

  virtual ~Executor() = default;

  /// Calls `fn(begin, end)` on disjoint ranges that together cover [0, n).
  /// Ranges hold about `grain` items. Returns once every range has finished.
  /// `fn` may be called concurrently from several threads. An exception thrown
  /// by `fn` must be propagated to the caller.
  virtual void parallel_for(std::size_t n, std::size_t grain, const RangeFn& fn) = 0;

  /// Number of threads that can run ranges at the same time.
  virtual unsigned concurrency() const noexcept = 0;
};

/// Runs every range on the calling thread.
class LSIFT_EXPORT SerialExecutor final : public Executor {
public:
  void parallel_for(std::size_t n, std::size_t grain, const RangeFn& fn) override;
  unsigned concurrency() const noexcept override { return 1; }
};

/// A fixed set of std::thread workers. The thread that calls parallel_for
/// takes part in the work, so a pool of N threads starts N - 1 workers.
///
/// Calls to parallel_for from inside a running range, or while another thread
/// is using the pool, run serially on the calling thread instead of blocking.
class LSIFT_EXPORT ThreadPool final : public Executor {
public:
  /// `num_threads` = 0 uses std::thread::hardware_concurrency().
  explicit ThreadPool(unsigned num_threads = 0);
  ~ThreadPool() override;

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  void parallel_for(std::size_t n, std::size_t grain, const RangeFn& fn) override;
  unsigned concurrency() const noexcept override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Process-wide thread pool with one thread per hardware thread, created on
/// first use. Used when no executor is passed to lsift::Sift.
LSIFT_EXPORT Executor& default_executor();

}  // namespace lsift
