//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Adaptive Compaction Scheduler - Load Observer Module
// Monitors system stress factors for adaptive compaction scheduling

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>

#include "port/port.h"
#include "rocksdb/env.h"
#include "util/autovector.h"

namespace ROCKSDB_NAMESPACE {

// Configuration for the Load Observer
struct LoadObserverOptions {
  // Sampling window duration in milliseconds
  uint64_t sampling_window_ms = 100;

  // Weight for PMem latency in stress calculation (w1)
  double pmem_latency_weight = 0.6;

  // Weight for CPU utilization in stress calculation (w2)
  double cpu_utilization_weight = 0.4;

  // Baseline PMem persist latency in nanoseconds (for normalization)
  // This should be calibrated during initialization
  uint64_t baseline_pmem_latency_ns = 300;

  // Maximum expected PMem latency for normalization
  uint64_t max_pmem_latency_ns = 3000;

  // Number of samples to keep for moving average
  size_t sample_history_size = 10;

  // Enable detailed logging
  bool enable_logging = false;

  void Validate() const {
    assert(pmem_latency_weight + cpu_utilization_weight == 1.0);
    assert(pmem_latency_weight >= 0.0 && pmem_latency_weight <= 1.0);
    assert(sampling_window_ms > 0);
    assert(sample_history_size > 0);
  }
};

// Represents a single observation sample
struct StressSample {
  uint64_t timestamp_us;
  double pmem_latency_normalized;  // [0, 1]
  double cpu_utilization;          // [0, 1]
  double stress_factor;            // σ(t) ∈ [0, 1]

  StressSample()
      : timestamp_us(0),
        pmem_latency_normalized(0.0),
        cpu_utilization(0.0),
        stress_factor(0.0) {}
};

// Main Load Observer class
// Thread-safe observer that tracks system stress for adaptive compaction
class LoadObserver {
 public:
  explicit LoadObserver(const LoadObserverOptions& options, Env* env);
  ~LoadObserver();

  // Start the background monitoring thread
  void Start();

  // Stop the background monitoring thread
  void Stop();

  // Get current stress factor σ(t) ∈ [0, 1]
  double GetStressFactor() const {
    return current_stress_.load(std::memory_order_acquire);
  }

  // Get detailed current sample
  StressSample GetCurrentSample() const;

  // Record a PMem persist operation latency (in nanoseconds)
  // This should be called after each CLWB + SFENCE sequence
  void RecordPMemLatency(uint64_t latency_ns);

  // Record CPU time spent on compaction work
  // Called by compaction threads to track utilization
  void RecordCompactionCPUTime(uint64_t cpu_time_us);

  // Get normalized PMem latency [0, 1]
  double GetNormalizedPMemLatency() const {
    return pmem_latency_normalized_.load(std::memory_order_acquire);
  }

  // Get CPU utilization [0, 1]
  double GetCPUUtilization() const {
    return cpu_utilization_.load(std::memory_order_acquire);
  }

  // Calibrate baseline latencies (call during DB initialization)
  void Calibrate();

  // Get historical samples for analysis
  void GetRecentSamples(std::vector<StressSample>* samples) const;

 private:
  // Background thread function
  static void BGThreadWrapper(void* arg);
  void BGThread();

  // Update stress calculation
  void UpdateStressFactor();

  // Calculate normalized PMem latency
  double CalculateNormalizedPMemLatency();

  // Calculate CPU utilization
  double CalculateCPUUtilization();

  // Helper: Add sample to circular buffer
  void AddSampleToHistory(const StressSample& sample);

  // Configuration
  LoadObserverOptions options_;
  Env* env_;

  // Current stress factor σ(t)
  std::atomic<double> current_stress_;

  // Component metrics (normalized to [0, 1])
  std::atomic<double> pmem_latency_normalized_;
  std::atomic<double> cpu_utilization_;

  // PMem latency tracking
  mutable port::Mutex pmem_mutex_;
  std::vector<uint64_t> pmem_latency_samples_;  // Recent latency samples in ns
  uint64_t pmem_sample_count_;
  uint64_t pmem_sample_sum_;

  // CPU utilization tracking
  mutable port::Mutex cpu_mutex_;
  uint64_t total_cpu_time_us_;         // Total compaction CPU time
  uint64_t last_measurement_time_us_;  // Last time we measured
  uint64_t measurement_window_us_;     // Window for CPU calculation

  // Sample history (circular buffer)
  mutable port::Mutex history_mutex_;
  std::vector<StressSample> sample_history_;
  size_t history_index_;

  // Background thread control
  std::atomic<bool> running_;
  port::Thread bg_thread_;

  // Statistics
  std::atomic<uint64_t> total_samples_;
  std::atomic<uint64_t> high_stress_count_;  // Count of samples with σ(t) > 0.7
};

// RAII helper to record compaction CPU time
class CompactionCPUTimeRecorder {
 public:
  explicit CompactionCPUTimeRecorder(LoadObserver* observer)
      : observer_(observer), start_time_(std::chrono::steady_clock::now()) {}

  ~CompactionCPUTimeRecorder() {
    if (observer_) {
      auto end_time = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                          end_time - start_time_)
                          .count();
      observer_->RecordCompactionCPUTime(static_cast<uint64_t>(duration));
    }
  }

  // Non-copyable
  CompactionCPUTimeRecorder(const CompactionCPUTimeRecorder&) = delete;
  CompactionCPUTimeRecorder& operator=(const CompactionCPUTimeRecorder&) =
      delete;

 private:
  LoadObserver* observer_;
  std::chrono::steady_clock::time_point start_time_;
};

// RAII helper to record PMem persist latency
class PMemLatencyRecorder {
 public:
  explicit PMemLatencyRecorder(LoadObserver* observer)
      : observer_(observer), start_time_(std::chrono::steady_clock::now()) {}

  ~PMemLatencyRecorder() {
    if (observer_) {
      auto end_time = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          end_time - start_time_)
                          .count();
      observer_->RecordPMemLatency(static_cast<uint64_t>(duration));
    }
  }

  // Non-copyable
  PMemLatencyRecorder(const PMemLatencyRecorder&) = delete;
  PMemLatencyRecorder& operator=(const PMemLatencyRecorder&) = delete;

 private:
  LoadObserver* observer_;
  std::chrono::steady_clock::time_point start_time_;
};

}  // namespace ROCKSDB_NAMESPACE
