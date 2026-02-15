//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "load_observer.h"

#include <algorithm>
#include <cmath>

#include "monitoring/perf_context_imp.h"
#include "port/port.h"
#include "test_util/sync_point.h"
#include "util/cast_util.h"

namespace ROCKSDB_NAMESPACE {

LoadObserver::LoadObserver(const LoadObserverOptions& options, Env* env)
    : options_(options),
      env_(env),
      current_stress_(0.0),
      pmem_latency_normalized_(0.0),
      cpu_utilization_(0.0),
      pmem_sample_count_(0),
      pmem_sample_sum_(0),
      total_cpu_time_us_(0),
      last_measurement_time_us_(0),
      measurement_window_us_(options.sampling_window_ms * 1000),
      history_index_(0),
      running_(false),
      total_samples_(0),
      high_stress_count_(0) {
  
  options_.Validate();
  
  // Pre-allocate sample history
  sample_history_.resize(options_.sample_history_size);
  
  // Reserve space for PMem latency samples
  pmem_latency_samples_.reserve(1000);
  
  // Initialize timestamps
  last_measurement_time_us_ = env_->NowMicros();
}

LoadObserver::~LoadObserver() {
  Stop();
}

void LoadObserver::Start() {
  bool expected = false;
  if (running_.compare_exchange_strong(expected, true)) {
    // Start background thread
    env_->StartThread(&LoadObserver::BGThreadWrapper, &bg_thread_);
    
    if (options_.enable_logging) {
      fprintf(stderr, "[LoadObserver] Started monitoring thread\n");
    }
  }
}

void LoadObserver::Stop() {
  bool expected = true;
  if (running_.compare_exchange_strong(expected, false)) {
    // Wait for background thread to finish
    if (bg_thread_.joinable()) {
      bg_thread_.join();
    }
    
    if (options_.enable_logging) {
      fprintf(stderr, 
              "[LoadObserver] Stopped. Total samples: %lu, High stress events: %lu\n",
              total_samples_.load(), high_stress_count_.load());
    }
  }
}

void LoadObserver::BGThreadWrapper(void* arg) {
  LoadObserver* observer = reinterpret_cast<LoadObserver*>(arg);
  observer->BGThread();
}

void LoadObserver::BGThread() {
  while (running_.load(std::memory_order_acquire)) {
    // Update stress calculation
    UpdateStressFactor();
    
    // Sleep for the sampling window
    env_->SleepForMicroseconds(options_.sampling_window_ms * 1000);
    
    TEST_SYNC_POINT("LoadObserver::BGThread::AfterSleep");
  }
}

void LoadObserver::UpdateStressFactor() {
  // Calculate normalized PMem latency L_pmem ∈ [0, 1]
  double pmem_latency_norm = CalculateNormalizedPMemLatency();
  pmem_latency_normalized_.store(pmem_latency_norm, std::memory_order_release);
  
  // Calculate CPU utilization U_cpu ∈ [0, 1]
  double cpu_util = CalculateCPUUtilization();
  cpu_utilization_.store(cpu_util, std::memory_order_release);
  
  // Calculate stress factor: σ(t) = w1 * L_pmem + w2 * U_cpu
  double stress = (options_.pmem_latency_weight * pmem_latency_norm) +
                  (options_.cpu_utilization_weight * cpu_util);
  
  // Clamp to [0, 1]
  stress = std::min(1.0, std::max(0.0, stress));
  
  current_stress_.store(stress, std::memory_order_release);
  
  // Record sample in history
  StressSample sample;
  sample.timestamp_us = env_->NowMicros();
  sample.pmem_latency_normalized = pmem_latency_norm;
  sample.cpu_utilization = cpu_util;
  sample.stress_factor = stress;
  
  AddSampleToHistory(sample);
  
  // Update statistics
  total_samples_.fetch_add(1, std::memory_order_relaxed);
  if (stress > 0.7) {
    high_stress_count_.fetch_add(1, std::memory_order_relaxed);
  }
  
  if (options_.enable_logging) {
    fprintf(stderr, 
            "[LoadObserver] σ(t)=%.3f (PMem=%.3f, CPU=%.3f)\n",
            stress, pmem_latency_norm, cpu_util);
  }
}

double LoadObserver::CalculateNormalizedPMemLatency() {
  port::Mutex* mu = const_cast<port::Mutex*>(&pmem_mutex_);
  mu->Lock();
  
  if (pmem_sample_count_ == 0) {
    mu->Unlock();
    return 0.0;
  }
  
  // Calculate average latency from recent samples
  uint64_t avg_latency = pmem_sample_sum_ / pmem_sample_count_;
  
  // Reset accumulators for next window
  uint64_t count = pmem_sample_count_;
  pmem_sample_count_ = 0;
  pmem_sample_sum_ = 0;
  pmem_latency_samples_.clear();
  
  mu->Unlock();
  
  // Normalize to [0, 1] using baseline and max expected latency
  // L_norm = (current - baseline) / (max - baseline)
  if (avg_latency <= options_.baseline_pmem_latency_ns) {
    return 0.0;
  }
  
  if (avg_latency >= options_.max_pmem_latency_ns) {
    return 1.0;
  }
  
  double normalized = static_cast<double>(avg_latency - options_.baseline_pmem_latency_ns) /
                      static_cast<double>(options_.max_pmem_latency_ns - 
                                        options_.baseline_pmem_latency_ns);
  
  return std::min(1.0, std::max(0.0, normalized));
}

double LoadObserver::CalculateCPUUtilization() {
  port::Mutex* mu = const_cast<port::Mutex*>(&cpu_mutex_);
  mu->Lock();
  
  uint64_t current_time = env_->NowMicros();
  uint64_t elapsed_time = current_time - last_measurement_time_us_;
  
  if (elapsed_time == 0) {
    mu->Unlock();
    return 0.0;
  }
  
  // CPU utilization = (total_cpu_time / elapsed_wall_time)
  // For multi-threaded compaction, this can exceed 1.0, so we normalize by available cores
  // For simplicity, we cap at 1.0 (100% utilization)
  double utilization = static_cast<double>(total_cpu_time_us_) / 
                       static_cast<double>(elapsed_time);
  
  // Reset for next measurement
  total_cpu_time_us_ = 0;
  last_measurement_time_us_ = current_time;
  
  mu->Unlock();
  
  return std::min(1.0, std::max(0.0, utilization));
}

void LoadObserver::RecordPMemLatency(uint64_t latency_ns) {
  pmem_mutex_.Lock();
  
  pmem_latency_samples_.push_back(latency_ns);
  pmem_sample_sum_ += latency_ns;
  pmem_sample_count_++;
  
  // Prevent unbounded growth (keep last 10000 samples max)
  if (pmem_latency_samples_.size() > 10000) {
    // Remove oldest half
    size_t remove_count = pmem_latency_samples_.size() / 2;
    for (size_t i = 0; i < remove_count; i++) {
      pmem_sample_sum_ -= pmem_latency_samples_[i];
      pmem_sample_count_--;
    }
    pmem_latency_samples_.erase(pmem_latency_samples_.begin(),
                                pmem_latency_samples_.begin() + remove_count);
  }
  
  pmem_mutex_.Unlock();
}

void LoadObserver::RecordCompactionCPUTime(uint64_t cpu_time_us) {
  cpu_mutex_.Lock();
  total_cpu_time_us_ += cpu_time_us;
  cpu_mutex_.Unlock();
}

void LoadObserver::Calibrate() {
  if (options_.enable_logging) {
    fprintf(stderr, "[LoadObserver] Starting calibration...\n");
  }
  
  // Perform calibration by measuring baseline PMem latency
  // This would involve doing some test persist operations
  // For now, we use the configured baseline
  
  // In a real implementation, you would:
  // 1. Allocate a small PMem region
  // 2. Perform multiple CLWB + SFENCE sequences
  // 3. Measure average latency
  // 4. Set as baseline
  
  if (options_.enable_logging) {
    fprintf(stderr, 
            "[LoadObserver] Calibration complete. Baseline: %lu ns, Max: %lu ns\n",
            options_.baseline_pmem_latency_ns,
            options_.max_pmem_latency_ns);
  }
}

StressSample LoadObserver::GetCurrentSample() const {
  StressSample sample;
  sample.timestamp_us = env_->NowMicros();
  sample.pmem_latency_normalized = pmem_latency_normalized_.load(std::memory_order_acquire);
  sample.cpu_utilization = cpu_utilization_.load(std::memory_order_acquire);
  sample.stress_factor = current_stress_.load(std::memory_order_acquire);
  return sample;
}

void LoadObserver::GetRecentSamples(std::vector<StressSample>* samples) const {
  samples->clear();
  
  history_mutex_.Lock();
  
  // Copy samples in chronological order
  size_t total_samples = std::min(sample_history_.size(), 
                                  static_cast<size_t>(total_samples_.load()));
  
  if (total_samples < options_.sample_history_size) {
    // Haven't filled the buffer yet
    samples->insert(samples->end(), 
                   sample_history_.begin(), 
                   sample_history_.begin() + total_samples);
  } else {
    // Buffer is full, need to handle wrap-around
    size_t start_idx = history_index_;
    for (size_t i = 0; i < options_.sample_history_size; i++) {
      size_t idx = (start_idx + i) % options_.sample_history_size;
      samples->push_back(sample_history_[idx]);
    }
  }
  
  history_mutex_.Unlock();
}

void LoadObserver::AddSampleToHistory(const StressSample& sample) {
  history_mutex_.Lock();
  
  sample_history_[history_index_] = sample;
  history_index_ = (history_index_ + 1) % options_.sample_history_size;
  
  history_mutex_.Unlock();
}

}  // namespace ROCKSDB_NAMESPACE
