//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "load_observer.h"

#include "test_util/testharness.h"
#include "test_util/testutil.h"

namespace ROCKSDB_NAMESPACE {

class LoadObserverTest : public testing::Test {
 public:
  LoadObserverTest() : env_(Env::Default()) {}

 protected:
  Env* env_;
};

TEST_F(LoadObserverTest, BasicInitialization) {
  LoadObserverOptions options;
  options.sampling_window_ms = 50;
  options.pmem_latency_weight = 0.6;
  options.cpu_utilization_weight = 0.4;
  
  LoadObserver observer(options, env_);
  
  // Initial stress should be 0
  ASSERT_EQ(observer.GetStressFactor(), 0.0);
  ASSERT_EQ(observer.GetNormalizedPMemLatency(), 0.0);
  ASSERT_EQ(observer.GetCPUUtilization(), 0.0);
}

TEST_F(LoadObserverTest, PMemLatencyRecording) {
  LoadObserverOptions options;
  options.baseline_pmem_latency_ns = 300;
  options.max_pmem_latency_ns = 3000;
  options.sampling_window_ms = 100;
  
  LoadObserver observer(options, env_);
  observer.Start();
  
  // Record some latencies
  observer.RecordPMemLatency(500);   // Low latency
  observer.RecordPMemLatency(600);
  observer.RecordPMemLatency(700);
  
  // Wait for sampling window
  env_->SleepForMicroseconds(150 * 1000);
  
  double normalized = observer.GetNormalizedPMemLatency();
  
  // Should be normalized: avg = 600ns
  // (600 - 300) / (3000 - 300) = 300/2700 ≈ 0.111
  ASSERT_GT(normalized, 0.0);
  ASSERT_LT(normalized, 0.3);  // Should be relatively low stress
  
  observer.Stop();
}

TEST_F(LoadObserverTest, HighPMemLatency) {
  LoadObserverOptions options;
  options.baseline_pmem_latency_ns = 300;
  options.max_pmem_latency_ns = 3000;
  options.sampling_window_ms = 100;
  options.pmem_latency_weight = 1.0;
  options.cpu_utilization_weight = 0.0;
  
  LoadObserver observer(options, env_);
  observer.Start();
  
  // Record high latencies
  for (int i = 0; i < 10; i++) {
    observer.RecordPMemLatency(2800);  // Near max
  }
  
  // Wait for sampling
  env_->SleepForMicroseconds(150 * 1000);
  
  double stress = observer.GetStressFactor();
  
  // Should indicate high stress
  ASSERT_GT(stress, 0.8);
  
  observer.Stop();
}

TEST_F(LoadObserverTest, CPUUtilizationTracking) {
  LoadObserverOptions options;
  options.sampling_window_ms = 100;
  options.pmem_latency_weight = 0.0;
  options.cpu_utilization_weight = 1.0;
  
  LoadObserver observer(options, env_);
  observer.Start();
  
  // Simulate 50ms of CPU time in a 100ms window (50% utilization)
  observer.RecordCompactionCPUTime(50 * 1000);
  
  // Wait for sampling
  env_->SleepForMicroseconds(150 * 1000);
  
  double cpu_util = observer.GetCPUUtilization();
  
  // Should be around 0.5 (50% utilization)
  ASSERT_GT(cpu_util, 0.3);
  ASSERT_LT(cpu_util, 0.7);
  
  observer.Stop();
}

TEST_F(LoadObserverTest, CombinedStressCalculation) {
  LoadObserverOptions options;
  options.baseline_pmem_latency_ns = 300;
  options.max_pmem_latency_ns = 3000;
  options.sampling_window_ms = 100;
  options.pmem_latency_weight = 0.6;
  options.cpu_utilization_weight = 0.4;
  
  LoadObserver observer(options, env_);
  observer.Start();
  
  // Simulate moderate PMem latency: 1500ns
  // (1500 - 300) / (3000 - 300) = 1200/2700 ≈ 0.44
  for (int i = 0; i < 5; i++) {
    observer.RecordPMemLatency(1500);
  }
  
  // Simulate 60% CPU utilization
  observer.RecordCompactionCPUTime(60 * 1000);
  
  // Wait for sampling
  env_->SleepForMicroseconds(150 * 1000);
  
  double stress = observer.GetStressFactor();
  
  // Expected: σ(t) = 0.6 * 0.44 + 0.4 * 0.6 = 0.264 + 0.24 = 0.504
  ASSERT_GT(stress, 0.4);
  ASSERT_LT(stress, 0.6);
  
  observer.Stop();
}

TEST_F(LoadObserverTest, SampleHistory) {
  LoadObserverOptions options;
  options.sampling_window_ms = 50;
  options.sample_history_size = 5;
  
  LoadObserver observer(options, env_);
  observer.Start();
  
  // Generate some samples
  for (int i = 0; i < 10; i++) {
    observer.RecordPMemLatency(500 + i * 100);
    env_->SleepForMicroseconds(60 * 1000);
  }
  
  std::vector<StressSample> samples;
  observer.GetRecentSamples(&samples);
  
  // Should have up to sample_history_size samples
  ASSERT_LE(samples.size(), 5);
  
  // Samples should be in chronological order
  for (size_t i = 1; i < samples.size(); i++) {
    ASSERT_GE(samples[i].timestamp_us, samples[i-1].timestamp_us);
  }
  
  observer.Stop();
}

TEST_F(LoadObserverTest, RAIIHelpers) {
  LoadObserverOptions options;
  LoadObserver observer(options, env_);
  observer.Start();
  
  // Test CompactionCPUTimeRecorder
  {
    CompactionCPUTimeRecorder recorder(&observer);
    env_->SleepForMicroseconds(10 * 1000);  // Sleep 10ms
    // Destructor should record ~10ms of CPU time
  }
  
  // Test PMemLatencyRecorder
  {
    PMemLatencyRecorder recorder(&observer);
    env_->SleepForMicroseconds(1000);  // Sleep 1ms
    // Destructor should record ~1ms latency
  }
  
  env_->SleepForMicroseconds(150 * 1000);
  
  // CPU utilization should be non-zero
  ASSERT_GT(observer.GetCPUUtilization(), 0.0);
  
  observer.Stop();
}

TEST_F(LoadObserverTest, StressFactorBounds) {
  LoadObserverOptions options;
  options.sampling_window_ms = 50;
  
  LoadObserver observer(options, env_);
  observer.Start();
  
  // Record extreme values
  for (int i = 0; i < 100; i++) {
    observer.RecordPMemLatency(100000);  // Way above max
    observer.RecordCompactionCPUTime(1000000);  // Way above window
  }
  
  env_->SleepForMicroseconds(100 * 1000);
  
  double stress = observer.GetStressFactor();
  
  // Stress should be clamped to [0, 1]
  ASSERT_GE(stress, 0.0);
  ASSERT_LE(stress, 1.0);
  
  observer.Stop();
}

TEST_F(LoadObserverTest, ZeroLatencyHandling) {
  LoadObserverOptions options;
  options.baseline_pmem_latency_ns = 300;
  options.max_pmem_latency_ns = 3000;
  
  LoadObserver observer(options, env_);
  observer.Start();
  
  // Record latencies below baseline
  observer.RecordPMemLatency(100);
  observer.RecordPMemLatency(200);
  
  env_->SleepForMicroseconds(150 * 1000);
  
  double normalized = observer.GetNormalizedPMemLatency();
  
  // Should be 0 since below baseline
  ASSERT_EQ(normalized, 0.0);
  
  observer.Stop();
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
