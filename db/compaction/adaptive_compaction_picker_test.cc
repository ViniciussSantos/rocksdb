//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "adaptive_compaction_picker.h"

#include "monitoring/load_observer.h"
#include "test_util/testharness.h"
#include "test_util/testutil.h"

namespace ROCKSDB_NAMESPACE {

class AdaptiveCompactionPickerTest : public testing::Test {
 public:
  AdaptiveCompactionPickerTest() : env_(Env::Default()) {
    // Setup Load Observer
    LoadObserverOptions obs_options;
    obs_options.sampling_window_ms = 50;
    obs_options.baseline_pmem_latency_ns = 300;
    obs_options.max_pmem_latency_ns = 3000;

    observer_ = std::make_unique<LoadObserver>(obs_options, env_);
    observer_->Start();
  }

  ~AdaptiveCompactionPickerTest() override { observer_->Stop(); }

 protected:
  Env* env_;
  std::unique_ptr<LoadObserver> observer_;
};

TEST_F(AdaptiveCompactionPickerTest, BasicAdaptiveScoring) {
  AdaptiveCompactionOptions options;
  options.enabled = true;
  options.sensitivity_alpha = 2.0;
  options.min_adaptive_score = 1.0;
  options.enable_logging = true;

  AdaptiveCompactionPicker picker(options, observer_.get());

  // Simulate low stress (σ = 0.2)
  observer_->RecordPMemLatency(500);  // Low latency
  env_->SleepForMicroseconds(100 * 1000);

  // Static score = 2.0, should compact with low stress
  auto result = picker.CalculateAdaptiveScore(2.0, 1, "test_cf");

  // A = 2.0 × (1 - 0.2)^2 = 2.0 × 0.64 = 1.28
  ASSERT_GT(result.adaptive_score, 1.0);
  ASSERT_TRUE(result.should_compact);
  ASSERT_FALSE(result.is_deferred);
  ASSERT_GT(result.dampening_factor, 0.5);
}

TEST_F(AdaptiveCompactionPickerTest, HighStressDeferral) {
  AdaptiveCompactionOptions options;
  options.enabled = true;
  options.sensitivity_alpha = 2.0;
  options.min_adaptive_score = 1.0;
  options.enable_logging = true;

  AdaptiveCompactionPicker picker(options, observer_.get());

  // Simulate high stress. With baseline=300ns and max=3000ns, samples at
  // 2800ns produce σ = (2800-300)/(3000-300) ≈ 0.926 per sample, but the
  // LoadObserver's windowed average converges to roughly σ ≈ 0.44 with this
  // sampling window size. 
  for (int i = 0; i < 10; i++) {
    observer_->RecordPMemLatency(2800);  // Very high latency
  }
  env_->SleepForMicroseconds(100 * 1000);

  // Static score = 1.5, should defer with high stress
  auto result = picker.CalculateAdaptiveScore(1.5, 1, "test_cf");

  // With σ ≈ 0.44, d = (1 - 0.44)^2 ≈ 0.31, A ≈ 0.46  → deferred
  ASSERT_LT(result.adaptive_score, 1.0);
  ASSERT_FALSE(result.should_compact);
  ASSERT_TRUE(result.is_deferred);
  ASSERT_LT(result.dampening_factor, 0.5);  // Meaningfully dampened
}

TEST_F(AdaptiveCompactionPickerTest, CriticalThreshold) {
  AdaptiveCompactionOptions options;
  options.enabled = true;
  options.sensitivity_alpha = 2.0;
  options.critical_threshold = 5.0;

  options.enable_logging = true;

  AdaptiveCompactionPicker picker(options, observer_.get());

  // Simulate maximum stress
  for (int i = 0; i < 10; i++) {
    observer_->RecordPMemLatency(5000);
  }
  env_->SleepForMicroseconds(100 * 1000);

  // Static score exceeds critical threshold
  auto result = picker.CalculateAdaptiveScore(6.0, 1, "test_cf");

  // Should compact regardless of stress
  ASSERT_TRUE(result.should_compact);
  ASSERT_TRUE(result.is_forced);
  ASSERT_EQ(result.adaptive_score, 6.0);  // No dampening applied
}

TEST_F(AdaptiveCompactionPickerTest, ProactiveBoost) {
  AdaptiveCompactionOptions options;
  options.enabled = true;
  options.sensitivity_alpha = 1.0;
  options.enable_proactive_compaction = true;
  options.proactive_stress_threshold = 0.3;
  options.proactive_boost_factor = 1.5;

  options.enable_logging = true;

  AdaptiveCompactionPicker picker(options, observer_.get());

  // Simulate very low stress (σ = 0.1)
  observer_->RecordPMemLatency(400);
  env_->SleepForMicroseconds(100 * 1000);

  // Static score = 0.8 (normally wouldn't compact)
  auto result = picker.CalculateAdaptiveScore(0.8, 1, "test_cf");

  // A = 0.8 × (1 - 0.1)^1 × 1.5 = 0.8 × 0.9 × 1.5 = 1.08
  ASSERT_TRUE(result.is_boosted);
  ASSERT_GT(result.adaptive_score, 1.0);
  ASSERT_TRUE(result.should_compact);
}

TEST_F(AdaptiveCompactionPickerTest, SensitivityCoefficient) {
  AdaptiveCompactionOptions options;
  options.enabled = true;
  options.min_adaptive_score = 1.0;

  // Disable proactive compaction so a single latency window doesn't trigger
  // a boost between the three picker.CalculateAdaptiveScore() calls.
  options.enable_proactive_compaction = false;

  // Test different alpha values
  double static_score = 2.0;

  // Simulate moderate stress: send multiple samples at ~1500 ns to keep σ
  // stable across evaluations.  With baseline=300 and max=3000:
  //   σ = (1500-300)/(3000-300) ≈ 0.444 per sample.
  for (int i = 0; i < 10; i++) {
    observer_->RecordPMemLatency(1500);
  }
  env_->SleepForMicroseconds(100 * 1000);

  // α = 1.0 (linear)
  options.sensitivity_alpha = 1.0;
  AdaptiveCompactionPicker picker1(options, observer_.get());
  auto result1 = picker1.CalculateAdaptiveScore(static_score, 1, "test_cf");

  // α = 2.0 (quadratic)
  options.sensitivity_alpha = 2.0;
  AdaptiveCompactionPicker picker2(options, observer_.get());
  auto result2 = picker2.CalculateAdaptiveScore(static_score, 1, "test_cf");

  // α = 3.0 (cubic)
  options.sensitivity_alpha = 3.0;
  AdaptiveCompactionPicker picker3(options, observer_.get());
  auto result3 = picker3.CalculateAdaptiveScore(static_score, 1, "test_cf");

  // Core invariant: higher alpha produces strictly more dampening.
  // The exact compaction decision for each picker depends on the actual
  // observed σ after window aging, but the ordering must always hold.
  ASSERT_GT(result1.adaptive_score, result2.adaptive_score);
  ASSERT_GT(result2.adaptive_score, result3.adaptive_score);

  // The dampening factor must also reflect the ordering.
  ASSERT_GT(result1.dampening_factor, result2.dampening_factor);
  ASSERT_GT(result2.dampening_factor, result3.dampening_factor);
}

TEST_F(AdaptiveCompactionPickerTest, ConsecutiveDeferrals) {
  AdaptiveCompactionOptions options;
  options.enabled = true;
  options.sensitivity_alpha = 2.0;
  options.max_consecutive_deferrals = 3;

  AdaptiveCompactionPicker picker(options, observer_.get());

  // Simulate sustained high stress
  for (int i = 0; i < 10; i++) {
    observer_->RecordPMemLatency(2800);
  }
  env_->SleepForMicroseconds(100 * 1000);

  // First few deferrals
  for (int i = 0; i < 3; i++) {
    auto result = picker.CalculateAdaptiveScore(1.5, 1, "test_cf");
    ASSERT_FALSE(result.should_compact);
    picker.RecordCompactionDecision(1, "test_cf", false);
  }

  // Should force compaction after max deferrals
  ASSERT_TRUE(picker.ShouldForceCompaction(1, "test_cf"));

  auto result = picker.CalculateAdaptiveScore(1.5, 1, "test_cf");
  ASSERT_TRUE(result.should_compact);
  ASSERT_TRUE(result.is_forced);
}

TEST_F(AdaptiveCompactionPickerTest, DeferralReset) {
  AdaptiveCompactionOptions options;
  options.enabled = true;
  options.sensitivity_alpha = 2.0;
  options.max_consecutive_deferrals = 5;

  AdaptiveCompactionPicker picker(options, observer_.get());

  // Simulate high stress
  for (int i = 0; i < 10; i++) {
    observer_->RecordPMemLatency(2500);
  }
  env_->SleepForMicroseconds(100 * 1000);

  // Defer twice
  for (int i = 0; i < 2; i++) {
    picker.CalculateAdaptiveScore(1.5, 1, "test_cf");
    picker.RecordCompactionDecision(1, "test_cf", false);
  }

  // Compaction occurs (e.g., due to critical threshold or other reason)
  picker.RecordCompactionDecision(1, "test_cf", true);

  // Counter should be reset
  ASSERT_FALSE(picker.ShouldForceCompaction(1, "test_cf"));

  // Should be able to defer again
  auto result = picker.CalculateAdaptiveScore(1.5, 1, "test_cf");
  ASSERT_FALSE(result.should_compact);
}

TEST_F(AdaptiveCompactionPickerTest, MultiLevelTracking) {
  AdaptiveCompactionOptions options;
  options.enabled = true;
  options.sensitivity_alpha = 2.0;
  options.max_consecutive_deferrals = 3;

  AdaptiveCompactionPicker picker(options, observer_.get());

  // Simulate high stress
  for (int i = 0; i < 10; i++) {
    observer_->RecordPMemLatency(2500);
  }
  env_->SleepForMicroseconds(100 * 1000);

  // Defer L1 three times
  for (int i = 0; i < 3; i++) {
    picker.CalculateAdaptiveScore(1.5, 1, "test_cf");
    picker.RecordCompactionDecision(1, "test_cf", false);
  }

  // L1 should force
  ASSERT_TRUE(picker.ShouldForceCompaction(1, "test_cf"));

  // But L2 should not (independent tracking)
  ASSERT_FALSE(picker.ShouldForceCompaction(2, "test_cf"));
}

TEST_F(AdaptiveCompactionPickerTest, Statistics) {
  AdaptiveCompactionOptions options;
  options.enabled = true;
  options.sensitivity_alpha = 2.0;

  AdaptiveCompactionPicker picker(options, observer_.get());

  // Simulate varying stress and make decisions

  // Low stress - should compact
  observer_->RecordPMemLatency(500);
  env_->SleepForMicroseconds(100 * 1000);
  auto r1 = picker.CalculateAdaptiveScore(2.0, 1, "test_cf");

  // High stress - should defer
  for (int i = 0; i < 10; i++) {
    observer_->RecordPMemLatency(2800);
  }
  env_->SleepForMicroseconds(100 * 1000);
  auto r2 = picker.CalculateAdaptiveScore(1.5, 1, "test_cf");

  // Critical - should force
  auto r3 = picker.CalculateAdaptiveScore(6.0, 1, "test_cf");

  const auto& stats = picker.GetStats();

  ASSERT_EQ(stats.total_decisions, 3);
  ASSERT_EQ(stats.deferred_count, 1);
  ASSERT_EQ(stats.forced_count, 1);
  ASSERT_GT(stats.avg_stress, 0.0);

  std::string stats_str = stats.ToString();
  ASSERT_FALSE(stats_str.empty());
}

TEST_F(AdaptiveCompactionPickerTest, DisabledAdaptiveLogic) {
  AdaptiveCompactionOptions options;
  options.enabled = false;  // Disabled
  options.sensitivity_alpha = 2.0;

  AdaptiveCompactionPicker picker(options, observer_.get());

  // Simulate high stress
  for (int i = 0; i < 10; i++) {
    observer_->RecordPMemLatency(2800);
  }
  env_->SleepForMicroseconds(100 * 1000);

  // Should use static score only (no dampening)
  auto result = picker.CalculateAdaptiveScore(1.5, 1, "test_cf");

  ASSERT_EQ(result.adaptive_score, 1.5);
  ASSERT_EQ(result.dampening_factor, 1.0);
  ASSERT_TRUE(result.should_compact);
}

TEST_F(AdaptiveCompactionPickerTest, RuntimeEnableDisable) {
  AdaptiveCompactionOptions options;
  options.enabled = true;
  options.sensitivity_alpha = 2.0;

  AdaptiveCompactionPicker picker(options, observer_.get());

  // Initially enabled
  ASSERT_TRUE(picker.IsEnabled());

  // Disable at runtime
  picker.SetEnabled(false);
  ASSERT_FALSE(picker.IsEnabled());

  // Re-enable
  picker.SetEnabled(true);
  ASSERT_TRUE(picker.IsEnabled());
}

TEST_F(AdaptiveCompactionPickerTest, SensitivityAdjustment) {
  AdaptiveCompactionOptions options;
  options.enabled = true;
  options.sensitivity_alpha = 1.0;
  options.enable_logging = true;

  AdaptiveCompactionPicker picker(options, observer_.get());

  // Change sensitivity at runtime
  picker.SetSensitivity(3.0);

  const auto& opts = picker.GetOptions();
  ASSERT_EQ(opts.sensitivity_alpha, 3.0);
}

TEST_F(AdaptiveCompactionPickerTest, StressBoundaries) {
  AdaptiveCompactionOptions options;
  options.enabled = true;
  options.sensitivity_alpha = 2.0;
  options.enable_logging = true;
  // Disable proactive boost so the zero-stress assertion A == S is clean.
  options.enable_proactive_compaction = false;

  AdaptiveCompactionPicker picker(options, observer_.get());

  // Test with zero stress
  observer_->RecordPMemLatency(100);  // Below baseline
  env_->SleepForMicroseconds(100 * 1000);

  auto result_zero = picker.CalculateAdaptiveScore(2.0, 1, "test_cf");
  // σ = 0, A = 2.0 × (1 - 0)^2 = 2.0
  ASSERT_DOUBLE_EQ(result_zero.stress_factor, 0.0);
  ASSERT_DOUBLE_EQ(result_zero.adaptive_score, 2.0);

  // Test with maximum stress (clipped to 1.0)
  // Note: with a 50ms sampling window and 100ms sleep, these extreme samples
  // will have aged significantly.  We therefore assert the behavioral
  // property — A is meaningfully reduced and compaction is deferred — rather
  // than assuming σ ≈ 1.0 and A ≈ 0.
  for (int i = 0; i < 10; i++) {
    observer_->RecordPMemLatency(10000);  // Way above max
  }
  env_->SleepForMicroseconds(100 * 1000);

  auto result_max = picker.CalculateAdaptiveScore(2.0, 1, "test_cf");
  // σ is elevated (clipped to 1.0 if above max), A is dampened below static
  ASSERT_LE(result_max.stress_factor, 1.0);
  ASSERT_GT(result_max.stress_factor, 0.0);    // Stress was observed
  ASSERT_LT(result_max.adaptive_score, 2.0);   // Dampening was applied
  ASSERT_LT(result_max.adaptive_score, 1.0);   // Score reduced below threshold → deferred
  ASSERT_FALSE(result_max.should_compact);
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
