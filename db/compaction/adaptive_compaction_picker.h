//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Adaptive Compaction Scheduler - Adaptive Compaction Picker
// Implements adaptive scoring: A_i = S_i × (1 - σ(t))^α

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "db/compaction/compaction_picker.h"
#include "db/compaction/compaction_picker_level.h"
#include "db/compaction/compaction_picker_universal.h"
#include "monitoring/load_observer.h"
#include "rocksdb/options.h"

namespace ROCKSDB_NAMESPACE {

// Options specific to adaptive compaction scheduling
struct AdaptiveCompactionOptions {
  // Enable adaptive compaction scheduling
  bool enabled = false;

  // Sensitivity coefficient α (controls how aggressively we defer compactions)
  // α = 1.0: Linear dampening (gentle)
  // α = 2.0: Quadratic dampening (recommended)
  // α = 3.0: Cubic dampening (aggressive)
  double sensitivity_alpha = 2.0;

  // Threshold for forced compaction (prevent indefinite deferral)
  // If static score exceeds this, compact regardless of stress
  double critical_threshold = 5.0;

  // Minimum adaptive score to trigger compaction
  double min_adaptive_score = 1.0;

  // Enable proactive compaction during low stress
  // If σ(t) < proactive_threshold, boost compaction score
  bool enable_proactive_compaction = true;
  double proactive_stress_threshold = 0.3;
  double proactive_boost_factor = 1.2;

  // Maximum number of consecutive deferrals before forcing compaction
  // Prevents starvation of compaction during sustained high stress
  uint32_t max_consecutive_deferrals = 10;

  // Enable detailed logging
  bool enable_logging = false;

  void Validate() const {
    assert(sensitivity_alpha > 0.0);
    assert(critical_threshold > 0.0);
    assert(min_adaptive_score > 0.0);
    assert(proactive_stress_threshold >= 0.0 &&
           proactive_stress_threshold <= 1.0);
    assert(proactive_boost_factor >= 1.0);
    assert(max_consecutive_deferrals > 0);
  }
};

// Statistics for adaptive compaction decisions
struct AdaptiveCompactionStats {
  // Total number of compaction decisions made
  uint64_t total_decisions = 0;

  // Number of compactions that were deferred due to high stress
  uint64_t deferred_count = 0;

  // Number of compactions that were boosted due to low stress
  uint64_t boosted_count = 0;

  // Number of forced compactions (critical threshold exceeded)
  uint64_t forced_count = 0;

  // Average stress at decision time
  double avg_stress = 0.0;

  // Average adaptive score
  double avg_adaptive_score = 0.0;

  // Histogram: stress distribution at decision points
  std::vector<uint64_t>
      stress_histogram;  // Buckets: [0-0.1, 0.1-0.2, ..., 0.9-1.0]

  AdaptiveCompactionStats() { stress_histogram.resize(10, 0); }

  void RecordDecision(double stress, double adaptive_score, bool deferred,
                      bool boosted, bool forced) {
    total_decisions++;
    if (deferred) deferred_count++;
    if (boosted) boosted_count++;
    if (forced) forced_count++;

    // Update running average for stress
    avg_stress =
        (avg_stress * (total_decisions - 1) + stress) / total_decisions;
    avg_adaptive_score =
        (avg_adaptive_score * (total_decisions - 1) + adaptive_score) /
        total_decisions;

    // Update histogram
    int bucket = std::min(9, static_cast<int>(stress * 10));
    stress_histogram[bucket]++;
  }

  void Reset() {
    total_decisions = 0;
    deferred_count = 0;
    boosted_count = 0;
    forced_count = 0;
    avg_stress = 0.0;
    avg_adaptive_score = 0.0;
    std::fill(stress_histogram.begin(), stress_histogram.end(), 0);
  }

  std::string ToString() const;
};

// Base class for adaptive compaction picking
// This extends the existing CompactionPicker with adaptive logic
class AdaptiveCompactionPicker {
 public:
  explicit AdaptiveCompactionPicker(const AdaptiveCompactionOptions& options,
                                    LoadObserver* load_observer);

  virtual ~AdaptiveCompactionPicker() = default;

  // Calculate adaptive score: A_i = S_i × (1 - σ(t))^α
  // Returns the adaptive score and whether compaction should proceed
  struct AdaptiveScoreResult {
    double static_score;
    double stress_factor;
    double dampening_factor;
    double adaptive_score;
    bool should_compact;
    bool is_forced;      // Critical threshold exceeded
    bool is_boosted;     // Proactive boost applied
    bool is_deferred;    // Deferred due to high stress
    std::string reason;  // Human-readable explanation
  };

  AdaptiveScoreResult CalculateAdaptiveScore(double static_score, int level,
                                             const std::string& cf_name);

  // Check if we should defer compaction based on consecutive deferrals
  bool ShouldForceCompaction(int level, const std::string& cf_name);

  // Update deferral tracking
  void RecordCompactionDecision(int level, const std::string& cf_name,
                                bool compaction_occurred);

  // Get current statistics
  const AdaptiveCompactionStats& GetStats() const { return stats_; }
  void ResetStats() { stats_.Reset(); }

  // Get/Set options
  const AdaptiveCompactionOptions& GetOptions() const { return options_; }
  void SetSensitivity(double alpha) {
    options_.sensitivity_alpha = alpha;
    if (options_.enable_logging) {
      fprintf(stderr, "[AdaptiveCompaction] Sensitivity α set to %.2f\n",
              alpha);
    }
  }

  // Enable/disable adaptive logic
  void SetEnabled(bool enabled) {
    options_.enabled = enabled;
    if (options_.enable_logging) {
      fprintf(stderr, "[AdaptiveCompaction] Adaptive logic %s\n",
              enabled ? "enabled" : "disabled");
    }
  }

  bool IsEnabled() const { return options_.enabled; }

  // Get current stress factor
  double GetCurrentStress() const {
    return load_observer_ ? load_observer_->GetStressFactor() : 0.0;
  }

 protected:
  // Calculate dampening factor: (1 - σ(t))^α
  double CalculateDampeningFactor(double stress) const;

  // Apply proactive boost if stress is low
  double ApplyProactiveBoost(double adaptive_score, double stress) const;

  // Check if static score exceeds critical threshold
  bool IsCriticalCompaction(double static_score) const;

 private:
  // Configuration
  AdaptiveCompactionOptions options_;

  // Reference to Load Observer
  LoadObserver* load_observer_;

  // Statistics
  AdaptiveCompactionStats stats_;

  // Track consecutive deferrals per level per CF
  // Key: "cf_name:level", Value: consecutive deferral count
  std::unordered_map<std::string, uint32_t> deferral_tracker_;

  // Mutex for thread safety
  mutable port::Mutex mutex_;

  // Helper to generate tracker key
  std::string MakeTrackerKey(const std::string& cf_name, int level) const {
    return cf_name + ":" + std::to_string(level);
  }
};

// Adaptive version of LevelCompactionPicker
// This is what will be used in RocksDB's compaction subsystem
class AdaptiveLevelCompactionPicker : public LevelCompactionPicker {
 public:
  AdaptiveLevelCompactionPicker(
      const ImmutableOptions& ioptions, const InternalKeyComparator* icmp,
      const AdaptiveCompactionOptions& adaptive_options,
      LoadObserver* load_observer)
      : LevelCompactionPicker(ioptions, icmp),
        adaptive_picker_(adaptive_options, load_observer) {}

  // Override PickCompaction to use adaptive scoring
  Compaction* PickCompaction(
      const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
      const MutableDBOptions& mutable_db_options,
      const std::vector<SequenceNumber>& /* existing_snapshots */,
      const SnapshotChecker* /* snapshot_checker */,
      VersionStorageInfo* vstorage, LogBuffer* log_buffer,
      const std::string& full_history_ts_low,
      bool /*require_max_output_level*/ = false) override;

  // Access to adaptive functionality
  AdaptiveCompactionPicker* GetAdaptivePicker() { return &adaptive_picker_; }
  const AdaptiveCompactionPicker* GetAdaptivePicker() const {
    return &adaptive_picker_;
  }

  // Enable/disable adaptive logic at runtime
  void SetAdaptiveEnabled(bool enabled) {
    adaptive_picker_.SetEnabled(enabled);
  }

  bool IsAdaptiveEnabled() const { return adaptive_picker_.IsEnabled(); }

 private:
  AdaptiveCompactionPicker adaptive_picker_;

  // Helper: Determine if we should pick compaction based on adaptive score
  bool ShouldPickCompactionAdaptive(double static_score, int level,
                                    const std::string& cf_name,
                                    LogBuffer* log_buffer);
};

// Adaptive version of UniversalCompactionPicker
// For completeness, though leveled is primary focus
class AdaptiveUniversalCompactionPicker : public UniversalCompactionPicker {
 public:
  AdaptiveUniversalCompactionPicker(
      const ImmutableOptions& ioptions, const InternalKeyComparator* icmp,
      const AdaptiveCompactionOptions& adaptive_options,
      LoadObserver* load_observer)
      : UniversalCompactionPicker(ioptions, icmp),
        adaptive_picker_(adaptive_options, load_observer) {}

  Compaction* PickCompaction(
      const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
      const MutableDBOptions& mutable_db_options,
      const std::vector<SequenceNumber>& /* existing_snapshots */,
      const SnapshotChecker* /* snapshot_checker */,
      VersionStorageInfo* vstorage, LogBuffer* log_buffer,
      const std::string& full_history_ts_low,
      bool /*require_max_output_level*/ = false) override;

  AdaptiveCompactionPicker* GetAdaptivePicker() { return &adaptive_picker_; }
  const AdaptiveCompactionPicker* GetAdaptivePicker() const {
    return &adaptive_picker_;
  }

 private:
  AdaptiveCompactionPicker adaptive_picker_;
};

// Factory function to create appropriate adaptive compaction picker
std::unique_ptr<CompactionPicker> NewAdaptiveCompactionPicker(
    const ImmutableOptions& ioptions, const InternalKeyComparator* icmp,
    const AdaptiveCompactionOptions& adaptive_options,
    LoadObserver* load_observer);

}  // namespace ROCKSDB_NAMESPACE
