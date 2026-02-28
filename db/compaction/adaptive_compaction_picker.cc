//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "adaptive_compaction_picker.h"

#include <cmath>
#include <sstream>

#include "db/column_family.h"
#include "db/compaction/compaction_picker_level.h"
#include "logging/log_buffer.h"
#include "logging/logging.h"
#include "util/string_util.h"

namespace ROCKSDB_NAMESPACE {

//==============================================================================
// AdaptiveCompactionStats
//==============================================================================

std::string AdaptiveCompactionStats::ToString() const {
  std::ostringstream oss;
  oss << "AdaptiveCompactionStats:\n"
      << "  Total decisions: " << total_decisions << "\n"
      << "  Deferred: " << deferred_count << " ("
      << (total_decisions > 0 ? (100.0 * deferred_count / total_decisions)
                              : 0.0)
      << "%)\n"
      << "  Boosted: " << boosted_count << " ("
      << (total_decisions > 0 ? (100.0 * boosted_count / total_decisions) : 0.0)
      << "%)\n"
      << "  Forced: " << forced_count << " ("
      << (total_decisions > 0 ? (100.0 * forced_count / total_decisions) : 0.0)
      << "%)\n"
      << "  Avg stress: " << avg_stress << "\n"
      << "  Avg adaptive score: " << avg_adaptive_score << "\n"
      << "  Stress distribution:\n";

  for (size_t i = 0; i < stress_histogram.size(); i++) {
    double bucket_start = i * 0.1;
    double bucket_end = (i + 1) * 0.1;
    oss << "    [" << bucket_start << "-" << bucket_end
        << "): " << stress_histogram[i] << "\n";
  }

  return oss.str();
}

//==============================================================================
// AdaptiveCompactionPicker
//==============================================================================

AdaptiveCompactionPicker::AdaptiveCompactionPicker(
    const AdaptiveCompactionOptions& options, LoadObserver* load_observer)
    : options_(options), load_observer_(load_observer) {
  options_.Validate();

  if (options_.enable_logging) {
    fprintf(stderr,
            "[AdaptiveCompaction] Initialized with α=%.2f, enabled=%d\n",
            options_.sensitivity_alpha, options_.enabled);
  }
}

double AdaptiveCompactionPicker::CalculateDampeningFactor(double stress) const {
  // Dampening factor = (1 - σ(t))^α
  // When stress is high (σ→1), dampening approaches 0
  // When stress is low (σ→0), dampening approaches 1

  double base = std::max(0.0, std::min(1.0, 1.0 - stress));
  double dampening = std::pow(base, options_.sensitivity_alpha);

  return dampening;
}

double AdaptiveCompactionPicker::ApplyProactiveBoost(double adaptive_score,
                                                     double stress) const {
  if (!options_.enable_proactive_compaction) {
    return adaptive_score;
  }

  // If stress is low, boost compaction to proactively reduce debt
  if (stress < options_.proactive_stress_threshold) {
    return adaptive_score * options_.proactive_boost_factor;
  }

  return adaptive_score;
}

bool AdaptiveCompactionPicker::IsCriticalCompaction(double static_score) const {
  return static_score >= options_.critical_threshold;
}

AdaptiveCompactionPicker::AdaptiveScoreResult
AdaptiveCompactionPicker::CalculateAdaptiveScore(double static_score, int level,
                                                 const std::string& cf_name) {
  AdaptiveScoreResult result;
  result.static_score = static_score;
  result.stress_factor =
      load_observer_ ? load_observer_->GetStressFactor() : 0.0;
  result.is_forced = false;
  result.is_boosted = false;
  result.is_deferred = false;

  // If adaptive logic is disabled, just use static score
  if (!options_.enabled) {
    result.dampening_factor = 1.0;
    result.adaptive_score = static_score;
    result.should_compact = (static_score >= options_.min_adaptive_score);
    result.reason = "Adaptive logic disabled";
    return result;
  }

  // Check if this is a critical compaction that must proceed
  if (IsCriticalCompaction(static_score)) {
    result.dampening_factor = 1.0;
    result.adaptive_score = static_score;
    result.should_compact = true;
    result.is_forced = true;
    result.reason = "Critical threshold exceeded";

    // Reset deferral counter since we're compacting
    std::string key = MakeTrackerKey(cf_name, level);
    mutex_.Lock();
    deferral_tracker_[key] = 0;
    stats_.RecordDecision(result.stress_factor, result.adaptive_score, false,
                          false, true);
    mutex_.Unlock();

    if (options_.enable_logging) {
      fprintf(
          stderr,
          "[AdaptiveCompaction] FORCED L%d (CF=%s): S=%.2f > threshold=%.2f\n",
          level, cf_name.c_str(), static_score, options_.critical_threshold);
    }

    return result;
  }

  // Check if we should force due to consecutive deferrals
  if (ShouldForceCompaction(level, cf_name)) {
    result.dampening_factor = 1.0;
    result.adaptive_score = static_score;
    result.should_compact = true;
    result.is_forced = true;
    result.reason = "Max consecutive deferrals reached";

    // Reset deferral counter
    std::string key = MakeTrackerKey(cf_name, level);
    mutex_.Lock();
    deferral_tracker_[key] = 0;
    stats_.RecordDecision(result.stress_factor, result.adaptive_score, false,
                          false, true);
    mutex_.Unlock();

    if (options_.enable_logging) {
      fprintf(
          stderr,
          "[AdaptiveCompaction] FORCED L%d (CF=%s): Max deferrals reached\n",
          level, cf_name.c_str());
    }

    return result;
  }

  // Calculate adaptive score: A_i = S_i × (1 - σ(t))^α
  result.dampening_factor = CalculateDampeningFactor(result.stress_factor);
  result.adaptive_score = static_score * result.dampening_factor;

  // Apply proactive boost if stress is low
  double original_adaptive_score = result.adaptive_score;
  result.adaptive_score =
      ApplyProactiveBoost(result.adaptive_score, result.stress_factor);
  result.is_boosted = (result.adaptive_score > original_adaptive_score);

  // Determine if we should compact
  result.should_compact =
      (result.adaptive_score >= options_.min_adaptive_score);
  result.is_deferred =
      !result.should_compact && (static_score >= options_.min_adaptive_score);

  // Build reason string
  std::ostringstream reason;
  if (result.should_compact) {
    reason << "Adaptive score " << result.adaptive_score << " >= threshold";
    if (result.is_boosted) {
      reason << " (boosted due to low stress)";
    }
  } else {
    reason << "Deferred: adaptive score " << result.adaptive_score
           << " < threshold (stress=" << result.stress_factor << ")";
  }
  result.reason = reason.str();

  // Record statistics
  mutex_.Lock();
  stats_.RecordDecision(result.stress_factor, result.adaptive_score,
                        result.is_deferred, result.is_boosted, false);
  mutex_.Unlock();

  if (options_.enable_logging) {
    fprintf(stderr,
            "[AdaptiveCompaction] L%d (CF=%s): S=%.2f, σ=%.3f, d=%.3f, A=%.2f "
            "→ %s\n",
            level, cf_name.c_str(), result.static_score, result.stress_factor,
            result.dampening_factor, result.adaptive_score,
            result.should_compact ? "COMPACT" : "DEFER");
  }

  return result;
}

bool AdaptiveCompactionPicker::ShouldForceCompaction(
    int level, const std::string& cf_name) {
  std::string key = MakeTrackerKey(cf_name, level);

  mutex_.Lock();
  uint32_t deferrals = deferral_tracker_[key];
  mutex_.Unlock();

  return deferrals >= options_.max_consecutive_deferrals;
}

void AdaptiveCompactionPicker::RecordCompactionDecision(
    int level, const std::string& cf_name, bool compaction_occurred) {
  std::string key = MakeTrackerKey(cf_name, level);

  mutex_.Lock();

  if (compaction_occurred) {
    // Reset deferral counter
    deferral_tracker_[key] = 0;
  } else {
    // Increment deferral counter
    deferral_tracker_[key]++;

    if (options_.enable_logging && deferral_tracker_[key] % 5 == 0) {
      fprintf(stderr,
              "[AdaptiveCompaction] L%d (CF=%s): %u consecutive deferrals\n",
              level, cf_name.c_str(), deferral_tracker_[key]);
    }
  }

  mutex_.Unlock();
}

//==============================================================================
// AdaptiveLevelCompactionPicker
//==============================================================================

Compaction* AdaptiveLevelCompactionPicker::PickCompaction(
    const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
    const MutableDBOptions& mutable_db_options,
    const std::vector<SequenceNumber>& /* existing_snapshots */,
    const SnapshotChecker* /* snapshot_checker */, VersionStorageInfo* vstorage,
    LogBuffer* log_buffer, const std::string& full_history_ts_low,
    bool /*require_max_output_level*/) {
  // If adaptive logic is not enabled, use default behavior
  if (!adaptive_picker_.IsEnabled()) {
    return LevelCompactionPicker::PickCompaction(
        cf_name, mutable_cf_options, mutable_db_options,
        /*existing_snapshots=*/{}, /* snapshot_checker */ nullptr, vstorage,
        log_buffer, /*full_history_ts_low=*/"");
  }

  // Find the level with highest priority using adaptive scoring
  int compaction_level = -1;
  double best_adaptive_score = 0.0;
  AdaptiveCompactionPicker::AdaptiveScoreResult best_result;

  for (int level = 0; level < vstorage->num_levels() - 1; level++) {
    double static_score = vstorage->CompactionScore(level);

    if (static_score <= 0) {
      continue;  // No compaction needed for this level
    }

    auto result =
        adaptive_picker_.CalculateAdaptiveScore(static_score, level, cf_name);

    // Track best candidate
    if (result.adaptive_score > best_adaptive_score) {
      best_adaptive_score = result.adaptive_score;
      best_result = result;
      compaction_level = level;
    }
  }

  // Check if we should compact based on best adaptive score
  if (compaction_level >= 0 && best_result.should_compact) {
    // Log the decision
    if (log_buffer) {
      ROCKS_LOG_BUFFER(log_buffer,
                       "[Adaptive] Picking L%d compaction (CF=%s): "
                       "static=%.2f, stress=%.3f, adaptive=%.2f, reason=%s",
                       compaction_level, cf_name.c_str(),
                       best_result.static_score, best_result.stress_factor,
                       best_result.adaptive_score, best_result.reason.c_str());
    }

    // Record that compaction occurred
    adaptive_picker_.RecordCompactionDecision(compaction_level, cf_name, true);

    // Use the parent class to actually pick the compaction
    return LevelCompactionPicker::PickCompaction(
        cf_name, mutable_cf_options, mutable_db_options, {}, nullptr, vstorage,
        log_buffer, "");
  }

  // No compaction selected - all deferred
  if (compaction_level >= 0) {
    // Record deferral
    adaptive_picker_.RecordCompactionDecision(compaction_level, cf_name, false);

    if (log_buffer) {
      ROCKS_LOG_BUFFER(log_buffer,
                       "[Adaptive] Deferring L%d compaction (CF=%s): "
                       "static=%.2f, stress=%.3f, adaptive=%.2f, reason=%s",
                       compaction_level, cf_name.c_str(),
                       best_result.static_score, best_result.stress_factor,
                       best_result.adaptive_score, best_result.reason.c_str());
    }
  }

  return nullptr;
}

bool AdaptiveLevelCompactionPicker::ShouldPickCompactionAdaptive(
    double static_score, int level, const std::string& cf_name,
    LogBuffer* log_buffer) {
  auto result =
      adaptive_picker_.CalculateAdaptiveScore(static_score, level, cf_name);

  if (log_buffer && result.should_compact) {
    ROCKS_LOG_BUFFER(
        log_buffer,
        "[Adaptive] L%d score: static=%.2f → adaptive=%.2f (σ=%.3f)", level,
        result.static_score, result.adaptive_score, result.stress_factor);
  }

  return result.should_compact;
}

//==============================================================================
// AdaptiveUniversalCompactionPicker
//==============================================================================

Compaction* AdaptiveUniversalCompactionPicker::PickCompaction(
    const std::string& cf_name, const MutableCFOptions& mutable_cf_options,
    const MutableDBOptions& mutable_db_options,
    const std::vector<SequenceNumber>& /* existing_snapshots */,
    const SnapshotChecker* /* snapshot_checker */, VersionStorageInfo* vstorage,
    LogBuffer* log_buffer, const std::string& full_history_ts_low,
    bool /*require_max_output_level*/) {
  // If adaptive logic is not enabled, use default behavior
  if (!adaptive_picker_.IsEnabled()) {
    return UniversalCompactionPicker::PickCompaction(
        cf_name, mutable_cf_options, mutable_db_options,
        /*existing_snapshots=*/{}, /* snapshot_checker */ nullptr, vstorage,
        log_buffer, /*full_history_ts_low=*/"");
  }

  // For universal compaction, we check if any compaction is needed
  // and apply adaptive logic to the overall decision

  // Universal compaction doesn't have explicit scores per level
  // We use a heuristic: if L0 files exceed threshold, that's our "score"
  int num_l0_files = vstorage->l0_delay_trigger_count();
  double static_score =
      static_cast<double>(num_l0_files) /
      static_cast<double>(
          mutable_cf_options.level0_file_num_compaction_trigger);

  auto result =
      adaptive_picker_.CalculateAdaptiveScore(static_score, 0, cf_name);

  if (result.should_compact) {
    if (log_buffer) {
      ROCKS_LOG_BUFFER(log_buffer,
                       "[Adaptive Universal] Picking compaction (CF=%s): "
                       "L0_files=%d, adaptive=%.2f",
                       cf_name.c_str(), num_l0_files, result.adaptive_score);
    }

    adaptive_picker_.RecordCompactionDecision(0, cf_name, true);

    return UniversalCompactionPicker::PickCompaction(
        cf_name, mutable_cf_options, mutable_db_options, {}, nullptr, vstorage,
        log_buffer, "");
  }

  // Defer compaction
  adaptive_picker_.RecordCompactionDecision(0, cf_name, false);

  if (log_buffer) {
    ROCKS_LOG_BUFFER(log_buffer,
                     "[Adaptive Universal] Deferring compaction (CF=%s): "
                     "stress=%.3f",
                     cf_name.c_str(), result.stress_factor);
  }

  return nullptr;
}

//==============================================================================
// Factory Function
//==============================================================================

std::unique_ptr<CompactionPicker> NewAdaptiveCompactionPicker(
    const ImmutableOptions& ioptions, const InternalKeyComparator* icmp,
    const AdaptiveCompactionOptions& adaptive_options,
    LoadObserver* load_observer) {
  switch (ioptions.compaction_style) {
    case kCompactionStyleLevel:
      return std::unique_ptr<CompactionPicker>(
          new AdaptiveLevelCompactionPicker(ioptions, icmp, adaptive_options,
                                            load_observer));

    case kCompactionStyleUniversal:
      return std::unique_ptr<CompactionPicker>(
          new AdaptiveUniversalCompactionPicker(
              ioptions, icmp, adaptive_options, load_observer));

    case kCompactionStyleFIFO:
    case kCompactionStyleNone:
    default:
      // FIFO and None don't benefit from adaptive logic
      // Fall back to standard picker
      return std::unique_ptr<CompactionPicker>(
          new LevelCompactionPicker(ioptions, icmp));
  }
}

}  // namespace ROCKSDB_NAMESPACE
