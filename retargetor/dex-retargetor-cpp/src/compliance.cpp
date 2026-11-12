#include "dexrt/compliance.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dexrt {
namespace {

// Motors (0-based) driving each joint, from the D21 protocol spec section 3.5: MCP flexion and
// abduction of each finger (and thumb abduction/rotation) share a differential motor pair.
const std::array<std::array<int, 2>, kJoints> kJointMotors = {{
    {0, 1}, {0, 1}, {2, 2}, {3, 3},        // thumb abduction, rotation, cmc, mcp
    {4, 5}, {4, 5}, {6, 6},                // index abduction, mcp, pip
    {7, 8}, {7, 8}, {9, 9},                // middle
    {10, 11}, {10, 11}, {12, 12},          // ring
    {13, 14}, {13, 14}, {15, 15},          // pinky
}};

int motor_count(int joint) { return kJointMotors[joint][0] == kJointMotors[joint][1] ? 1 : 2; }

// The other joint driven by the same motor pair (abduction <-> flexion), or -1.
int pair_partner(int joint) {
  if (motor_count(joint) == 1) return -1;
  for (int j = 0; j < kJoints; ++j) {
    if (j != joint && kJointMotors[j] == kJointMotors[joint]) return j;
  }
  return -1;
}

double sign_of(double v) { return v > 0 ? 1.0 : v < 0 ? -1.0 : 0.0; }

void check_vector(const Eigen::VectorXd& v, const char* name) {
  if (v.size() != kJoints || !v.allFinite()) {
    throw std::invalid_argument(std::string("compliance: ") + name + " must be 16 finite values");
  }
}

}  // namespace

const std::array<std::array<int, 2>, kJoints>& ComplianceController::joint_motors() { return kJointMotors; }

ComplianceController::ComplianceController(ComplianceConfig config) : config_(config) {
  const auto& c = config_;
  if (!(c.min_idle_current_a >= 0) || !(c.idle_sigma >= 0) || !(c.gain_dps_per_a >= 0) || !(c.return_rate > 0) ||
      !(c.offset_max_deg >= 0) || !(c.error_clamp_deg >= 0) || !(c.velocity_gate_dps > 0) ||
      !(c.deviation_min_deg >= 0) || !(c.velocity_window_s > 0) || !(c.max_dt_s > 0) || !(c.rest_bias_tau_s > 0) ||
      c.sign_min_samples < 1 || !(c.sign_min_agreement > 0 && c.sign_min_agreement <= 1) ||
      !(c.target_motion_gate_dps > 0) || !(c.target_holdoff_s >= 0)) {
    throw std::invalid_argument("compliance: invalid configuration");
  }
  for (int j = 0; j < kJoints; ++j) {
    if (std::fabs(c.readback_sign[j]) != 1.0 || !(c.lower_deg[j] < c.upper_deg[j])) {
      throw std::invalid_argument("compliance: invalid readback sign or joint limits");
    }
  }
  idle_ = Eigen::VectorXd::Constant(kJoints, c.min_idle_current_a);
  offset_ = Eigen::VectorXd::Zero(kJoints);
  last_cmd_ = Eigen::VectorXd::Zero(kJoints);
  rest_bias_ = Eigen::VectorXd::Zero(kJoints);
  bias_known_ = Eigen::Matrix<bool, Eigen::Dynamic, 1>::Constant(kJoints, false);
  sign_sum_ = Eigen::MatrixXd::Zero(kJoints, 2);
  sign_count_ = Eigen::MatrixXi::Zero(kJoints, 2);
  last_target_ = Eigen::VectorXd::Zero(kJoints);
  target_moved_at_ = Eigen::VectorXd::Constant(kJoints, -1e300);
}

void ComplianceController::add_idle_sample(const Eigen::VectorXd& currents_a) {
  check_vector(currents_a, "currents");
  idle_samples_.push_back(currents_a.cwiseAbs());
}

void ComplianceController::finish_idle_calibration() {
  if (idle_samples_.size() < 2) throw std::runtime_error("compliance: need at least 2 idle samples");
  const double n = static_cast<double>(idle_samples_.size());
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(kJoints), sq = Eigen::VectorXd::Zero(kJoints);
  for (const auto& s : idle_samples_) {
    mean += s;
    sq += s.cwiseProduct(s);
  }
  mean /= n;
  const Eigen::VectorXd stddev = (sq / n - mean.cwiseProduct(mean)).cwiseMax(0.0).cwiseSqrt();
  idle_ = (mean + config_.idle_sigma * stddev).cwiseMax(config_.min_idle_current_a);
  idle_samples_.clear();
}

void ComplianceController::reset() {
  // Offsets and short-term history go; learned idle currents and drive signs are hardware facts and stay.
  offset_.setZero();
  has_last_ = false;
  history_.clear();
  bias_known_.setConstant(false);
}

Eigen::MatrixXd ComplianceController::drive_signs() const {
  Eigen::MatrixXd out = Eigen::MatrixXd::Zero(kJoints, 2);
  for (int j = 0; j < kJoints; ++j) {
    for (int k = 0; k < motor_count(j); ++k) {
      const int n = sign_count_(j, k);
      if (n >= config_.sign_min_samples) {
        const double mean = sign_sum_(j, k) / n;
        if (std::fabs(mean) >= config_.sign_min_agreement) out(j, k) = sign_of(mean);
      }
    }
  }
  return out;
}

ComplianceOutput ComplianceController::step(const Eigen::VectorXd& q_target, const Eigen::VectorXd& q_read,
                                            const Eigen::VectorXd& currents_a, double now) {
  check_vector(q_target, "q_target");
  check_vector(q_read, "q_read");
  check_vector(currents_a, "currents");
  if (!std::isfinite(now)) throw std::invalid_argument("compliance: time must be finite");
  const auto& c = config_;

  ComplianceOutput out;
  out.q_meas.resize(kJoints);
  for (int j = 0; j < kJoints; ++j) out.q_meas[j] = c.readback_sign[j] * q_read[j];

  // Velocity from the oldest sample inside the window; time going backwards resets the history.
  if (!history_.empty() && now < history_.back().first) history_.clear();
  history_.emplace_back(now, out.q_meas);
  while (history_.size() > 1 && history_[1].first <= now - c.velocity_window_s) history_.pop_front();
  const double span = now - history_.front().first;
  out.velocity = span > 0 ? ((out.q_meas - history_.front().second) / span).eval() : Eigen::VectorXd::Zero(kJoints);

  const double dt = has_last_ ? std::clamp(now - last_time_, 0.0, c.max_dt_s) : 0.0;
  const Eigen::VectorXd previous_cmd = has_last_ ? last_cmd_ : q_target;
  const Eigen::MatrixXd signs = drive_signs();
  // A target that moves fast is a commanded move; the current it draws is tracking, not a push.
  for (int j = 0; j < kJoints; ++j) {
    const double rate = has_last_ && dt > 0 ? std::fabs(q_target[j] - last_target_[j]) / dt : 0.0;
    if (!has_last_ || rate >= c.target_motion_gate_dps) target_moved_at_[j] = now;
  }
  last_target_ = q_target;

  out.effort.resize(kJoints);
  out.yielding = Eigen::VectorXd::Zero(kJoints);
  out.deviation.resize(kJoints);
  out.source = Eigen::VectorXd::Zero(kJoints);
  const double bias_alpha = has_last_ ? std::min(1.0, dt / c.rest_bias_tau_s) : 1.0;
  for (int j = 0; j < kJoints; ++j) {
    const int n_motors = motor_count(j);
    const bool fast = std::fabs(out.velocity[j]) >= c.velocity_gate_dps;
    const bool target_moving = now - target_moved_at_[j] < c.target_holdoff_s;

    // Learn each motor's drive sign from firmware-driven motion of this joint (not while yielding).
    if (fast && std::fabs(offset_[j]) < 0.5) {
      for (int k = 0; k < n_motors; ++k) {
        const int m = kJointMotors[j][k];
        if (std::fabs(currents_a[m]) > idle_[m]) {
          sign_sum_(j, k) += sign_of(currents_a[m]) * sign_of(out.velocity[j]);
          sign_count_(j, k) += 1;
        }
      }
    }

    // Effort and push direction: from the signed drive when every motor's sign is known.
    bool signs_known = c.direction_from_current;
    for (int k = 0; k < n_motors && signs_known; ++k) signs_known = signs(j, k) != 0.0;
    // A pair partner's known signs still give this joint's drive magnitude: the pair acts on
    // the partner with equal signs and on this joint with opposite signs (spec section 3.5).
    const int partner = pair_partner(j);
    bool partner_known = c.direction_from_current && !signs_known && partner >= 0;
    for (int k = 0; k < n_motors && partner_known; ++k) partner_known = signs(partner, k) != 0.0;
    double effort = 0.0, drive = 0.0, idle = 0.0;
    if (signs_known || partner_known) {
      for (int k = 0; k < n_motors; ++k) {
        const int m = kJointMotors[j][k];
        const double s = signs_known ? signs(j, k) : signs(partner, k) * (k == 0 ? 1.0 : -1.0);
        drive += s * currents_a[m];
        idle = std::max(idle, idle_[m]);
      }
      effort = std::max(0.0, std::fabs(drive) / n_motors - idle);
    } else {
      for (int k = 0; k < n_motors; ++k) {
        const int m = kJointMotors[j][k];
        effort = std::max(effort, std::fabs(currents_a[m]) - idle_[m]);
      }
    }
    out.effort[j] = effort;
    // Free motion goes along the drive; a yielding finger moves against it and must not be gated.
    const bool free_moving = fast && (!(signs_known || partner_known) || sign_of(out.velocity[j]) == sign_of(drive));

    // Unloaded and still, with a command actually written: track the firmware's resting
    // tracking bias for this joint. Before the first write the deviation is meaningless.
    const double raw_deviation = out.q_meas[j] - previous_cmd[j];
    if (has_last_ && effort <= 0 && !fast) {
      if (!bias_known_[j]) {
        rest_bias_[j] = raw_deviation;
        bias_known_[j] = true;
      } else {
        rest_bias_[j] += bias_alpha * (raw_deviation - rest_bias_[j]);
      }
    }
    const double deviation = raw_deviation - (bias_known_[j] ? rest_bias_[j] : 0.0);
    out.deviation[j] = deviation;

    double direction = 0.0;
    if (effort > 0 && !free_moving && !target_moving) {
      if (signs_known) {
        direction = -sign_of(drive);  // the motor drives against the push
        out.source[j] = 2.0;
      } else if (std::fabs(deviation) > c.deviation_min_deg) {
        direction = sign_of(deviation);
        out.source[j] = 1.0;
      }
    }
    const double push = direction * c.gain_dps_per_a * effort;
    out.yielding[j] = direction != 0.0 ? 1.0 : 0.0;
    offset_[j] += (push - c.return_rate * offset_[j]) * dt;
    offset_[j] = std::clamp(offset_[j], -c.offset_max_deg, c.offset_max_deg);
  }
  out.offset = offset_;

  out.q_cmd = q_target + offset_;
  for (int j = 0; j < kJoints; ++j) {
    if (c.error_clamp_deg > 0) {
      out.q_cmd[j] = std::clamp(out.q_cmd[j], out.q_meas[j] - c.error_clamp_deg, out.q_meas[j] + c.error_clamp_deg);
    }
    out.q_cmd[j] = std::clamp(out.q_cmd[j], c.lower_deg[j], c.upper_deg[j]);
  }
  last_cmd_ = out.q_cmd;
  last_time_ = now;
  has_last_ = true;
  return out;
}

}  // namespace dexrt
