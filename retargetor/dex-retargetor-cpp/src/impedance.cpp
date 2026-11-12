#include "dexrt/impedance.hpp"

#include <cmath>
#include <stdexcept>

namespace dexrt {

ImpedanceController::ImpedanceController(ImpedanceConfig config) : config_(std::move(config)) {
  n_ = static_cast<int>(config_.kp.size());
  if (n_ == 0 || config_.kd.size() != n_ || config_.tau_max.size() != n_) {
    throw std::invalid_argument("impedance: kp, kd and tau_max must have the same non-zero size");
  }
  if (!config_.kp.allFinite() || !config_.kd.allFinite() || !config_.tau_max.allFinite() ||
      (config_.kp.array() < 0).any() || (config_.kd.array() < 0).any() || (config_.tau_max.array() < 0).any()) {
    throw std::invalid_argument("impedance: gains and limits must be finite and non-negative");
  }
  if (!(config_.velocity_window_s > 0) || !std::isfinite(config_.velocity_window_s)) {
    throw std::invalid_argument("impedance: velocity_window_s must be positive");
  }
  if (config_.deadband < 0) throw std::invalid_argument("impedance: deadband must be non-negative");
  target_ = Eigen::VectorXd::Zero(n_);
}

void ImpedanceController::set_target(const Eigen::VectorXd& q_target, double now) {
  if (q_target.size() != n_) throw std::invalid_argument("impedance: target has wrong size");
  if (!q_target.allFinite()) throw std::invalid_argument("impedance: target must be finite");
  target_ = q_target;
  target_time_ = now;
  has_target_ = true;
}

void ImpedanceController::set_gains(const Eigen::VectorXd& kp, const Eigen::VectorXd& kd) {
  if (kp.size() != n_ || kd.size() != n_ || !kp.allFinite() || !kd.allFinite() ||
      (kp.array() < 0).any() || (kd.array() < 0).any()) {
    throw std::invalid_argument("impedance: invalid gains");
  }
  config_.kp = kp;
  config_.kd = kd;
}

void ImpedanceController::reset() {
  has_target_ = false;
  history_.clear();
}

ImpedanceOutput ImpedanceController::step(const Eigen::VectorXd& q, const Eigen::VectorXd* qdot_measured, double now) {
  if (q.size() != n_) throw std::invalid_argument("impedance: q has wrong size");
  if (!q.allFinite() || !std::isfinite(now)) throw std::invalid_argument("impedance: q and now must be finite");
  ImpedanceOutput out;
  out.tau = Eigen::VectorXd::Zero(n_);

  if (qdot_measured != nullptr) {
    if (qdot_measured->size() != n_ || !qdot_measured->allFinite()) {
      throw std::invalid_argument("impedance: measured velocity invalid");
    }
    out.qdot = *qdot_measured;
  } else {
    // Finite difference against the oldest sample still inside the window.
    // Time going backwards invalidates the history rather than producing a bogus slope.
    if (!history_.empty() && now < history_.back().first) history_.clear();
    history_.emplace_back(now, q);
    while (history_.size() > 1 && history_[1].first <= now - config_.velocity_window_s) history_.pop_front();
    const double dt = now - history_.front().first;
    out.qdot = dt > 0 ? ((q - history_.front().second) / dt).eval() : Eigen::VectorXd::Zero(n_);
  }

  out.has_target = has_target_;
  if (!has_target_) return out;
  out.target_fresh = config_.target_timeout_s <= 0 || now - target_time_ <= config_.target_timeout_s;
  if (!out.target_fresh && config_.zero_torque_on_timeout) return out;

  Eigen::VectorXd error = target_ - q;
  if (config_.deadband > 0) {
    error = (error.array().abs() <= config_.deadband).select(0.0, error);
  }
  out.tau = config_.kp.cwiseProduct(error) - config_.kd.cwiseProduct(out.qdot);
  const Eigen::VectorXd clamped = out.tau.cwiseMax(-config_.tau_max).cwiseMin(config_.tau_max);
  out.saturated = (clamped.array() != out.tau.array()).any();
  out.tau = clamped;
  return out;
}

}  // namespace dexrt
