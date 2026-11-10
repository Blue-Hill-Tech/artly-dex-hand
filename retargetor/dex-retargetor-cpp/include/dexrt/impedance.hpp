// Joint-space impedance (spring-damper) law applied after retargeting:
//     tau = kp .* (q_target - q) - kd .* qdot,  clamped to +-tau_max.
// Unit-agnostic: use degrees or radians consistently for q, qdot, kp and kd.
// Velocity is either supplied (measured) or estimated by a finite difference
// over a short window; there is no low-pass filter on positions or targets.
#pragma once

#include <Eigen/Core>
#include <deque>
#include <utility>

namespace dexrt {

struct ImpedanceConfig {
  Eigen::VectorXd kp;       // torque units per position unit
  Eigen::VectorXd kd;       // torque units per (position unit / s)
  Eigen::VectorXd tau_max;  // saturation, same units as the output
  double target_timeout_s = 0.5;        // <= 0 disables the target watchdog
  bool zero_torque_on_timeout = false;  // false: keep pulling toward the last target
  double velocity_window_s = 0.01;      // finite-difference span when no measured velocity is given
  double deadband = 0.0;                // |q_target - q| below this contributes no spring torque
};

struct ImpedanceOutput {
  Eigen::VectorXd tau;
  Eigen::VectorXd qdot;      // velocity used by the damper
  bool has_target = false;
  bool target_fresh = false;
  bool saturated = false;
};

class ImpedanceController {
 public:
  explicit ImpedanceController(ImpedanceConfig config);

  void set_target(const Eigen::VectorXd& q_target, double now);
  void set_gains(const Eigen::VectorXd& kp, const Eigen::VectorXd& kd);
  // qdot_measured may be null; then velocity is estimated from the position history.
  ImpedanceOutput step(const Eigen::VectorXd& q, const Eigen::VectorXd* qdot_measured, double now);
  void reset();

  const ImpedanceConfig& config() const { return config_; }
  bool has_target() const { return has_target_; }
  const Eigen::VectorXd& target() const { return target_; }
  int size() const { return n_; }

 private:
  ImpedanceConfig config_;
  int n_ = 0;
  Eigen::VectorXd target_;
  double target_time_ = 0.0;
  bool has_target_ = false;
  std::deque<std::pair<double, Eigen::VectorXd>> history_;
};

}  // namespace dexrt
