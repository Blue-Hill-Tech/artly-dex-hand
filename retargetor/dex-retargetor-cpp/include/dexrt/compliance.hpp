// Position-mode compliance for the AAC D21 hand.
//
// The hand's firmware keeps closing its own position loop; this class only moves the
// position target it is given. Per joint (degrees, SDK joint order, command frame):
//
//   effort  = max(0, |motor current| - idle current)            (pair motors share it)
//   rest    = slow average of (q_meas - q_cmd) while unloaded   (firmware tracking bias)
//   push    = sign(q_meas - q_cmd_prev - rest) * gain * effort   only while the joint is
//             displaced beyond that bias and not moving freely
//
// The firmware position loop is stiff: a hand push shows up as only ~0.2 deg of
// displacement on top of a ~0.1 deg resting bias, so the bias is learned per joint and
// the threshold sits just above the sensor noise (~0.03 deg).
//
// The stronger direction signal is the motor current's sign: the motor drives against
// the push. Each (joint, motor) drive sign is learned online whenever the firmware moves
// that joint (sign(current) vs sign(velocity)); once known, push direction and effort
// come from the signed drive  d_j = sum_m s_jm * i_m  (a motor pair decomposes into
// flexion = same signs, abduction = opposite signs), and displacement is only a fallback.
//   offset += (push - return_rate * offset) * dt                 clamped to +-offset_max
//   q_cmd   = clamp(q_target + offset, q_meas +- error_clamp, joint limits)
//
// A push displaces the finger from its command and raises current, so the offset follows
// the push and the finger yields; when the push stops the offset decays and the finger
// returns to the target. A finger stalled on an object behind its command backs the
// command off until the current falls, which limits grip force. Free motion (finger
// moving toward its command) is not mistaken for a push because of the velocity gate.
#pragma once

#include <Eigen/Core>
#include <array>
#include <deque>
#include <utility>

namespace dexrt {

constexpr int kJoints = 16;

struct ComplianceConfig {
  // Readback sign per joint: finger abduction (4, 7, 10, 13) reads back mirrored on the D21.
  std::array<double, kJoints> readback_sign{1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1};
  std::array<double, kJoints> lower_deg{0, -35, 0, 0, -20, 0, 0, -20, 0, 0, -20, 0, 0, -20, 0, 0};
  std::array<double, kJoints> upper_deg{100, 35, 50, 70, 20, 90, 100, 20, 90, 100, 20, 90, 100, 20, 90, 100};
  double min_idle_current_a = 0.10;  // floor for the per-motor idle level
  double idle_sigma = 3.0;           // idle level = mean|i| + idle_sigma * std|i| from calibration
  double gain_dps_per_a = 150.0;     // offset speed per amp of effort
  double return_rate = 2.0;          // 1/s: how fast the offset decays back to the target
  double offset_max_deg = 30.0;
  double error_clamp_deg = 15.0;     // max |q_cmd - q_meas|; 0 disables
  double velocity_gate_dps = 15.0;   // a joint moving faster than this is moving freely
  double deviation_min_deg = 0.08;   // displacement beyond the resting bias that counts as a push
  double velocity_window_s = 0.1;    // feedback updates at ~50 Hz
  double max_dt_s = 0.1;
  double rest_bias_tau_s = 1.0;      // time constant of the unloaded (q_meas - q_cmd) average
  bool direction_from_current = true;  // use learned current signs when available
  int sign_min_samples = 10;         // driven samples (~0.2 s) before a (joint, motor) sign is trusted
  double sign_min_agreement = 0.6;   // |mean sign product| needed to trust it
  double target_motion_gate_dps = 10.0;  // a target moving faster than this is a commanded move...
  double target_holdoff_s = 0.2;         // ...and pushes are ignored on that joint for this long after
};

struct ComplianceOutput {
  Eigen::VectorXd q_cmd;    // command-frame degrees to write
  Eigen::VectorXd offset;   // degrees
  Eigen::VectorXd effort;   // amps above idle, per joint
  Eigen::VectorXd q_meas;   // command-frame measured degrees
  Eigen::VectorXd velocity; // deg/s
  Eigen::VectorXd yielding; // 1 where a push is being followed
  Eigen::VectorXd deviation; // q_meas - previous command - resting bias, degrees
  Eigen::VectorXd source;    // per joint: 0 no push, 1 direction from displacement, 2 from current sign
};

class ComplianceController {
 public:
  explicit ComplianceController(ComplianceConfig config = {});

  // Idle-current calibration: feed samples while the hand holds still, then finish.
  void add_idle_sample(const Eigen::VectorXd& currents_a);
  void finish_idle_calibration();
  const Eigen::VectorXd& idle_current() const { return idle_; }

  // q_target: command-frame degrees; q_read: raw readback degrees; currents: 16 motor currents (A).
  ComplianceOutput step(const Eigen::VectorXd& q_target, const Eigen::VectorXd& q_read,
                        const Eigen::VectorXd& currents_a, double now);
  void reset();  // forget offsets and history (e.g. when execution stops)

  const ComplianceConfig& config() const { return config_; }
  const Eigen::VectorXd& rest_bias() const { return rest_bias_; }
  // Learned drive sign per joint and motor slot (0 = unknown); 16 x 2.
  Eigen::MatrixXd drive_signs() const;
  static const std::array<std::array<int, 2>, kJoints>& joint_motors();

 private:
  ComplianceConfig config_;
  Eigen::VectorXd idle_, offset_, last_cmd_, rest_bias_;
  Eigen::Matrix<bool, Eigen::Dynamic, 1> bias_known_;
  Eigen::MatrixXd sign_sum_;    // 16 x 2 running sum of sign(i_m) * sign(v_j)
  Eigen::MatrixXi sign_count_;  // 16 x 2
  Eigen::VectorXd last_target_, target_moved_at_;
  bool has_last_ = false;
  double last_time_ = 0.0;
  std::deque<std::pair<double, Eigen::VectorXd>> history_;
  std::deque<Eigen::VectorXd> idle_samples_;
};

}  // namespace dexrt
