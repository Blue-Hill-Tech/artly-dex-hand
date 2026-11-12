#include "check.hpp"
#include "dexrt/impedance.hpp"

int main() {
  dexrt::ImpedanceConfig cfg;
  cfg.kp = Eigen::VectorXd::Constant(3, 2.0);
  cfg.kd = Eigen::VectorXd::Constant(3, 0.5);
  cfg.tau_max = Eigen::VectorXd::Constant(3, 10.0);
  cfg.target_timeout_s = 0.5;
  cfg.velocity_window_s = 0.01;
  dexrt::ImpedanceController ctl(cfg);

  // No target: zero torque, flags clear.
  auto out = ctl.step(Eigen::Vector3d(1, 2, 3), nullptr, 0.0);
  CHECK(!out.has_target);
  CHECK_NEAR(out.tau.norm(), 0.0, 0.0);

  // Spring term with measured velocity.
  ctl.set_target(Eigen::Vector3d(2, 2, 2), 0.0);
  Eigen::VectorXd qdot = Eigen::Vector3d(0, 4, -4);
  out = ctl.step(Eigen::Vector3d(1, 2, 3), &qdot, 0.001);
  CHECK(out.has_target && out.target_fresh && !out.saturated);
  CHECK_NEAR(out.tau[0], 2.0, 1e-12);          // kp * (2 - 1)
  CHECK_NEAR(out.tau[1], -2.0, 1e-12);         // -kd * 4
  CHECK_NEAR(out.tau[2], -2.0 + 2.0, 1e-12);   // kp * (2 - 3) - kd * (-4)

  // Saturation.
  out = ctl.step(Eigen::Vector3d(-20, 2, 2), nullptr, 0.002);
  CHECK(out.saturated);
  CHECK_NEAR(out.tau[0], 10.0, 1e-12);

  // Finite-difference velocity over the window: q moves 1 unit per 0.01 s on joint 0.
  dexrt::ImpedanceController fd(cfg);
  fd.set_target(Eigen::Vector3d::Zero(), 0.0);
  for (int k = 0; k <= 20; ++k) {
    out = fd.step(Eigen::Vector3d(100.0 * k * 0.001, 0, 0), nullptr, k * 0.001);
  }
  CHECK_NEAR(out.qdot[0], 100.0, 1e-6);
  CHECK_NEAR(out.qdot[1], 0.0, 1e-12);

  // Watchdog: hold-last keeps the spring; zero-on-timeout cuts it.
  out = ctl.step(Eigen::Vector3d(1, 2, 2), nullptr, 0.4);
  CHECK(out.target_fresh);
  out = ctl.step(Eigen::Vector3d(1, 2, 2), nullptr, 0.6);
  CHECK(!out.target_fresh);
  CHECK_NEAR(out.tau[0], 2.0, 1e-9);  // still pulling toward the last target
  cfg.zero_torque_on_timeout = true;
  dexrt::ImpedanceController cut(cfg);
  cut.set_target(Eigen::Vector3d(2, 2, 2), 0.0);
  out = cut.step(Eigen::Vector3d(1, 2, 2), nullptr, 0.6);
  CHECK(!out.target_fresh);
  CHECK_NEAR(out.tau.norm(), 0.0, 0.0);

  // Deadband.
  cfg.zero_torque_on_timeout = false;
  cfg.deadband = 0.5;
  dexrt::ImpedanceController db(cfg);
  db.set_target(Eigen::Vector3d(2, 2, 2), 0.0);
  out = db.step(Eigen::Vector3d(1.7, 2, 2), nullptr, 0.0);
  CHECK_NEAR(out.tau[0], 0.0, 0.0);

  // Validation.
  CHECK_THROWS(ctl.set_target(Eigen::Vector2d(0, 0), 0.0));
  Eigen::VectorXd nan_target = Eigen::Vector3d(0, 0, std::numeric_limits<double>::quiet_NaN());
  CHECK_THROWS(ctl.set_target(nan_target, 0.0));
  dexrt::ImpedanceConfig bad = cfg;
  bad.kd[0] = -1;
  CHECK_THROWS(dexrt::ImpedanceController{bad});
  return dexrt_test::finish("test_impedance");
}
