// Compliance against a simulated D21 joint: the firmware position loop moves the finger toward
// its command at a speed cap and draws current proportional to the remaining error.
#include <functional>

#include "check.hpp"
#include "dexrt/compliance.hpp"

namespace {

constexpr double kDt = 0.02;       // 50 Hz, the hand's feedback rate
constexpr double kSpeed = 100.0;   // deg/s firmware speed cap
constexpr double kAmpPerDeg = 0.03;
constexpr double kIdle = 0.05;

struct Sim {
  int joint;
  int motor_a, motor_b;
  double sign = 1.0;  // readback sign of the simulated joint
  double q = 0.0;     // true command-frame angle
  double cmd = 0.0;
  // Returns the forced position while a hand pushes, or NAN.
  std::function<double(double)> push = [](double) { return NAN; };
  double obstacle = 1e9;  // finger cannot pass this angle

  Eigen::VectorXd read() const {
    Eigen::VectorXd r = Eigen::VectorXd::Zero(16);
    r[joint] = sign * q;
    return r;
  }
  // Signed like the real hand: the motor drives toward its command, so the current sign follows
  // (cmd - q); idle noise alternates sign. While a hand holds the finger the firmware's integral
  // action builds a stall current (push_current) even for a tiny displacement.
  double push_current = 0.0;
  bool pushed = false;
  mutable int tick = 0;
  Eigen::VectorXd currents() const {
    Eigen::VectorXd i = Eigen::VectorXd::Constant(16, (++tick % 2) ? kIdle : -kIdle);
    double load = std::clamp(kAmpPerDeg * (cmd - q), -1.0, 1.0);
    if (pushed && push_current > 0) load = (cmd > q ? 1.0 : -1.0) * push_current;
    const double value = std::fabs(load) > kIdle ? load : i[motor_a];
    i[motor_a] = value;
    i[motor_b] = value;
    return i;
  }
  void advance(double t) {
    const double forced = push(t);
    pushed = !std::isnan(forced);
    if (pushed) {
      q = forced;
      return;
    }
    const double step = std::clamp(cmd - q, -kSpeed * kDt, kSpeed * kDt);
    q = std::min(q + step, obstacle);
  }
};

// Runs `seconds` of closed loop with a constant target; returns the controller's last output.
dexrt::ComplianceOutput run(dexrt::ComplianceController& ctl, Sim& sim, double target, double& t, double seconds) {
  Eigen::VectorXd q_target = Eigen::VectorXd::Zero(16);
  q_target[sim.joint] = target;
  dexrt::ComplianceOutput out;
  for (double end = t + seconds; t < end; t += kDt) {
    out = ctl.step(q_target, sim.read(), sim.currents(), t);
    sim.cmd = out.q_cmd[sim.joint];
    sim.advance(t);
  }
  return out;
}

}  // namespace

int main() {
  // Idle calibration from noisy samples.
  {
    dexrt::ComplianceController ctl;
    for (int k = 0; k < 50; ++k) ctl.add_idle_sample(Eigen::VectorXd::Constant(16, (k % 2) ? 0.20 : -0.24));
    ctl.finish_idle_calibration();
    CHECK(ctl.idle_current().minCoeff() > 0.22 && ctl.idle_current().maxCoeff() < 0.30);
    dexrt::ComplianceController empty;
    CHECK_THROWS(empty.finish_idle_calibration());
  }

  // 1. Hold, push, release: the finger yields while pushed and returns afterwards.
  {
    dexrt::ComplianceController ctl;
    Sim sim{6, 6, 6};  // index PIP, motor 7
    sim.q = sim.cmd = 45.0;
    double t = 0;
    run(ctl, sim, 45.0, t, 0.5);
    CHECK_NEAR(sim.q, 45.0, 0.5);
    sim.push = [](double time) { return time >= 0.5 && time < 2.5 ? 60.0 : NAN; };
    auto out = run(ctl, sim, 45.0, t, 2.0);
    const double pushed_offset = out.offset[6];
    CHECK(pushed_offset > 5.0);                  // command followed the push
    const double pushed_current = sim.currents()[6];
    CHECK(pushed_current < kIdle + kAmpPerDeg * 15.0);  // less effort than a rigid hold
    out = run(ctl, sim, 45.0, t, 3.0);
    CHECK_NEAR(sim.q, 45.0, 1.0);                // back at the target
    CHECK(std::fabs(out.offset[6]) < 1.0);
    std::printf("push: offset while pushed %.1f deg, current %.3f A; after release q %.2f\n", pushed_offset,
                pushed_current, sim.q);
  }

  // 2. Grasp: an object blocks the finger short of the target; the command backs off.
  {
    dexrt::ComplianceController ctl;
    Sim sim{5, 4, 5};  // index MCP, motor pair 5/6
    sim.q = sim.cmd = 10.0;
    sim.obstacle = 50.0;
    double t = 0;
    auto out = run(ctl, sim, 85.0, t, 4.0);
    CHECK_NEAR(sim.q, 50.0, 0.1);
    const double rigid_current = kIdle + kAmpPerDeg * std::min(15.0, 85.0 - 50.0);  // error clamp alone
    CHECK(sim.currents()[4] < rigid_current);
    CHECK(out.offset[5] < -10.0);
    std::printf("grasp: command %.1f deg vs target 85, current %.3f A (clamp-only %.3f A)\n", out.q_cmd[5],
                sim.currents()[4], rigid_current);
  }

  // 3. Free motion is not mistaken for a push.
  {
    dexrt::ComplianceController ctl;
    Sim sim{9, 9, 9};  // middle PIP
    sim.q = sim.cmd = 10.0;
    double t = 0;
    double reached = -1;
    Eigen::VectorXd q_target = Eigen::VectorXd::Zero(16);
    q_target[9] = 80.0;
    for (; t < 3.0; t += kDt) {
      const auto out = ctl.step(q_target, sim.read(), sim.currents(), t);
      sim.cmd = out.q_cmd[9];
      sim.advance(t);
      if (reached < 0 && std::fabs(sim.q - 80.0) < 1.0) reached = t;
    }
    CHECK(reached > 0 && reached < 1.2);  // 70 deg at 100 deg/s is 0.7 s
    CHECK_NEAR(sim.q, 80.0, 0.5);
    std::printf("free motion: reached 80 deg after %.2f s\n", reached);
  }

  // 4. Mirrored abduction readback: a push still moves the command the right way.
  {
    dexrt::ComplianceController ctl;
    Sim sim{4, 4, 5};  // index abduction, readback sign -1 in the default config
    sim.sign = -1.0;
    sim.q = sim.cmd = 5.0;
    double t = 0;
    run(ctl, sim, 5.0, t, 0.5);
    sim.push = [](double time) { return time >= 0.5 ? 12.0 : NAN; };
    const auto out = run(ctl, sim, 5.0, t, 1.5);
    CHECK(out.q_meas[4] > 11.0);   // measured converted back to the command frame
    CHECK(out.offset[4] > 2.0);    // yields toward +12, not away from it
  }

  // 5. Current-sign direction: after a driven move teaches the sign, a push that displaces the
  //    finger by less than the displacement threshold still yields, in the right direction.
  {
    dexrt::ComplianceController ctl;
    Sim sim{6, 6, 6};
    sim.push_current = 0.25;
    sim.q = sim.cmd = 10.0;
    double t = 0;
    run(ctl, sim, 90.0, t, 1.5);  // 0.8 s driven move: learns sign(current) vs sign(velocity)
    CHECK(ctl.drive_signs()(6, 0) != 0.0);
    CHECK_NEAR(sim.q, 90.0, 0.5);
    run(ctl, sim, 60.0, t, 1.5);
    // Stiff push: the hand holds the finger only 0.05 deg past its command, but the motor
    // fights it with a clearly negative current (driving back toward the command).
    sim.push = [&](double) { return sim.cmd + 0.05; };
    auto out = run(ctl, sim, 60.0, t, 1.0);
    CHECK(out.source[6] == 2.0);
    CHECK(out.offset[6] > 3.0);  // yields toward the push (+)
    // Same with displacement-only mode: the 0.05 deg push is below the threshold and is ignored.
    dexrt::ComplianceConfig disp;
    disp.direction_from_current = false;
    dexrt::ComplianceController ctl2(disp);
    Sim sim2{6, 6, 6};
    sim2.push_current = 0.25;
    sim2.q = sim2.cmd = 60.0;
    double t2 = 0;
    run(ctl2, sim2, 60.0, t2, 0.5);
    sim2.push = [&](double) { return sim2.cmd + 0.05; };
    out = run(ctl2, sim2, 60.0, t2, 1.0);
    CHECK_NEAR(out.offset[6], 0.0, 1e-9);
    std::printf("current-sign direction: learned sign %+.0f, offset %.1f deg from a 0.05 deg displacement\n",
                ctl.drive_signs()(6, 0), run(ctl, sim, 60.0, t, 0.02).offset[6]);
  }

  // 6. Motor pair decomposition: a flexion push on the MCP must not make abduction yield.
  {
    dexrt::ComplianceController ctl;
    Eigen::VectorXd target = Eigen::VectorXd::Zero(16), read = Eigen::VectorXd::Zero(16);
    Eigen::VectorXd currents = Eigen::VectorXd::Zero(16);
    // Teach both motors of the index pair with a flexion move (both currents same sign as velocity).
    double t = 0;
    for (int k = 0; k < 60; ++k, t += kDt) {
      target[5] = 40.0;
      read[5] = std::min(40.0, k * kSpeed * kDt);
      currents[4] = currents[5] = read[5] < 40.0 ? 0.4 : 0.0;
      ctl.step(target, read, currents, t);
    }
    CHECK(ctl.drive_signs()(5, 0) != 0.0 && ctl.drive_signs()(5, 1) != 0.0);
    // Now a flexion push: finger held at 40, both motors push back with the same sign.
    for (int k = 0; k < 50; ++k, t += kDt) {
      currents[4] = currents[5] = -0.4;
      const auto out = ctl.step(target, read, currents, t);
      if (k == 49) {
        CHECK(out.source[5] == 2.0);
        CHECK_NEAR(out.effort[4], 0.0, 1e-12);  // abduction sees no net drive
        CHECK(out.yielding[4] == 0.0);
      }
    }
  }

  // 7. Limits, clamp and validation.
  {
    dexrt::ComplianceController ctl;
    Eigen::VectorXd target = Eigen::VectorXd::Zero(16), read = Eigen::VectorXd::Zero(16);
    target[6] = 150.0;
    const auto out = ctl.step(target, read, Eigen::VectorXd::Zero(16), 0.0);
    CHECK_NEAR(out.q_cmd[6], 15.0, 1e-12);  // error clamp around the measured 0 deg
    target[6] = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS(ctl.step(target, read, Eigen::VectorXd::Zero(16), 0.02));
    dexrt::ComplianceConfig bad;
    bad.readback_sign[3] = 0.5;
    CHECK_THROWS(dexrt::ComplianceController{bad});
  }
  return dexrt_test::finish("test_compliance");
}
