// Fast inner loop: read the AAC hand, run the impedance law, write torques.
//
// Targets arrive as UDP datagrams of 16 little-endian float64 joint angles in
// DEGREES (SDK joint order) on 127.0.0.1:<udp-port>. Until the first datagram
// the target is the pose measured at start, so the spring exerts no force.
//
// Without --torque-mode this is a DRY RUN: it reads the hand and computes
// torques but never writes a register. With --torque-mode it switches the hand
// to torque mode (register 21 = 2) and writes registers 68-83 every cycle.
// On exit, error or Ctrl+C it writes zero torque, copies the measured pose into
// the position targets, and restores position mode.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "dexrt/d21_canfd.hpp"
#include "dexrt/impedance.hpp"
#include "dexrt/motor_map.hpp"

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Args {
  std::string port;
  int address = 1;
  double rate = 500;
  double kp = 2.0;        // torque units per degree
  double kd = 0.02;       // torque units per (deg/s)
  double tau_max = 200;   // of 1000
  int udp_port = 6400;
  double timeout = 0.5;
  bool zero_on_timeout = false;
  bool measured_velocity = false;
  double velocity_window = 0.01;
  double deadband = 0.0;
  bool torque_mode = false;
  std::string log;
  double duration = 0;    // 0 = until Ctrl+C
  double sleep_after_open = 2.0;
  dexrt::HandSide hand = dexrt::HandSide::Left;   // motor-pair polarity (dexrt/motor_map.hpp)
};

void usage() {
  std::fprintf(stderr,
      "usage: dexrt_impedance_loop --port DEV [--address N] [--rate HZ] [--kp K] [--kd D] [--tau-max T]\n"
      "       [--udp-port P] [--timeout S] [--zero-on-timeout] [--measured-velocity] [--velocity-window S]\n"
      "       [--deadband DEG] [--torque-mode] [--log FILE] [--duration S] [--sleep-after-open S] [--hand left|right]\n"
      "Dry run unless --torque-mode is given. kp is torque units/deg, kd torque units/(deg/s), tau_max of 1000.\n");
}

bool parse(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto value = [&](double& out) {
      if (i + 1 >= argc) return false;
      out = std::atof(argv[++i]);
      return true;
    };
    double v = 0;
    if (k == "--port" && i + 1 < argc) a.port = argv[++i];
    else if (k == "--address" && value(v)) a.address = static_cast<int>(v);
    else if (k == "--rate" && value(v)) a.rate = v;
    else if (k == "--kp" && value(v)) a.kp = v;
    else if (k == "--kd" && value(v)) a.kd = v;
    else if (k == "--tau-max" && value(v)) a.tau_max = v;
    else if (k == "--udp-port" && value(v)) a.udp_port = static_cast<int>(v);
    else if (k == "--timeout" && value(v)) a.timeout = v;
    else if (k == "--zero-on-timeout") a.zero_on_timeout = true;
    else if (k == "--measured-velocity") a.measured_velocity = true;
    else if (k == "--velocity-window" && value(v)) a.velocity_window = v;
    else if (k == "--deadband" && value(v)) a.deadband = v;
    else if (k == "--torque-mode") a.torque_mode = true;
    else if (k == "--log" && i + 1 < argc) a.log = argv[++i];
    else if (k == "--duration" && value(v)) a.duration = v;
    else if (k == "--sleep-after-open" && value(v)) a.sleep_after_open = v;
    else if (k == "--hand" && i + 1 < argc) {
      try {
        a.hand = dexrt::parse_hand_side(argv[++i]);
      } catch (const std::invalid_argument&) {
        return false;
      }
    }
    else return false;
  }
  return !a.port.empty() && a.rate > 0 && a.tau_max >= 0 && a.tau_max <= 1000;
}

class TargetSocket {
 public:
  explicit TargetSocket(int port) {
    fd_ = ::socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (fd_ < 0) throw std::runtime_error("udp: socket failed");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      ::close(fd_);
      throw std::runtime_error("udp: bind failed on port " + std::to_string(port));
    }
  }
  ~TargetSocket() { ::close(fd_); }
  // Drains the queue; returns true if a new 16-float64 target was received.
  bool poll(Eigen::VectorXd& target) {
    bool got = false;
    double buf[16];
    for (;;) {
      const ssize_t n = ::recv(fd_, buf, sizeof(buf), 0);
      if (n < 0) break;
      if (n != static_cast<ssize_t>(sizeof(buf))) continue;
      bool finite = true;
      for (double v : buf) finite &= std::isfinite(v);
      if (!finite) continue;
      target = Eigen::Map<Eigen::VectorXd>(buf, 16);
      got = true;
    }
    return got;
  }

 private:
  int fd_ = -1;
};

Eigen::VectorXd to_eigen(const std::array<double, 16>& a) { return Eigen::Map<const Eigen::VectorXd>(a.data(), 16); }

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse(argc, argv, args)) {
    usage();
    return 2;
  }
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  dexrt::SlcanSerial bus;
  bool torque_mode_entered = false;
  int exit_code = 0;
  try {
    const dexrt::D21RegisterMap& regmap = dexrt::d21_register_map_from_env();
    bus.open(args.port, 10000000, 1000000, regmap.canfd_data_bitrate, args.sleep_after_open);
    dexrt::D21Modbus hand(bus, static_cast<uint8_t>(args.address), 0.1, regmap);
    hand.check_register_map();
    const int mode = hand.read_control_mode();
    const int enable = hand.read_hand_enable();
    std::array<double, 16> measured = hand.read_positions_deg();
    std::printf("[loop] control mode %d, hand enable %d, %s\n", mode, enable,
                args.torque_mode ? "TORQUE MODE (writes torques)" : "DRY RUN (no writes)");
    std::printf("[loop] %s hand motor-pair polarity\n", args.hand == dexrt::HandSide::Right ? "right" : "left");

    dexrt::ImpedanceConfig cfg;
    cfg.kp = Eigen::VectorXd::Constant(16, args.kp);
    cfg.kd = Eigen::VectorXd::Constant(16, args.kd);
    cfg.tau_max = Eigen::VectorXd::Constant(16, args.tau_max);
    cfg.target_timeout_s = args.timeout;
    cfg.zero_torque_on_timeout = args.zero_on_timeout;
    cfg.velocity_window_s = args.velocity_window;
    cfg.deadband = args.deadband;
    dexrt::ImpedanceController controller(cfg);
    controller.set_target(to_eigen(measured), now_s());
    TargetSocket targets(args.udp_port);
    if (args.measured_velocity) {
      std::printf("[loop] using firmware velocity registers (raw/5; unit undocumented) for damping\n");
    }

    std::ofstream log;
    if (!args.log.empty()) {
      log.open(args.log, std::ios::out | std::ios::trunc);
      log << "t,cycle_ms,fresh";
      for (int i = 0; i < 16; ++i) log << ",q" << i;
      for (int i = 0; i < 16; ++i) log << ",qt" << i;
      for (int i = 0; i < 16; ++i) log << ",qd" << i;
      for (int i = 0; i < 16; ++i) log << ",tau" << i;
      log << "\n";
    }

    if (args.torque_mode) {
      hand.write_torques(std::array<int, 16>{});  // zero before enabling torque mode
      hand.set_control_mode(2);
      torque_mode_entered = true;
      std::printf("[loop] torque mode entered; Ctrl+C restores position mode\n");
    }

    const double period = 1.0 / args.rate;
    const double start = now_s();
    double next = start + period;
    double report = start + 1.0;
    int cycles = 0, late = 0;
    double max_tau = 0, max_err = 0, max_cycle_ms = 0;
    Eigen::VectorXd new_target(16);
    std::array<int, 16> torque_cmd{};
    while (!g_stop) {
      const double t0 = now_s();
      if (args.duration > 0 && t0 - start >= args.duration) break;
      measured = hand.read_positions_deg();
      Eigen::VectorXd qdot;
      const Eigen::VectorXd* qdot_ptr = nullptr;
      if (args.measured_velocity) {
        qdot = to_eigen(hand.read_velocities_raw());
        qdot_ptr = &qdot;
      }
      if (targets.poll(new_target)) controller.set_target(new_target, t0);
      const dexrt::ImpedanceOutput out = controller.step(to_eigen(measured), qdot_ptr, t0);
      torque_cmd = dexrt::joint_to_motor_torques(out.tau, args.tau_max, args.hand);
      if (torque_mode_entered) hand.write_torques(torque_cmd);

      const double cycle_ms = (now_s() - t0) * 1000;
      ++cycles;
      max_cycle_ms = std::max(max_cycle_ms, cycle_ms);
      max_tau = std::max(max_tau, out.tau.cwiseAbs().maxCoeff());
      max_err = std::max(max_err, (controller.target() - to_eigen(measured)).cwiseAbs().maxCoeff());
      if (log.is_open()) {
        log << t0 - start << ',' << cycle_ms << ',' << (out.target_fresh ? 1 : 0);
        for (int i = 0; i < 16; ++i) log << ',' << measured[i];
        for (int i = 0; i < 16; ++i) log << ',' << controller.target()[i];
        for (int i = 0; i < 16; ++i) log << ',' << out.qdot[i];
        for (int i = 0; i < 16; ++i) log << ',' << out.tau[i];
        log << '\n';
      }
      if (t0 >= report) {
        std::printf("[loop] %5.1f s  %4d Hz  late %3d  cycle max %.2f ms  |tau| max %6.1f  |err| max %6.2f deg  %s\n",
                    t0 - start, cycles, late, max_cycle_ms, max_tau, max_err,
                    out.target_fresh ? "target fresh" : "target STALE");
        cycles = late = 0;
        max_tau = max_err = max_cycle_ms = 0;
        report = t0 + 1.0;
      }
      const double remaining = next - now_s();
      if (remaining > 0) {
        std::this_thread::sleep_for(std::chrono::duration<double>(remaining));
      } else {
        ++late;
        next = now_s();  // never try to catch up with a burst of writes
      }
      next += period;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[loop] error: %s\n", e.what());
    exit_code = 1;
  }

  if (torque_mode_entered) {
    // Best-effort safe shutdown: zero torque, position target = measured, position mode.
    try {
      dexrt::D21Modbus hand(bus, static_cast<uint8_t>(args.address), 0.2, dexrt::d21_register_map_from_env());
      hand.write_torques(std::array<int, 16>{});
      const std::array<double, 16> q = hand.read_positions_deg();
      hand.write_positions_deg(q);
      hand.set_control_mode(1);
      std::printf("[loop] zero torque written, position targets = measured pose, position mode restored\n");
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[loop] WARNING: could not restore position mode: %s\n", e.what());
      exit_code = 1;
    }
  }
  bus.close();
  return exit_code;
}
