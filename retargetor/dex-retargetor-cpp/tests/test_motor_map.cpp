// Per-hand motor-pair polarity (v2 bench 2026-11-03): the right hand's finger abduction pairs are reversed against
// the left; its thumb pair is not. Matches teleop/hand_controller_aac.py PAIR_POLARITY.
#include <array>

#include "check.hpp"
#include "dexrt/motor_map.hpp"

using dexrt::HandSide;
using dexrt::joint_to_motor_torques;

int main() {
  std::array<double, 16> tau{};
  tau[1] = 30;                                       // thumb rotation +
  auto l = joint_to_motor_torques(tau, 200, HandSide::Left), r = joint_to_motor_torques(tau, 200, HandSide::Right);
  CHECK(l[0] == -30 && l[1] == 30);
  CHECK(r[0] == -30 && r[1] == 30);                  // thumb: same on both hands

  tau = {};
  tau[4] = 20;                                       // index abduction +
  l = joint_to_motor_torques(tau, 200, HandSide::Left);
  r = joint_to_motor_torques(tau, 200, HandSide::Right);
  CHECK(l[4] == -20 && l[5] == 20);
  CHECK(r[4] == 20 && r[5] == -20);                  // fingers: reversed on the right
  tau = {};
  tau[13] = 10;                                      // pinky abduction +
  r = joint_to_motor_torques(tau, 200, HandSide::Right);
  CHECK(r[13] == 10 && r[14] == -10);

  tau = {};
  tau[5] = 40;                                       // index MCP flexion: same sign on both motors, either hand
  r = joint_to_motor_torques(tau, 200, HandSide::Right);
  CHECK(r[4] == 40 && r[5] == 40);
  tau[3] = 500;                                      // single motor, clamped
  CHECK(joint_to_motor_torques(tau, 200, HandSide::Right)[3] == 200);
  CHECK(joint_to_motor_torques(tau, 200) == joint_to_motor_torques(tau, 200, HandSide::Left));   // default left

  CHECK(dexrt::parse_hand_side("RightHand") == HandSide::Right);
  CHECK(dexrt::parse_hand_side("left") == HandSide::Left);
  CHECK_THROWS(dexrt::parse_hand_side("both"));
  return dexrt_test::finish("motor_map");
}
