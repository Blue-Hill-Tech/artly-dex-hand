// AAC D21 joint torques (SDK joint order) -> 16 motor torques, per hand.
//
// Each finger's abduction and flexion joints share a differential motor pair: the same sign on both motors gives
// flexion (thumb: abduction), opposite signs give abduction (thumb: rotation). The sign of the opposite-sign half
// differs per hand (v2 firmware bench, from the home pose):
//   left  (2026-11-02): every pair -1, i.e. +abduction / +thumb rotation needs motor a negative, motor b positive;
//   right (2026-11-03): thumb pair -1 like the left, the four finger abduction pairs +1.
// Same table as teleop/hand_controller_aac.py PAIR_POLARITY. Units: torque counts as the caller uses them.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace dexrt {

enum class HandSide { Left, Right };

// Per pair (thumb, index, middle, ring, pinky).
inline std::array<double, 5> pair_polarity(HandSide side) {
  if (side == HandSide::Right) return {-1.0, 1.0, 1.0, 1.0, 1.0};
  return {-1.0, -1.0, -1.0, -1.0, -1.0};
}

inline HandSide parse_hand_side(const std::string& s) {
  if (s == "left" || s == "Left" || s == "LeftHand") return HandSide::Left;
  if (s == "right" || s == "Right" || s == "RightHand") return HandSide::Right;
  throw std::invalid_argument("hand side must be left or right, not '" + s + "'");
}

// tau: any indexable container of 16 joint torques (SDK joint order).
template <typename Vec>
std::array<int, 16> joint_to_motor_torques(const Vec& tau, double limit, HandSide side = HandSide::Left) {
  // {same-sign joint, opposite-sign joint, motor a, motor b}
  constexpr int kPairs[5][4] = {{0, 1, 0, 1}, {5, 4, 4, 5}, {8, 7, 7, 8}, {11, 10, 10, 11}, {14, 13, 13, 14}};
  constexpr int kSingles[6][2] = {{2, 2}, {3, 3}, {6, 6}, {9, 9}, {12, 12}, {15, 15}};  // {joint, motor}
  const std::array<double, 5> polarity = pair_polarity(side);
  std::array<double, 16> motor{};
  for (int k = 0; k < 5; ++k) {
    const auto& p = kPairs[k];
    motor[p[2]] = tau[p[0]] + polarity[k] * tau[p[1]];
    motor[p[3]] = tau[p[0]] - polarity[k] * tau[p[1]];
  }
  for (const auto& s : kSingles) motor[s[1]] = tau[s[0]];
  std::array<int, 16> out{};
  for (int i = 0; i < 16; ++i) out[i] = static_cast<int>(std::lround(std::clamp(motor[i], -limit, limit)));
  return out;
}

}  // namespace dexrt
