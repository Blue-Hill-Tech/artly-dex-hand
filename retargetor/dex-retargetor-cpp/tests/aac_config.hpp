// AAC D21 retargeting configuration used by the tests; mirrors
// teleop/manus/dex_vector_retargeting.py (VECTOR_TABLE, COUPLINGS, MODEL_JOINTS).
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "dexrt/retargeter.hpp"

namespace dexrt_test {

inline std::string urdf_path(const std::string& side = "Left") {
  return std::string(DEXRT_REPO_ROOT) + "/hand/aac/description/D21_" + side + ".urdf";
}

inline dexrt::RetargeterConfig aac_config() {
  dexrt::RetargeterConfig c;
  const char* fingers[] = {"thumb", "index", "middle", "ring", "pinky"};
  for (int i = 1; i <= 4; ++i) c.independent_joints.push_back("thumb_joint" + std::to_string(i));
  for (int f = 1; f < 5; ++f) {
    for (int i = 1; i <= 3; ++i) c.independent_joints.push_back(std::string(fingers[f]) + "_joint" + std::to_string(i));
  }
  c.couplings.push_back({"thumb_joint5", "thumb_joint4", 0.85});
  for (int f = 1; f < 5; ++f) c.couplings.push_back({std::string(fingers[f]) + "_joint4", std::string(fingers[f]) + "_joint3", 0.8});
  for (bool tip : {true, false}) {
    for (const char* f : fingers) {
      const std::string finger = f;
      const std::string origin = finger == "thumb" ? "thumb_pitch" : finger + "_mcp";
      const std::string task = tip ? finger + "_tip" : (finger == "thumb" ? "thumb_mcp" : finger + "_pip");
      c.vectors.push_back({origin, task});
    }
  }
  // SDK joint limits in degrees, same order as the independent joints.
  const double limits[16][2] = {{0, 100}, {-35, 35}, {0, 50}, {0, 70},
                                {-20, 20}, {0, 90}, {0, 100}, {-20, 20}, {0, 90}, {0, 100},
                                {-20, 20}, {0, 90}, {0, 100}, {-20, 20}, {0, 90}, {0, 100}};
  c.lower.resize(16);
  c.upper.resize(16);
  for (int i = 0; i < 16; ++i) {
    c.lower[i] = limits[i][0] * M_PI / 180.0;
    c.upper[i] = limits[i][1] * M_PI / 180.0;
  }
  return c;
}

// DexPilot layout (dex-retargeting DexPilotOptimizer with 5 fingers): 4 thumb->finger tip vectors
// (primary), 6 finger->finger tip vectors (secondary, gated by the thumb pairs of both fingers),
// 4 finger knuckle->tip shape vectors, the thumb base->knuckle vector. Vector = task - origin.
inline dexrt::RetargeterConfig aac_dexpilot_config() {
  dexrt::RetargeterConfig c = aac_config();
  c.vectors.clear();
  const char* tips[] = {"thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"};
  for (int i = 1; i <= 4; ++i) {  // thumb (0) - finger i: origin finger tip, task thumb tip
    dexrt::VectorTask t;
    t.origin_link = tips[i];
    t.task_link = tips[0];
    t.projection = 1;
    c.vectors.push_back(t);
  }
  for (int i = 1; i <= 3; ++i) {
    for (int j = i + 1; j <= 4; ++j) {
      dexrt::VectorTask t;
      t.origin_link = tips[j];
      t.task_link = tips[i];
      t.projection = 2;
      t.gate_a = i - 1;  // thumb-finger i
      t.gate_b = j - 1;  // thumb-finger j
      c.vectors.push_back(t);
    }
  }
  const char* bases[] = {"index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"};
  for (int i = 1; i <= 4; ++i) {
    dexrt::VectorTask t;
    t.origin_link = bases[i - 1];
    t.task_link = tips[i];
    t.weight = 15.0;  // dex-retargeting: len_proj + num_fingers
    c.vectors.push_back(t);
  }
  // The D21 thumb has 4 joints for a 3-dof tip position: keep the base->knuckle vector (low weight) so
  // the thumb posture is determined, not just its tip.
  dexrt::VectorTask prox;
  prox.origin_link = "thumb_pitch";
  prox.task_link = "thumb_mcp";
  c.vectors.push_back(prox);
  c.options.huber_delta = 0.03;
  return c;
}

}  // namespace dexrt_test
