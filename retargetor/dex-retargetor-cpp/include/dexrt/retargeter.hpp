// Vector retargeting as iteratively reweighted least squares on a QP.
//
// Each frame linearizes the ten finger vectors around the previous solution,
// weights every vector with its Huber weight (1 inside huber_delta, delta/|e|
// outside), adds the same |q - q_last|^2 term dex-retargeting uses, and solves
// the resulting box-constrained QP with OSQP. A few Gauss-Newton steps per
// frame recover the nonlinearity. No output filter is applied.
//
// DexPilot mode (Handa et al. 2019, as in dex-retargeting's DexPilotOptimizer): vectors may
// carry a weight and a projection class. A primary vector (thumb tip -> finger tip) whose
// human length drops under project_dist is projected: its target becomes eta_primary along
// its direction and its weight projected_weight_primary, until the length exceeds escape_dist.
// A secondary vector (finger tip -> finger tip) is projected while both primary vectors it is
// gated by are projected and its own length is under project_dist. Unprojected vectors are
// the human vector times `scaling`.
#pragma once

#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

#include "dexrt/kinematics.hpp"

namespace dexrt {

struct Coupling {
  std::string dependent;  // joint driven by `source` (e.g. index_joint4)
  std::string source;     // independent joint (e.g. index_joint3)
  double factor = 1.0;    // q_dependent = factor * q_source
};

struct VectorTask {
  std::string origin_link;
  std::string task_link;
  // Points fixed on the links (link frame, metres); zero = the link origin. A vector from a link's origin to
  // an offset point on the same link is that link's axis rotated by FK, i.e. an orientation target.
  Eigen::Vector3d origin_offset = Eigen::Vector3d::Zero();
  Eigen::Vector3d task_offset = Eigen::Vector3d::Zero();
  double weight = 1.0;   // multiplies this vector's Huber term
  int projection = 0;    // 0 plain, 1 primary (projected by its own length), 2 secondary (gated)
  int gate_a = -1;       // projection 2: indices of the two primary vectors that must both be projected
  int gate_b = -1;
};

struct RetargeterOptions {
  double huber_delta = 0.02;   // metres; matches dex-retargeting's SmoothL1 beta
  double norm_delta = 4e-3;    // weight of |q - q_last|^2 (dex-retargeting norm_delta)
  double scaling = 1.0;        // human vector scaling (dex-retargeting scaling_factor)
  int max_steps = 3;           // Gauss-Newton/IRLS steps per frame
  double step_limit = 0.0;     // rad per step and joint; 0 disables
  double levenberg = 1e-6;     // extra diagonal damping for conditioning
  double stop_step = 1e-5;     // early exit when max |dq| falls below this
  double eps_abs = 1e-7;
  double eps_rel = 1e-7;
  int max_iter = 4000;
  bool polish = true;
  // DexPilot projection (metres); only vectors with projection != 0 use them.
  double project_dist = 0.03;
  double escape_dist = 0.05;
  double eta_primary = 1e-4;
  double eta_secondary = 3e-2;
  double projected_weight_primary = 200.0;
  double projected_weight_secondary = 400.0;
};

struct RetargeterConfig {
  std::vector<std::string> independent_joints;  // optimisation variables, output order
  std::vector<Coupling> couplings;
  std::vector<VectorTask> vectors;
  Eigen::VectorXd lower;  // optional extra bounds (rad) per independent joint; empty = URDF only
  Eigen::VectorXd upper;
  RetargeterOptions options;
};

struct SolveResult {
  Eigen::VectorXd q;        // independent joints, rad
  Eigen::VectorXd q_full;   // all model joints in URDF q order, rad
  Eigen::VectorXd errors;   // |robot vector - target| per vector, metres
  Eigen::VectorXd weights;  // Huber weights used in the last step
  Eigen::VectorXi projected;  // 1 where a vector's target was projected (DexPilot), per vector
  double cost = 0.0;        // Huber mean + norm_delta * |q - q_last|^2, as dex-retargeting reports it
  int steps = 0;
  int osqp_iterations = 0;  // sum over steps
  int osqp_status = 0;      // last OSQP status value
  bool converged = false;   // every QP solved to tolerance
  double solve_seconds = 0.0;
};

class VectorRetargeter {
 public:
  VectorRetargeter(const UrdfModel& model, RetargeterConfig config);
  ~VectorRetargeter();
  VectorRetargeter(const VectorRetargeter&) = delete;
  VectorRetargeter& operator=(const VectorRetargeter&) = delete;

  void reset();                                   // last_q = middle of the bounds; projection state cleared
  void reset(const Eigen::VectorXd& q_independent);
  void set_scaling(double scaling);               // human -> robot length scale applied to unprojected targets
  double scaling() const;
  // Optional per-joint targets: adds weight[i] * (q[i] - target[i])^2 to the objective for every joint with
  // weight > 0 (e.g. holding the thumb rotation joint at a fixed angle). Sizes are dof().
  void set_joint_targets(const Eigen::VectorXd& target, const Eigen::VectorXd& weight);
  void clear_joint_targets();

  // target_vectors: one row per VectorTask, in the URDF world frame, metres.
  SolveResult solve(const Eigen::Matrix<double, Eigen::Dynamic, 3>& target_vectors);

  Eigen::VectorXd full_qpos(const Eigen::VectorXd& q_independent) const;
  Eigen::Matrix<double, Eigen::Dynamic, 3> robot_vectors(const Eigen::VectorXd& q_independent) const;
  Eigen::Vector3d link_position(const Eigen::VectorXd& q_independent, const std::string& link) const;
  // d(link position)/d(independent joints), 3 x dof, couplings folded in (metres per radian).
  Eigen::Matrix<double, 3, Eigen::Dynamic> link_jacobian(const Eigen::VectorXd& q_independent,
                                                         const std::string& link) const;

  const Eigen::VectorXd& lower() const;
  const Eigen::VectorXd& upper() const;
  const Eigen::VectorXd& last_q() const;
  const Eigen::VectorXi& projected() const;   // DexPilot projection state per vector
  int dof() const;
  int vector_count() const;
  const UrdfModel& model() const;
  const RetargeterConfig& config() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace dexrt
