// Retargeter: recovers a known pose from its own vectors, respects bounds,
// tolerates an outlier vector (Huber/IRLS), and stays fast.
#include <chrono>
#include <random>

#include "aac_config.hpp"
#include "check.hpp"
#include "dexrt/retargeter.hpp"

int main() {
  const dexrt::UrdfModel model = dexrt::UrdfModel::load(dexrt_test::urdf_path());
  dexrt::RetargeterConfig config = dexrt_test::aac_config();
  dexrt::VectorRetargeter solver(model, config);
  CHECK(solver.dof() == 16);
  CHECK(solver.vector_count() == 10);
  const Eigen::VectorXd lo = solver.lower(), hi = solver.upper();
  CHECK((hi.array() > lo.array()).all());
  // Coupled distal limits tighten the source: index_joint3 <= 1.309/0.8 and <= 100 deg SDK -> min(1.745, 1.636, 1.745).
  CHECK_NEAR(hi[solver.dof() - 10], 1.309 / 0.8, 1e-9);

  std::mt19937 rng(7);
  std::uniform_real_distribution<double> unit(0.15, 0.85);

  // 1. Pose recovery: feed the vectors of a random pose repeatedly; the solution must converge to it.
  double worst_recovery_m = 0;
  for (int trial = 0; trial < 5; ++trial) {
    Eigen::VectorXd q_true(16);
    for (int i = 0; i < 16; ++i) q_true[i] = lo[i] + unit(rng) * (hi[i] - lo[i]);
    const auto target = solver.robot_vectors(q_true);
    solver.reset();
    dexrt::SolveResult result;
    for (int frame = 0; frame < 25; ++frame) result = solver.solve(target);
    CHECK(result.converged);
    CHECK((result.q.array() >= lo.array() - 1e-12).all() && (result.q.array() <= hi.array() + 1e-12).all());
    worst_recovery_m = std::max(worst_recovery_m, result.errors.maxCoeff());
  }
  // The |q - q_last| term stops exactly at the optimum only in the limit; 0.5 mm is far below the 10 mm "poor fit" line.
  CHECK(worst_recovery_m < 5e-4);
  std::printf("pose recovery worst vector error %.3e m\n", worst_recovery_m);

  // 2. One-step tracking of a slowly moving target stays close.
  {
    Eigen::VectorXd qa(16), qb(16);
    for (int i = 0; i < 16; ++i) {
      qa[i] = lo[i] + 0.3 * (hi[i] - lo[i]);
      qb[i] = lo[i] + 0.6 * (hi[i] - lo[i]);
    }
    solver.reset(qa);
    double worst = 0;
    for (int k = 1; k <= 60; ++k) {
      const double s = k / 60.0;
      const auto result = solver.solve(solver.robot_vectors(qa + s * (qb - qa)));
      worst = std::max(worst, result.errors.maxCoeff());
    }
    CHECK(worst < 2e-3);
    std::printf("tracking worst vector error %.3e m\n", worst);
  }

  // 3. Outlier robustness: corrupt the thumb tip vector; the four fingers must stay accurate.
  {
    Eigen::VectorXd q_true(16);
    for (int i = 0; i < 16; ++i) q_true[i] = lo[i] + 0.5 * (hi[i] - lo[i]);
    auto target = solver.robot_vectors(q_true);
    target.row(0) += Eigen::RowVector3d(0.08, -0.05, 0.06);  // 11 cm gross error on the thumb
    solver.reset();
    dexrt::SolveResult result;
    for (int frame = 0; frame < 25; ++frame) result = solver.solve(target);
    CHECK(result.weights[0] < 0.3);  // outlier down-weighted
    double finger_worst = 0;
    for (int i = 1; i < 10; ++i) {
      if (i == 5) continue;  // thumb proximal shares joints with the corrupted vector
      finger_worst = std::max(finger_worst, result.errors[i]);
    }
    CHECK(finger_worst < 2e-3);
    std::printf("outlier test: thumb weight %.3f, other fingers worst %.3e m\n", result.weights[0], finger_worst);
  }

  // 4. Speed: median solve well under a millisecond on this problem size.
  {
    Eigen::VectorXd q_true(16);
    for (int i = 0; i < 16; ++i) q_true[i] = lo[i] + unit(rng) * (hi[i] - lo[i]);
    const auto target = solver.robot_vectors(q_true);
    solver.reset();
    std::vector<double> times;
    for (int k = 0; k < 200; ++k) times.push_back(solver.solve(target).solve_seconds * 1000);
    std::sort(times.begin(), times.end());
    std::printf("solve time p50 %.3f ms, p95 %.3f ms, max %.3f ms\n", times[100], times[190], times.back());
    CHECK(times[100] < 2.0);
  }

  // 5. Input validation.
  Eigen::Matrix<double, Eigen::Dynamic, 3> bad(10, 3);
  bad.setZero();
  bad(2, 1) = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(solver.solve(bad));
  CHECK_THROWS(solver.solve(Eigen::Matrix<double, Eigen::Dynamic, 3>(9, 3)));
  dexrt::RetargeterConfig broken = config;
  broken.couplings.pop_back();  // pinky_joint4 unassigned
  CHECK_THROWS(dexrt::VectorRetargeter(model, broken));

  // 5. DexPilot: 15 inter-fingertip / wrist vectors, projection when the thumb closes on a finger.
  {
    dexrt::RetargeterConfig dp = dexrt_test::aac_dexpilot_config();
    dexrt::VectorRetargeter dps(model, dp);
    CHECK(dps.vector_count() == 15);
    // Pose recovery still holds with the new vector set (no projection: open hand, all pairs > escape_dist).
    Eigen::VectorXd q_open = lo + 0.2 * (hi - lo);
    auto open = dps.robot_vectors(q_open);
    bool far = true;
    for (int i = 0; i < 4; ++i) far = far && open.row(i).norm() > dp.options.escape_dist;  // thumb pairs open
    CHECK(far);
    dps.reset();
    dexrt::SolveResult r;
    for (int k = 0; k < 300; ++k) r = dps.solve(open);  // the thumb converges slowly with tip-only vectors
    CHECK(r.converged);
    CHECK((r.projected.array() == 0).all());
    CHECK((r.q - q_open).cwiseAbs().maxCoeff() < 0.01);
    // Scaling: a human hand 20% smaller is matched after set_scaling(1.25).
    dps.set_scaling(1.25);
    CHECK_NEAR(dps.scaling(), 1.25, 1e-12);
    dps.reset();
    for (int k = 0; k < 300; ++k) r = dps.solve(open / 1.25);
    CHECK((r.q - q_open).cwiseAbs().maxCoeff() < 0.01);
    dps.set_scaling(1.0);
    // Projection: thumb-index human distance 2 cm (< project_dist) -> the pair is projected, the robot
    // thumb and index tips are pulled much closer than the unprojected solve would bring them.
    auto pinch = open;
    pinch.row(0) = open.row(0).normalized() * 0.02;
    dps.reset();
    for (int k = 0; k < 40; ++k) r = dps.solve(pinch);
    CHECK(r.projected[0] == 1);
    for (int i = 1; i < 15; ++i) CHECK(r.projected[i] == 0);
    const double projected_gap = dps.robot_vectors(r.q).row(0).norm();
    dexrt::RetargeterConfig plain = dp;
    for (auto& v : plain.vectors) v.projection = 0;
    dexrt::VectorRetargeter plain_solver(model, plain);
    for (int k = 0; k < 40; ++k) r = plain_solver.solve(pinch);
    const double plain_gap = plain_solver.robot_vectors(r.q).row(0).norm();
    CHECK(projected_gap < 0.5 * plain_gap);
    // Hysteresis: opening to 4 cm (between project and escape) keeps the projection; 6 cm releases it.
    pinch.row(0) = open.row(0).normalized() * 0.04;
    r = dps.solve(pinch);
    CHECK(r.projected[0] == 1);
    pinch.row(0) = open.row(0).normalized() * 0.06;
    r = dps.solve(pinch);
    CHECK(r.projected[0] == 0);
    dps.reset();
    CHECK((dps.projected().array() == 0).all());
    // Validation: a secondary vector must be gated by primary vectors.
    dexrt::RetargeterConfig bad = dp;
    bad.vectors[4].gate_a = 14;
    CHECK_THROWS(dexrt::VectorRetargeter(model, bad));
    CHECK_THROWS(dps.set_scaling(0.0));
  }

  // 6. Joint target: a straight thumb's twist (thumb_joint2) barely moves any tracked point, so the vectors
  // alone leave it wherever it was; a weighted joint target makes the solution follow it.
  {
    dexrt::VectorRetargeter s2(model, config);
    Eigen::VectorXd q_base = lo + 0.3 * (hi - lo);
    q_base[3] = 0.0;  // thumb MCP straight
    auto track = [&](bool with_target) {
      s2.reset();
      s2.clear_joint_targets();
      double worst = 0.0;
      for (int k = 0; k <= 20; ++k) {
        Eigen::VectorXd q = q_base;
        q[1] = -0.5 + 0.05 * k;  // rotation sweep -0.5..+0.5 rad
        if (with_target) {
          Eigen::VectorXd t = Eigen::VectorXd::Zero(16), w = Eigen::VectorXd::Zero(16);
          t[1] = q[1];
          w[1] = 5.0;
          s2.set_joint_targets(t, w);
        }
        dexrt::SolveResult r;
        for (int it = 0; it < 5; ++it) r = s2.solve(s2.robot_vectors(q));
        worst = std::max(worst, std::fabs(r.q[1] - q[1]));
      }
      return worst;
    };
    const double without = track(false), with = track(true);
    CHECK(with < 0.02);
    CHECK(without > with);
    Eigen::VectorXd bad = Eigen::VectorXd::Zero(16);
    bad[0] = -1.0;
    CHECK_THROWS(s2.set_joint_targets(Eigen::VectorXd::Zero(16), bad));
    CHECK_THROWS(s2.set_joint_targets(Eigen::VectorXd::Zero(15), Eigen::VectorXd::Zero(15)));
  }
  // 7. Offset points: a vector from a link's origin to a point fixed on it is the link's axis (orientation).
  //    Its analytic Jacobian matches finite differences, and orientation targets recover the thumb twist.
  {
    dexrt::RetargeterConfig oc = config;
    dexrt::VectorTask ax, ay;
    ax.origin_link = ax.task_link = "thumb_mcp";
    ax.task_offset = Eigen::Vector3d(0.03, 0, 0);
    ay.origin_link = ay.task_link = "thumb_mcp";
    ay.task_offset = Eigen::Vector3d(0, 0.03, 0);
    ax.weight = ay.weight = 5.0;  // a 3 cm axis turns only ~0.5 mm per degree: weight it up
    oc.vectors.push_back(ax);
    oc.vectors.push_back(ay);
    dexrt::VectorRetargeter so(model, oc);
    CHECK(so.vector_count() == 12);
    Eigen::VectorXd q = lo + 0.35 * (hi - lo);
    const auto v0 = so.robot_vectors(q);
    CHECK_NEAR(v0.row(10).norm(), 0.03, 1e-12);  // a rotated axis keeps its length
    CHECK_NEAR(v0.row(11).norm(), 0.03, 1e-12);
    // Finite-difference check of d(vector 10)/dq against the solver's own linearisation via a tiny solve step:
    // compare robot_vectors at q +/- h on each thumb joint with the offset-point Jacobian.
    const dexrt::UrdfModel& mdl = so.model();
    dexrt::FkState fk;
    Eigen::Matrix3Xd jo(3, mdl.dof()), jl(3, mdl.dof());
    mdl.forward(so.full_qpos(q), fk);
    const int link = mdl.link_index("thumb_mcp");
    mdl.point_jacobian(fk, link, Eigen::Vector3d(0.03, 0, 0), jo);
    mdl.point_jacobian(fk, link, jl);
    double worst = 0.0;
    for (int j = 0; j < 4; ++j) {  // thumb joints 1-4 are independent q 0..3 and URDF q in the same order
      const double h = 1e-6;
      Eigen::VectorXd qp = q, qm = q;
      qp[j] += h;
      qm[j] -= h;
      const Eigen::Vector3d fd = (so.robot_vectors(qp).row(10) - so.robot_vectors(qm).row(10)).transpose() / (2 * h);
      const int qj = mdl.joint_index("thumb_joint" + std::to_string(j + 1));
      worst = std::max(worst, (fd - (jo.col(qj) - jl.col(qj))).norm());
    }
    CHECK(worst < 1e-6);
    // Straight thumb twisting: positions alone lose it (section 6); with the two axis vectors it is recovered.
    Eigen::VectorXd q_base = lo + 0.3 * (hi - lo);
    q_base[3] = 0.0;
    double lost = 0.0;
    so.reset();
    for (int k = 0; k <= 20; ++k) {
      Eigen::VectorXd qt = q_base;
      qt[1] = -0.5 + 0.05 * k;
      dexrt::SolveResult r;
      for (int it = 0; it < 5; ++it) r = so.solve(so.robot_vectors(qt));
      lost = std::max(lost, std::fabs(r.q[1] - qt[1]));
    }
    std::printf("orientation recovery worst %.4f rad\n", lost);
    CHECK(lost < 0.02);
  }
  return dexrt_test::finish("test_retargeter");
}
