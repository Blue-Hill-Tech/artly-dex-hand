#include "dexrt/retargeter.hpp"

#include <osqp.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace dexrt {

struct VectorRetargeter::Impl {
  UrdfModel model;
  RetargeterConfig config;
  int n = 0;                       // independent joints
  int m = 0;                       // vectors
  std::vector<int> ind_q;          // independent joint -> model q index
  struct CoupledJoint { int dep_q; int source; double factor; };
  std::vector<CoupledJoint> coupled;
  std::vector<int> origin_link, task_link;
  Eigen::VectorXd lower, upper, last_q;
  Eigen::VectorXi projected;       // DexPilot projection state per vector (sticky between frames)
  Eigen::VectorXd joint_target, joint_weight;  // per independent joint; weight 0 = no target
  Eigen::VectorXd task_weight;
  Eigen::Matrix<double, Eigen::Dynamic, 3> target;  // per-frame targets after scaling/projection

  // Work buffers.
  FkState fk;
  Eigen::Matrix3Xd j_full_a, j_full_b;
  std::vector<Eigen::Matrix<double, 3, Eigen::Dynamic>> j_ind;  // per vector, 3 x n
  Eigen::Matrix<double, Eigen::Dynamic, 3> e;                    // per vector error
  Eigen::MatrixXd P;
  Eigen::VectorXd q_lin, l, u, dq, weights, zeros;

  // OSQP.
  OSQPSolver* solver = nullptr;
  OSQPSettings settings{};
  OSQPCscMatrix P_csc{}, A_csc{};
  std::vector<OSQPFloat> Px, Ax;
  std::vector<OSQPInt> Pi, Pp, Ai, Ap;

  Impl(const UrdfModel& m_, RetargeterConfig c_) : model(m_), config(std::move(c_)) { build(); }
  ~Impl() {
    if (solver != nullptr) osqp_cleanup(solver);
  }

  void build() {
    n = static_cast<int>(config.independent_joints.size());
    m = static_cast<int>(config.vectors.size());
    if (n == 0) throw std::invalid_argument("retargeter: no independent joints");
    if (m == 0) throw std::invalid_argument("retargeter: no vector tasks");
    if (config.options.huber_delta <= 0) throw std::invalid_argument("retargeter: huber_delta must be positive");
    if (config.options.max_steps < 1) throw std::invalid_argument("retargeter: max_steps must be >= 1");

    std::vector<int> assigned(model.dof(), 0);
    ind_q.resize(n);
    for (int i = 0; i < n; ++i) {
      ind_q[i] = model.joint_index(config.independent_joints[i]);
      if (assigned[ind_q[i]]++) throw std::invalid_argument("retargeter: duplicate independent joint");
    }
    for (const auto& c : config.couplings) {
      CoupledJoint cj;
      cj.dep_q = model.joint_index(c.dependent);
      cj.factor = c.factor;
      cj.source = -1;
      for (int i = 0; i < n; ++i) {
        if (config.independent_joints[i] == c.source) cj.source = i;
      }
      if (cj.source < 0) throw std::invalid_argument("retargeter: coupling source is not an independent joint");
      if (!(cj.factor > 0)) throw std::invalid_argument("retargeter: coupling factor must be positive");
      if (assigned[cj.dep_q]++) throw std::invalid_argument("retargeter: joint assigned twice");
      coupled.push_back(cj);
    }
    for (int k = 0; k < model.dof(); ++k) {
      if (!assigned[k]) {
        throw std::invalid_argument("retargeter: model joint '" + model.joint_names()[k] +
                                    "' is neither independent nor coupled");
      }
    }
    origin_link.resize(m);
    task_link.resize(m);
    task_weight.resize(m);
    for (int i = 0; i < m; ++i) {
      const VectorTask& t = config.vectors[i];
      origin_link[i] = model.link_index(t.origin_link);
      task_link[i] = model.link_index(t.task_link);
      if (!(t.weight > 0)) throw std::invalid_argument("retargeter: vector weight must be positive");
      task_weight[i] = t.weight;
      if (t.projection < 0 || t.projection > 2) throw std::invalid_argument("retargeter: projection must be 0, 1 or 2");
      if (t.projection == 2) {
        for (int g : {t.gate_a, t.gate_b}) {
          if (g < 0 || g >= m || config.vectors[g].projection != 1) {
            throw std::invalid_argument("retargeter: a secondary vector must be gated by two primary vectors");
          }
        }
      }
    }
    if (config.options.project_dist <= 0 || config.options.escape_dist < config.options.project_dist) {
      throw std::invalid_argument("retargeter: need 0 < project_dist <= escape_dist");
    }
    projected = Eigen::VectorXi::Zero(m);
    joint_target = Eigen::VectorXd::Zero(n);
    joint_weight = Eigen::VectorXd::Zero(n);

    // Effective bounds: URDF, optional SDK bounds, and coupled-joint limits mapped back to the source.
    const Eigen::VectorXd urdf_lower = model.lower_limits(), urdf_upper = model.upper_limits();
    lower.resize(n);
    upper.resize(n);
    for (int i = 0; i < n; ++i) {
      lower[i] = urdf_lower[ind_q[i]];
      upper[i] = urdf_upper[ind_q[i]];
    }
    if (config.lower.size() != 0 || config.upper.size() != 0) {
      if (config.lower.size() != n || config.upper.size() != n) {
        throw std::invalid_argument("retargeter: lower/upper must have one entry per independent joint");
      }
      lower = lower.cwiseMax(config.lower);
      upper = upper.cwiseMin(config.upper);
    }
    for (const auto& cj : coupled) {
      lower[cj.source] = std::max(lower[cj.source], urdf_lower[cj.dep_q] / cj.factor);
      upper[cj.source] = std::min(upper[cj.source], urdf_upper[cj.dep_q] / cj.factor);
    }
    for (int i = 0; i < n; ++i) {
      if (!std::isfinite(lower[i]) || !std::isfinite(upper[i]) || !(lower[i] < upper[i])) {
        throw std::invalid_argument("retargeter: empty effective bounds on " + config.independent_joints[i]);
      }
    }

    j_full_a.resize(3, model.dof());
    j_full_b.resize(3, model.dof());
    j_ind.assign(m, Eigen::Matrix<double, 3, Eigen::Dynamic>(3, n));
    e.resize(m, 3);
    P.resize(n, n);
    q_lin.resize(n);
    l.resize(n);
    u.resize(n);
    dq.resize(n);
    weights.resize(m);
    target.resize(m, 3);
    zeros = Eigen::VectorXd::Zero(n);
    setup_osqp();
    last_q = 0.5 * (lower + upper);
  }

  void setup_osqp() {
    // P: dense upper triangle in CSC; A: identity (box constraints).
    const int nnz = n * (n + 1) / 2;
    Px.assign(nnz, 0.0);
    Pi.resize(nnz);
    Pp.resize(n + 1);
    int k = 0;
    for (int j = 0; j < n; ++j) {
      Pp[j] = k;
      for (int i = 0; i <= j; ++i) Pi[k++] = i;
    }
    Pp[n] = k;
    Ax.assign(n, 1.0);
    Ai.resize(n);
    Ap.resize(n + 1);
    for (int j = 0; j < n; ++j) {
      Ai[j] = j;
      Ap[j] = j;
    }
    Ap[n] = n;
    OSQPCscMatrix_set_data(&P_csc, n, n, nnz, Px.data(), Pi.data(), Pp.data());
    OSQPCscMatrix_set_data(&A_csc, n, n, n, Ax.data(), Ai.data(), Ap.data());

    osqp_set_default_settings(&settings);
    settings.verbose = 0;
    settings.eps_abs = config.options.eps_abs;
    settings.eps_rel = config.options.eps_rel;
    settings.max_iter = config.options.max_iter;
    settings.polishing = config.options.polish ? 1 : 0;
    settings.warm_starting = 1;

    // Any positive-definite placeholder keeps setup happy; real data arrives per solve.
    for (int j = 0; j < n; ++j) Px[Pp[j + 1] - 1] = 1.0;
    q_lin.setZero();
    l.setConstant(-1.0);
    u.setConstant(1.0);
    const OSQPInt status = osqp_setup(&solver, &P_csc, q_lin.data(), &A_csc, l.data(), u.data(), n, n, &settings);
    if (status != 0 || solver == nullptr) throw std::runtime_error("retargeter: osqp_setup failed");
  }

  Eigen::VectorXd full(const Eigen::VectorXd& q) const {
    if (q.size() != n) throw std::invalid_argument("retargeter: q has wrong size");
    Eigen::VectorXd out = Eigen::VectorXd::Zero(model.dof());
    for (int i = 0; i < n; ++i) out[ind_q[i]] = q[i];
    for (const auto& cj : coupled) out[cj.dep_q] = cj.factor * q[cj.source];
    return out;
  }

  // Reduce a full 3 x dof Jacobian to the independent variables.
  void reduce(const Eigen::Matrix3Xd& full_j, Eigen::Ref<Eigen::Matrix<double, 3, Eigen::Dynamic>> out) const {
    for (int i = 0; i < n; ++i) out.col(i) = full_j.col(ind_q[i]);
    for (const auto& cj : coupled) out.col(cj.source) += cj.factor * full_j.col(cj.dep_q);
  }

  // DexPilot: update the sticky projection flags from the human (unscaled) vector lengths and build
  // this frame's targets. Primary vectors have hysteresis; secondary ones follow their gates.
  void build_targets(const Eigen::Matrix<double, Eigen::Dynamic, 3>& raw) {
    const RetargeterOptions& o = config.options;
    for (int i = 0; i < m; ++i) {
      if (config.vectors[i].projection != 1) continue;
      const double d = raw.row(i).norm();
      if (d < o.project_dist) projected[i] = 1;
      else if (d > o.escape_dist) projected[i] = 0;
    }
    for (int i = 0; i < m; ++i) {
      const VectorTask& t = config.vectors[i];
      if (t.projection == 2) {
        projected[i] = (projected[t.gate_a] && projected[t.gate_b] && raw.row(i).norm() <= o.project_dist) ? 1 : 0;
      }
      if (t.projection != 0 && projected[i]) {
        const double eta = t.projection == 1 ? o.eta_primary : o.eta_secondary;
        target.row(i) = raw.row(i) / (raw.row(i).norm() + 1e-6) * eta;
      } else {
        target.row(i) = raw.row(i) * o.scaling;
      }
    }
  }

  double effective_weight(int i) const {
    const VectorTask& t = config.vectors[i];
    if (t.projection != 0 && projected[i]) {
      return t.projection == 1 ? config.options.projected_weight_primary : config.options.projected_weight_secondary;
    }
    return task_weight[i];
  }

  // Robot vectors and (optionally) reduced Jacobians at q.
  void evaluate(const Eigen::VectorXd& q, const Eigen::Matrix<double, Eigen::Dynamic, 3>& target, bool with_jacobian) {
    model.forward(full(q), fk);
    for (int i = 0; i < m; ++i) {
      const VectorTask& t = config.vectors[i];
      const Eigen::Vector3d v = fk.link[task_link[i]] * t.task_offset - fk.link[origin_link[i]] * t.origin_offset;
      e.row(i) = (v - target.row(i).transpose()).transpose();
      if (with_jacobian) {
        model.point_jacobian(fk, task_link[i], t.task_offset, j_full_a);
        model.point_jacobian(fk, origin_link[i], t.origin_offset, j_full_b);
        j_full_a -= j_full_b;
        reduce(j_full_a, j_ind[i]);
      }
    }
  }

  double huber_mean() const {
    const double d = config.options.huber_delta;
    double total = 0.0;
    for (int i = 0; i < m; ++i) {
      const double r = e.row(i).norm();
      total += effective_weight(i) * (r <= d ? 0.5 * r * r / d : r - 0.5 * d);
    }
    return total / m;
  }

  SolveResult solve(const Eigen::Matrix<double, Eigen::Dynamic, 3>& raw_target) {
    if (raw_target.rows() != m) throw std::invalid_argument("retargeter: expected one target vector per task");
    if (!raw_target.allFinite()) throw std::invalid_argument("retargeter: target vectors must be finite");
    const auto started = std::chrono::steady_clock::now();
    const RetargeterOptions& o = config.options;
    build_targets(raw_target);
    const double c = 1.0 / (m * o.huber_delta);

    SolveResult result;
    result.converged = true;
    Eigen::VectorXd q = last_q;
    for (int step = 0; step < o.max_steps; ++step) {
      evaluate(q, target, true);
      P.setZero();
      q_lin = 2.0 * o.norm_delta * (q - last_q);
      for (int i = 0; i < m; ++i) {
        const double r = e.row(i).norm();
        weights[i] = r <= o.huber_delta ? 1.0 : o.huber_delta / r;
        const double w = c * weights[i] * effective_weight(i);
        P.noalias() += w * j_ind[i].transpose() * j_ind[i];
        q_lin.noalias() += w * j_ind[i].transpose() * e.row(i).transpose();
      }
      P.diagonal().array() += 2.0 * o.norm_delta + o.levenberg;
      for (int i = 0; i < n; ++i) {
        if (joint_weight[i] > 0) {  // weight * (q + dq - target)^2
          P(i, i) += 2.0 * joint_weight[i];
          q_lin[i] += 2.0 * joint_weight[i] * (q[i] - joint_target[i]);
        }
      }
      for (int j = 0, k = 0; j < n; ++j) {
        for (int i = 0; i <= j; ++i) Px[k++] = P(i, j);
      }
      l = lower - q;
      u = upper - q;
      if (o.step_limit > 0) {
        l = l.cwiseMax(-o.step_limit);
        u = u.cwiseMin(o.step_limit);
      }
      if (osqp_update_data_mat(solver, Px.data(), nullptr, static_cast<OSQPInt>(Px.size()), nullptr, nullptr, 0) != 0 ||
          osqp_update_data_vec(solver, q_lin.data(), l.data(), u.data()) != 0) {
        throw std::runtime_error("retargeter: OSQP data update failed");
      }
      osqp_warm_start(solver, zeros.data(), nullptr);
      if (osqp_solve(solver) != 0) throw std::runtime_error("retargeter: osqp_solve failed");
      result.osqp_status = static_cast<int>(solver->info->status_val);
      result.osqp_iterations += static_cast<int>(solver->info->iter);
      if (result.osqp_status != OSQP_SOLVED && result.osqp_status != OSQP_SOLVED_INACCURATE) {
        result.converged = false;
        break;
      }
      for (int i = 0; i < n; ++i) dq[i] = solver->solution->x[i];
      if (!dq.allFinite()) throw std::runtime_error("retargeter: OSQP returned a non-finite step");
      q = (q + dq).cwiseMax(lower).cwiseMin(upper);
      ++result.steps;
      if (dq.cwiseAbs().maxCoeff() < o.stop_step) break;
    }
    evaluate(q, target, false);
    result.errors.resize(m);
    for (int i = 0; i < m; ++i) result.errors[i] = e.row(i).norm();
    result.cost = huber_mean() + o.norm_delta * (q - last_q).squaredNorm() +
                  (joint_weight.array() * (q - joint_target).array().square()).sum();
    result.weights = weights;
    result.projected = projected;
    result.q = q;
    result.q_full = full(q);
    last_q = q;
    result.solve_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
  }
};

VectorRetargeter::VectorRetargeter(const UrdfModel& model, RetargeterConfig config)
    : impl_(std::make_unique<Impl>(model, std::move(config))) {}
VectorRetargeter::~VectorRetargeter() = default;

void VectorRetargeter::reset() {
  impl_->last_q = 0.5 * (impl_->lower + impl_->upper);
  impl_->projected.setZero();
}
void VectorRetargeter::set_scaling(double scaling) {
  if (!(scaling > 0) || !std::isfinite(scaling)) throw std::invalid_argument("retargeter: scaling must be positive");
  impl_->config.options.scaling = scaling;
}
double VectorRetargeter::scaling() const { return impl_->config.options.scaling; }
void VectorRetargeter::set_joint_targets(const Eigen::VectorXd& target, const Eigen::VectorXd& weight) {
  if (target.size() != impl_->n || weight.size() != impl_->n) {
    throw std::invalid_argument("retargeter: joint targets need one entry per independent joint");
  }
  if (!target.allFinite() || !weight.allFinite() || (weight.array() < 0).any()) {
    throw std::invalid_argument("retargeter: joint targets must be finite with non-negative weights");
  }
  impl_->joint_target = target;
  impl_->joint_weight = weight;
}
void VectorRetargeter::clear_joint_targets() {
  impl_->joint_target.setZero();
  impl_->joint_weight.setZero();
}
const Eigen::VectorXi& VectorRetargeter::projected() const { return impl_->projected; }

void VectorRetargeter::reset(const Eigen::VectorXd& q) {
  if (q.size() != impl_->n) throw std::invalid_argument("retargeter: reset with wrong size");
  if (!q.allFinite()) throw std::invalid_argument("retargeter: reset with non-finite values");
  impl_->last_q = q.cwiseMax(impl_->lower).cwiseMin(impl_->upper);
  impl_->projected.setZero();
}

SolveResult VectorRetargeter::solve(const Eigen::Matrix<double, Eigen::Dynamic, 3>& target) { return impl_->solve(target); }
Eigen::VectorXd VectorRetargeter::full_qpos(const Eigen::VectorXd& q) const { return impl_->full(q); }

Eigen::Matrix<double, Eigen::Dynamic, 3> VectorRetargeter::robot_vectors(const Eigen::VectorXd& q) const {
  FkState fk;
  impl_->model.forward(impl_->full(q), fk);
  Eigen::Matrix<double, Eigen::Dynamic, 3> out(impl_->m, 3);
  for (int i = 0; i < impl_->m; ++i) {
    const VectorTask& t = impl_->config.vectors[i];
    out.row(i) = (fk.link[impl_->task_link[i]] * t.task_offset - fk.link[impl_->origin_link[i]] * t.origin_offset).transpose();
  }
  return out;
}

Eigen::Matrix<double, 3, Eigen::Dynamic> VectorRetargeter::link_jacobian(const Eigen::VectorXd& q,
                                                                         const std::string& link) const {
  FkState fk;
  impl_->model.forward(impl_->full(q), fk);
  Eigen::Matrix3Xd full(3, impl_->model.dof());
  impl_->model.point_jacobian(fk, impl_->model.link_index(link), full);
  Eigen::Matrix<double, 3, Eigen::Dynamic> out(3, impl_->n);
  impl_->reduce(full, out);
  return out;
}

Eigen::Vector3d VectorRetargeter::link_position(const Eigen::VectorXd& q, const std::string& link) const {
  FkState fk;
  impl_->model.forward(impl_->full(q), fk);
  return fk.link[impl_->model.link_index(link)].translation();
}

const Eigen::VectorXd& VectorRetargeter::lower() const { return impl_->lower; }
const Eigen::VectorXd& VectorRetargeter::upper() const { return impl_->upper; }
const Eigen::VectorXd& VectorRetargeter::last_q() const { return impl_->last_q; }
int VectorRetargeter::dof() const { return impl_->n; }
int VectorRetargeter::vector_count() const { return impl_->m; }
const UrdfModel& VectorRetargeter::model() const { return impl_->model; }
const RetargeterConfig& VectorRetargeter::config() const { return impl_->config; }

}  // namespace dexrt
