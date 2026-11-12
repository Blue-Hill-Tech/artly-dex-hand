#include "dexrt/c_api.h"

#include <cstring>
#include <exception>
#include <string>

#include "dexrt/compliance.hpp"
#include "dexrt/impedance.hpp"
#include "dexrt/retargeter.hpp"

struct dexrt_retargeter {
  dexrt::VectorRetargeter solver;
  dexrt_retargeter(const dexrt::UrdfModel& model, dexrt::RetargeterConfig config) : solver(model, std::move(config)) {}
};

struct dexrt_impedance {
  dexrt::ImpedanceController controller;
  explicit dexrt_impedance(dexrt::ImpedanceConfig config) : controller(std::move(config)) {}
};

struct dexrt_compliance {
  dexrt::ComplianceController controller;
  explicit dexrt_compliance(const dexrt::ComplianceConfig& config) : controller(config) {}
};

namespace {

void set_error(char* err, size_t err_len, const char* message) {
  if (err == nullptr || err_len == 0) return;
  std::strncpy(err, message, err_len - 1);
  err[err_len - 1] = '\0';
}

template <typename F>
int guarded(char* err, size_t err_len, F&& f) {
  try {
    f();
    return 0;
  } catch (const std::exception& e) {
    set_error(err, err_len, e.what());
    return -1;
  } catch (...) {
    set_error(err, err_len, "unknown error");
    return -1;
  }
}

Eigen::VectorXd to_vector(const double* data, int n) { return Eigen::Map<const Eigen::VectorXd>(data, n); }

}  // namespace

extern "C" {

void dexrt_default_options(dexrt_options* o) {
  dexrt::RetargeterOptions d;
  o->huber_delta = d.huber_delta;
  o->norm_delta = d.norm_delta;
  o->scaling = d.scaling;
  o->max_steps = d.max_steps;
  o->step_limit = d.step_limit;
  o->levenberg = d.levenberg;
  o->stop_step = d.stop_step;
  o->eps_abs = d.eps_abs;
  o->eps_rel = d.eps_rel;
  o->max_iter = d.max_iter;
  o->polish = d.polish ? 1 : 0;
  o->project_dist = d.project_dist;
  o->escape_dist = d.escape_dist;
  o->eta_primary = d.eta_primary;
  o->eta_secondary = d.eta_secondary;
  o->projected_weight_primary = d.projected_weight_primary;
  o->projected_weight_secondary = d.projected_weight_secondary;
}

const char* dexrt_version(void) { return "0.1.0"; }

dexrt_retargeter* dexrt_retargeter_create(
    const char* urdf_path, const char* const* independent_joints, int n_independent,
    const char* const* coupling_dependent, const char* const* coupling_source, const double* coupling_factor,
    int n_couplings, const char* const* origin_links, const char* const* task_links, int n_vectors,
    const double* vector_weights, const int* vector_projection, const int* vector_gate_a, const int* vector_gate_b,
    const double* origin_offsets, const double* task_offsets,
    const double* lower, const double* upper, const dexrt_options* options, char* err, size_t err_len) {
  dexrt_retargeter* handle = nullptr;
  guarded(err, err_len, [&] {
    if (urdf_path == nullptr) throw std::invalid_argument("urdf_path is null");
    dexrt::RetargeterConfig config;
    for (int i = 0; i < n_independent; ++i) config.independent_joints.emplace_back(independent_joints[i]);
    for (int i = 0; i < n_couplings; ++i) {
      config.couplings.push_back({coupling_dependent[i], coupling_source[i], coupling_factor[i]});
    }
    for (int i = 0; i < n_vectors; ++i) {
      dexrt::VectorTask task;
      task.origin_link = origin_links[i];
      task.task_link = task_links[i];
      if (vector_weights != nullptr) task.weight = vector_weights[i];
      if (vector_projection != nullptr) task.projection = vector_projection[i];
      if (vector_gate_a != nullptr) task.gate_a = vector_gate_a[i];
      if (vector_gate_b != nullptr) task.gate_b = vector_gate_b[i];
      if (origin_offsets != nullptr) task.origin_offset = Eigen::Vector3d(origin_offsets + 3 * i);
      if (task_offsets != nullptr) task.task_offset = Eigen::Vector3d(task_offsets + 3 * i);
      config.vectors.push_back(task);
    }
    if (lower != nullptr && upper != nullptr) {
      config.lower = to_vector(lower, n_independent);
      config.upper = to_vector(upper, n_independent);
    }
    if (options != nullptr) {
      dexrt::RetargeterOptions& o = config.options;
      o.huber_delta = options->huber_delta;
      o.norm_delta = options->norm_delta;
      o.scaling = options->scaling;
      o.max_steps = options->max_steps;
      o.step_limit = options->step_limit;
      o.levenberg = options->levenberg;
      o.stop_step = options->stop_step;
      o.eps_abs = options->eps_abs;
      o.eps_rel = options->eps_rel;
      o.max_iter = options->max_iter;
      o.polish = options->polish != 0;
      o.project_dist = options->project_dist;
      o.escape_dist = options->escape_dist;
      o.eta_primary = options->eta_primary;
      o.eta_secondary = options->eta_secondary;
      o.projected_weight_primary = options->projected_weight_primary;
      o.projected_weight_secondary = options->projected_weight_secondary;
    }
    const dexrt::UrdfModel model = dexrt::UrdfModel::load(urdf_path);
    handle = new dexrt_retargeter(model, std::move(config));
  });
  return handle;
}

void dexrt_retargeter_destroy(dexrt_retargeter* handle) { delete handle; }

int dexrt_retargeter_dof(const dexrt_retargeter* h) { return h ? h->solver.dof() : -1; }
int dexrt_retargeter_full_dof(const dexrt_retargeter* h) { return h ? h->solver.model().dof() : -1; }
int dexrt_retargeter_vector_count(const dexrt_retargeter* h) { return h ? h->solver.vector_count() : -1; }

int dexrt_retargeter_bounds(const dexrt_retargeter* h, double* lower, double* upper) {
  if (h == nullptr || lower == nullptr || upper == nullptr) return -1;
  Eigen::Map<Eigen::VectorXd>(lower, h->solver.dof()) = h->solver.lower();
  Eigen::Map<Eigen::VectorXd>(upper, h->solver.dof()) = h->solver.upper();
  return 0;
}

int dexrt_retargeter_last_q(const dexrt_retargeter* h, double* q) {
  if (h == nullptr || q == nullptr) return -1;
  Eigen::Map<Eigen::VectorXd>(q, h->solver.dof()) = h->solver.last_q();
  return 0;
}

int dexrt_retargeter_set_scaling(dexrt_retargeter* h, double scaling, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (h == nullptr) throw std::invalid_argument("null handle");
    h->solver.set_scaling(scaling);
  });
}

double dexrt_retargeter_scaling(const dexrt_retargeter* h) { return h ? h->solver.scaling() : 0.0; }

int dexrt_retargeter_set_joint_targets(dexrt_retargeter* h, const double* target, const double* weight, char* err,
                                       size_t err_len) {
  return guarded(err, err_len, [&] {
    if (h == nullptr) throw std::invalid_argument("null handle");
    if (target == nullptr || weight == nullptr) {
      h->solver.clear_joint_targets();
    } else {
      h->solver.set_joint_targets(to_vector(target, h->solver.dof()), to_vector(weight, h->solver.dof()));
    }
  });
}

int dexrt_retargeter_projected(const dexrt_retargeter* h, int* projected) {
  if (h == nullptr || projected == nullptr) return -1;
  const Eigen::VectorXi& p = h->solver.projected();
  for (int i = 0; i < p.size(); ++i) projected[i] = p[i];
  return 0;
}

int dexrt_retargeter_reset(dexrt_retargeter* h, const double* q, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (h == nullptr) throw std::invalid_argument("null handle");
    if (q == nullptr) {
      h->solver.reset();
    } else {
      h->solver.reset(to_vector(q, h->solver.dof()));
    }
  });
}

int dexrt_retargeter_solve(dexrt_retargeter* h, const double* vectors, double* q_out, double* errors_out,
                           dexrt_solve_info* info, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (h == nullptr || vectors == nullptr || q_out == nullptr) throw std::invalid_argument("null argument");
    const int m = h->solver.vector_count();
    Eigen::Matrix<double, Eigen::Dynamic, 3> target(m, 3);
    for (int i = 0; i < m; ++i) {
      for (int k = 0; k < 3; ++k) target(i, k) = vectors[3 * i + k];
    }
    const dexrt::SolveResult result = h->solver.solve(target);
    Eigen::Map<Eigen::VectorXd>(q_out, h->solver.dof()) = result.q;
    if (errors_out != nullptr) Eigen::Map<Eigen::VectorXd>(errors_out, m) = result.errors;
    if (info != nullptr) {
      info->cost = result.cost;
      info->steps = result.steps;
      info->osqp_iterations = result.osqp_iterations;
      info->osqp_status = result.osqp_status;
      info->converged = result.converged ? 1 : 0;
      info->solve_seconds = result.solve_seconds;
    }
  });
}

int dexrt_retargeter_full_qpos(const dexrt_retargeter* h, const double* q, double* q_full, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (h == nullptr || q == nullptr || q_full == nullptr) throw std::invalid_argument("null argument");
    const Eigen::VectorXd full = h->solver.full_qpos(to_vector(q, h->solver.dof()));
    Eigen::Map<Eigen::VectorXd>(q_full, full.size()) = full;
  });
}

int dexrt_retargeter_robot_vectors(const dexrt_retargeter* h, const double* q, double* vectors, char* err,
                                   size_t err_len) {
  return guarded(err, err_len, [&] {
    if (h == nullptr || q == nullptr || vectors == nullptr) throw std::invalid_argument("null argument");
    const auto v = h->solver.robot_vectors(to_vector(q, h->solver.dof()));
    for (int i = 0; i < v.rows(); ++i) {
      for (int k = 0; k < 3; ++k) vectors[3 * i + k] = v(i, k);
    }
  });
}

int dexrt_retargeter_link_jacobian(const dexrt_retargeter* h, const double* q, const char* link, double* jacobian,
                                   char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (h == nullptr || q == nullptr || link == nullptr || jacobian == nullptr) throw std::invalid_argument("null argument");
    const auto j = h->solver.link_jacobian(to_vector(q, h->solver.dof()), link);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < j.cols(); ++c) jacobian[r * j.cols() + c] = j(r, c);
    }
  });
}

int dexrt_retargeter_link_position(const dexrt_retargeter* h, const double* q, const char* link, double* xyz,
                                   char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (h == nullptr || q == nullptr || link == nullptr || xyz == nullptr) throw std::invalid_argument("null argument");
    const Eigen::Vector3d p = h->solver.link_position(to_vector(q, h->solver.dof()), link);
    xyz[0] = p.x();
    xyz[1] = p.y();
    xyz[2] = p.z();
  });
}

dexrt_impedance* dexrt_impedance_create(int n, const double* kp, const double* kd, const double* tau_max,
                                        double target_timeout_s, int zero_torque_on_timeout, double velocity_window_s,
                                        double deadband, char* err, size_t err_len) {
  dexrt_impedance* handle = nullptr;
  guarded(err, err_len, [&] {
    if (n <= 0 || kp == nullptr || kd == nullptr || tau_max == nullptr) throw std::invalid_argument("invalid gains");
    dexrt::ImpedanceConfig config;
    config.kp = to_vector(kp, n);
    config.kd = to_vector(kd, n);
    config.tau_max = to_vector(tau_max, n);
    config.target_timeout_s = target_timeout_s;
    config.zero_torque_on_timeout = zero_torque_on_timeout != 0;
    config.velocity_window_s = velocity_window_s;
    config.deadband = deadband;
    handle = new dexrt_impedance(std::move(config));
  });
  return handle;
}

void dexrt_impedance_destroy(dexrt_impedance* handle) { delete handle; }

int dexrt_impedance_set_target(dexrt_impedance* h, const double* q_target, double now, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (h == nullptr || q_target == nullptr) throw std::invalid_argument("null argument");
    h->controller.set_target(to_vector(q_target, h->controller.size()), now);
  });
}

int dexrt_impedance_set_gains(dexrt_impedance* h, const double* kp, const double* kd, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (h == nullptr || kp == nullptr || kd == nullptr) throw std::invalid_argument("null argument");
    const int n = h->controller.size();
    h->controller.set_gains(to_vector(kp, n), to_vector(kd, n));
  });
}

int dexrt_impedance_step(dexrt_impedance* h, const double* q, const double* qdot_measured, double now, double* tau,
                         double* qdot_used, int* flags, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (h == nullptr || q == nullptr || tau == nullptr) throw std::invalid_argument("null argument");
    const int n = h->controller.size();
    const Eigen::VectorXd qv = to_vector(q, n);
    Eigen::VectorXd qdot;
    if (qdot_measured != nullptr) qdot = to_vector(qdot_measured, n);
    const dexrt::ImpedanceOutput out = h->controller.step(qv, qdot_measured ? &qdot : nullptr, now);
    Eigen::Map<Eigen::VectorXd>(tau, n) = out.tau;
    if (qdot_used != nullptr) Eigen::Map<Eigen::VectorXd>(qdot_used, n) = out.qdot;
    if (flags != nullptr) *flags = (out.has_target ? 1 : 0) | (out.target_fresh ? 2 : 0) | (out.saturated ? 4 : 0);
  });
}

void dexrt_impedance_reset(dexrt_impedance* h) {
  if (h != nullptr) h->controller.reset();
}

void dexrt_compliance_default_config(dexrt_compliance_config* out) {
  const dexrt::ComplianceConfig d;
  for (int j = 0; j < dexrt::kJoints; ++j) {
    out->readback_sign[j] = d.readback_sign[j];
    out->lower_deg[j] = d.lower_deg[j];
    out->upper_deg[j] = d.upper_deg[j];
  }
  out->min_idle_current_a = d.min_idle_current_a;
  out->idle_sigma = d.idle_sigma;
  out->gain_dps_per_a = d.gain_dps_per_a;
  out->return_rate = d.return_rate;
  out->offset_max_deg = d.offset_max_deg;
  out->error_clamp_deg = d.error_clamp_deg;
  out->velocity_gate_dps = d.velocity_gate_dps;
  out->deviation_min_deg = d.deviation_min_deg;
  out->velocity_window_s = d.velocity_window_s;
  out->max_dt_s = d.max_dt_s;
  out->rest_bias_tau_s = d.rest_bias_tau_s;
  out->direction_from_current = d.direction_from_current ? 1 : 0;
  out->sign_min_samples = d.sign_min_samples;
  out->sign_min_agreement = d.sign_min_agreement;
  out->target_motion_gate_dps = d.target_motion_gate_dps;
  out->target_holdoff_s = d.target_holdoff_s;
}

dexrt_compliance* dexrt_compliance_create(const dexrt_compliance_config* in, char* err, size_t err_len) {
  dexrt_compliance* handle = nullptr;
  guarded(err, err_len, [&] {
    dexrt::ComplianceConfig c;
    if (in != nullptr) {
      for (int j = 0; j < dexrt::kJoints; ++j) {
        c.readback_sign[j] = in->readback_sign[j];
        c.lower_deg[j] = in->lower_deg[j];
        c.upper_deg[j] = in->upper_deg[j];
      }
      c.min_idle_current_a = in->min_idle_current_a;
      c.idle_sigma = in->idle_sigma;
      c.gain_dps_per_a = in->gain_dps_per_a;
      c.return_rate = in->return_rate;
      c.offset_max_deg = in->offset_max_deg;
      c.error_clamp_deg = in->error_clamp_deg;
      c.velocity_gate_dps = in->velocity_gate_dps;
      c.deviation_min_deg = in->deviation_min_deg;
      c.velocity_window_s = in->velocity_window_s;
      c.max_dt_s = in->max_dt_s;
      c.rest_bias_tau_s = in->rest_bias_tau_s;
      c.direction_from_current = in->direction_from_current != 0;
      c.sign_min_samples = in->sign_min_samples;
      c.sign_min_agreement = in->sign_min_agreement;
      c.target_motion_gate_dps = in->target_motion_gate_dps;
      c.target_holdoff_s = in->target_holdoff_s;
    }
    handle = new dexrt_compliance(c);
  });
  return handle;
}

void dexrt_compliance_destroy(dexrt_compliance* handle) { delete handle; }

int dexrt_compliance_add_idle_sample(dexrt_compliance* h, const double* currents, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (h == nullptr || currents == nullptr) throw std::invalid_argument("null argument");
    h->controller.add_idle_sample(to_vector(currents, dexrt::kJoints));
  });
}

int dexrt_compliance_finish_idle(dexrt_compliance* h, double* idle_out, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (h == nullptr) throw std::invalid_argument("null handle");
    h->controller.finish_idle_calibration();
    if (idle_out != nullptr) Eigen::Map<Eigen::VectorXd>(idle_out, dexrt::kJoints) = h->controller.idle_current();
  });
}

int dexrt_compliance_step(dexrt_compliance* h, const double* q_target, const double* q_read, const double* currents,
                          double now, double* q_cmd, double* offset, double* effort, double* q_meas, double* velocity,
                          double* yielding, double* deviation, double* source, char* err, size_t err_len) {
  return guarded(err, err_len, [&] {
    if (h == nullptr || q_target == nullptr || q_read == nullptr || currents == nullptr || q_cmd == nullptr) {
      throw std::invalid_argument("null argument");
    }
    const int n = dexrt::kJoints;
    const dexrt::ComplianceOutput out =
        h->controller.step(to_vector(q_target, n), to_vector(q_read, n), to_vector(currents, n), now);
    Eigen::Map<Eigen::VectorXd>(q_cmd, n) = out.q_cmd;
    if (offset != nullptr) Eigen::Map<Eigen::VectorXd>(offset, n) = out.offset;
    if (effort != nullptr) Eigen::Map<Eigen::VectorXd>(effort, n) = out.effort;
    if (q_meas != nullptr) Eigen::Map<Eigen::VectorXd>(q_meas, n) = out.q_meas;
    if (velocity != nullptr) Eigen::Map<Eigen::VectorXd>(velocity, n) = out.velocity;
    if (yielding != nullptr) Eigen::Map<Eigen::VectorXd>(yielding, n) = out.yielding;
    if (deviation != nullptr) Eigen::Map<Eigen::VectorXd>(deviation, n) = out.deviation;
    if (source != nullptr) Eigen::Map<Eigen::VectorXd>(source, n) = out.source;
  });
}

void dexrt_compliance_reset(dexrt_compliance* h) {
  if (h != nullptr) h->controller.reset();
}

int dexrt_compliance_drive_signs(const dexrt_compliance* h, double* signs32) {
  if (h == nullptr || signs32 == nullptr) return -1;
  const Eigen::MatrixXd s = h->controller.drive_signs();
  for (int j = 0; j < dexrt::kJoints; ++j) {
    signs32[2 * j] = s(j, 0);
    signs32[2 * j + 1] = s(j, 1);
  }
  return 0;
}

}  // extern "C"
