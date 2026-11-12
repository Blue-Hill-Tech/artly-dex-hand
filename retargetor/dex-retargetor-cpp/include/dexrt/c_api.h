/* Flat C ABI for ctypes. Every function returns 0 on success; on failure it
 * returns -1 and writes a message into (err, err_len) when provided. */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DEXRT_API __attribute__((visibility("default")))

typedef struct dexrt_retargeter dexrt_retargeter;
typedef struct dexrt_impedance dexrt_impedance;

typedef struct dexrt_options {
  double huber_delta;
  double norm_delta;
  double scaling;
  int max_steps;
  double step_limit;
  double levenberg;
  double stop_step;
  double eps_abs;
  double eps_rel;
  int max_iter;
  int polish;
  /* DexPilot projection (metres / weights); used only by vectors with a projection class. */
  double project_dist;
  double escape_dist;
  double eta_primary;
  double eta_secondary;
  double projected_weight_primary;
  double projected_weight_secondary;
} dexrt_options;

typedef struct dexrt_solve_info {
  double cost;
  int steps;
  int osqp_iterations;
  int osqp_status;
  int converged;
  double solve_seconds;
} dexrt_solve_info;

DEXRT_API void dexrt_default_options(dexrt_options* options);
DEXRT_API const char* dexrt_version(void);

/* independent_joints: n_independent names. couplings: n_couplings (dependent, source, factor).
 * vectors: n_vectors (origin_link, task_link); vector_weights (or NULL = 1), vector_projection
 * (0 plain, 1 primary, 2 secondary; or NULL = all 0), vector_gate_a/b (primary vector indices gating
 * each secondary vector; NULL when no secondary vectors). origin_offsets/task_offsets: n_vectors * 3 metres in
 * the link frames, or NULL = link origins. lower/upper: n_independent radians or NULL. */
DEXRT_API dexrt_retargeter* dexrt_retargeter_create(
    const char* urdf_path,
    const char* const* independent_joints, int n_independent,
    const char* const* coupling_dependent, const char* const* coupling_source, const double* coupling_factor,
    int n_couplings,
    const char* const* origin_links, const char* const* task_links, int n_vectors,
    const double* vector_weights, const int* vector_projection, const int* vector_gate_a, const int* vector_gate_b,
    const double* origin_offsets, const double* task_offsets,
    const double* lower, const double* upper,
    const dexrt_options* options,
    char* err, size_t err_len);
DEXRT_API void dexrt_retargeter_destroy(dexrt_retargeter* handle);

DEXRT_API int dexrt_retargeter_dof(const dexrt_retargeter* handle);
DEXRT_API int dexrt_retargeter_full_dof(const dexrt_retargeter* handle);
DEXRT_API int dexrt_retargeter_vector_count(const dexrt_retargeter* handle);
DEXRT_API int dexrt_retargeter_bounds(const dexrt_retargeter* handle, double* lower, double* upper);
DEXRT_API int dexrt_retargeter_last_q(const dexrt_retargeter* handle, double* q);
/* q == NULL resets to the middle of the bounds. */
DEXRT_API int dexrt_retargeter_reset(dexrt_retargeter* handle, const double* q, char* err, size_t err_len);
/* Human -> robot length scale applied to unprojected targets (dexrt_options.scaling), changeable per session. */
DEXRT_API int dexrt_retargeter_set_scaling(dexrt_retargeter* handle, double scaling, char* err, size_t err_len);
DEXRT_API double dexrt_retargeter_scaling(const dexrt_retargeter* handle);
/* Per-joint quadratic targets (dof entries each; weight 0 = none). target == NULL clears them. */
DEXRT_API int dexrt_retargeter_set_joint_targets(dexrt_retargeter* handle, const double* target, const double* weight,
                                                 char* err, size_t err_len);
/* projected: n_vectors ints, 1 where the last solve projected that vector's target (DexPilot). */
DEXRT_API int dexrt_retargeter_projected(const dexrt_retargeter* handle, int* projected);
/* vectors: n_vectors * 3 row-major metres (URDF world frame). q_out: dof radians.
 * errors_out (n_vectors) and info may be NULL. */
DEXRT_API int dexrt_retargeter_solve(dexrt_retargeter* handle, const double* vectors, double* q_out, double* errors_out,
                                     dexrt_solve_info* info, char* err, size_t err_len);
DEXRT_API int dexrt_retargeter_full_qpos(const dexrt_retargeter* handle, const double* q, double* q_full,
                                         char* err, size_t err_len);
DEXRT_API int dexrt_retargeter_robot_vectors(const dexrt_retargeter* handle, const double* q, double* vectors,
                                             char* err, size_t err_len);
/* jacobian: 3 * dof doubles, row-major (x row, y row, z row), metres per radian. */
DEXRT_API int dexrt_retargeter_link_jacobian(const dexrt_retargeter* handle, const double* q, const char* link,
                                             double* jacobian, char* err, size_t err_len);
DEXRT_API int dexrt_retargeter_link_position(const dexrt_retargeter* handle, const double* q, const char* link,
                                             double* xyz, char* err, size_t err_len);

DEXRT_API dexrt_impedance* dexrt_impedance_create(int n, const double* kp, const double* kd, const double* tau_max,
                                                  double target_timeout_s, int zero_torque_on_timeout,
                                                  double velocity_window_s, double deadband,
                                                  char* err, size_t err_len);
DEXRT_API void dexrt_impedance_destroy(dexrt_impedance* handle);
DEXRT_API int dexrt_impedance_set_target(dexrt_impedance* handle, const double* q_target, double now,
                                         char* err, size_t err_len);
DEXRT_API int dexrt_impedance_set_gains(dexrt_impedance* handle, const double* kp, const double* kd,
                                        char* err, size_t err_len);
/* qdot_measured may be NULL. Outputs tau (n), qdot_used (n, may be NULL) and flags (may be NULL):
 * flags bit0 has_target, bit1 target_fresh, bit2 saturated. */
DEXRT_API int dexrt_impedance_step(dexrt_impedance* handle, const double* q, const double* qdot_measured, double now,
                                   double* tau, double* qdot_used, int* flags, char* err, size_t err_len);
DEXRT_API void dexrt_impedance_reset(dexrt_impedance* handle);

/* Position-mode compliance (see compliance.hpp). Arrays are 16 entries, SDK joint order, degrees. */
typedef struct dexrt_compliance dexrt_compliance;

typedef struct dexrt_compliance_config {
  double readback_sign[16];
  double lower_deg[16];
  double upper_deg[16];
  double min_idle_current_a;
  double idle_sigma;
  double gain_dps_per_a;
  double return_rate;
  double offset_max_deg;
  double error_clamp_deg;
  double velocity_gate_dps;
  double deviation_min_deg;
  double velocity_window_s;
  double max_dt_s;
  double rest_bias_tau_s;
  int direction_from_current;
  int sign_min_samples;
  double sign_min_agreement;
  double target_motion_gate_dps;
  double target_holdoff_s;
} dexrt_compliance_config;

DEXRT_API void dexrt_compliance_default_config(dexrt_compliance_config* config);
DEXRT_API dexrt_compliance* dexrt_compliance_create(const dexrt_compliance_config* config, char* err, size_t err_len);
DEXRT_API void dexrt_compliance_destroy(dexrt_compliance* handle);
DEXRT_API int dexrt_compliance_add_idle_sample(dexrt_compliance* handle, const double* currents, char* err, size_t err_len);
DEXRT_API int dexrt_compliance_finish_idle(dexrt_compliance* handle, double* idle_out, char* err, size_t err_len);
/* Outputs (16 each, any may be NULL except q_cmd): q_cmd, offset, effort, q_meas, velocity, yielding (0/1),
 * deviation (q_meas - previous command - resting bias), source (0 none, 1 displacement, 2 current sign). */
DEXRT_API int dexrt_compliance_step(dexrt_compliance* handle, const double* q_target, const double* q_read,
                                    const double* currents, double now, double* q_cmd, double* offset, double* effort,
                                    double* q_meas, double* velocity, double* yielding, double* deviation,
                                    double* source, char* err, size_t err_len);
DEXRT_API void dexrt_compliance_reset(dexrt_compliance* handle);
/* Learned drive signs, 32 values row-major (16 joints x 2 motor slots), 0 = not yet known. */
DEXRT_API int dexrt_compliance_drive_signs(const dexrt_compliance* handle, double* signs32);

#ifdef __cplusplus
}
#endif
