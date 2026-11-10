// Minimal URDF kinematics: revolute and fixed joints only, which is all the
// AAC D21 description uses. Forward kinematics and point Jacobians in the
// URDF world frame; no dynamics, no meshes, no collision.
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <vector>

namespace dexrt {

struct UrdfJoint {
  std::string name;
  int parent_link = -1;
  int child_link = -1;
  Eigen::Isometry3d origin = Eigen::Isometry3d::Identity();
  Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
  bool revolute = false;
  int q_index = -1;  // -1 for fixed joints
  double lower = 0.0;
  double upper = 0.0;
};

struct UrdfLink {
  std::string name;
  int parent_joint = -1;          // -1 for the root link
  std::vector<int> ancestor_q;    // q indices of revolute joints between root and this link
};

// Result of one forward-kinematics pass, reused between calls to avoid allocation.
struct FkState {
  std::vector<Eigen::Isometry3d> link;    // world pose per link
  std::vector<Eigen::Vector3d> axis_w;    // per q index: rotation axis in world
  std::vector<Eigen::Vector3d> origin_w;  // per q index: joint origin in world
};

class UrdfModel {
 public:
  // Parses the file; throws std::runtime_error on malformed or unsupported input.
  static UrdfModel load(const std::string& path);

  int dof() const { return dof_; }
  int link_count() const { return static_cast<int>(links_.size()); }
  const std::string& path() const { return path_; }

  int joint_index(const std::string& name) const;  // q index; throws if unknown or fixed
  int link_index(const std::string& name) const;   // throws if unknown
  const std::vector<std::string>& joint_names() const { return q_names_; }
  const std::string& link_name(int index) const { return links_.at(index).name; }
  const UrdfJoint& joint_of_q(int q_index) const { return joints_.at(q_to_joint_.at(q_index)); }

  Eigen::VectorXd lower_limits() const;
  Eigen::VectorXd upper_limits() const;

  void forward(const Eigen::VectorXd& q, FkState& state) const;
  // Jacobian of the link origin position with respect to q, 3 x dof, fully overwritten.
  void point_jacobian(const FkState& state, int link, Eigen::Ref<Eigen::Matrix3Xd> jacobian) const;
  // Same for a point fixed on the link at `offset` (link frame, metres).
  void point_jacobian(const FkState& state, int link, const Eigen::Vector3d& offset,
                      Eigen::Ref<Eigen::Matrix3Xd> jacobian) const;

 private:
  std::vector<UrdfJoint> joints_;  // topological order from the root
  std::vector<UrdfLink> links_;
  std::vector<std::string> q_names_;
  std::vector<int> q_to_joint_;
  int dof_ = 0;
  int root_link_ = -1;
  std::string path_;
};

}  // namespace dexrt
