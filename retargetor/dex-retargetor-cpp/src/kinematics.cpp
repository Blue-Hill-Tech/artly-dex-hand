#include "dexrt/kinematics.hpp"

#include <tinyxml2.h>

#include <cmath>
#include <deque>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace dexrt {
namespace {

Eigen::Vector3d parse_xyz(const char* text, const Eigen::Vector3d& fallback) {
  if (text == nullptr) return fallback;
  std::istringstream in(text);
  Eigen::Vector3d v;
  if (!(in >> v.x() >> v.y() >> v.z())) throw std::runtime_error("URDF: malformed xyz/rpy triple");
  return v;
}

Eigen::Isometry3d parse_origin(const tinyxml2::XMLElement* element) {
  Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
  if (element == nullptr) return t;
  const Eigen::Vector3d xyz = parse_xyz(element->Attribute("xyz"), Eigen::Vector3d::Zero());
  const Eigen::Vector3d rpy = parse_xyz(element->Attribute("rpy"), Eigen::Vector3d::Zero());
  // URDF convention: R = Rz(yaw) * Ry(pitch) * Rx(roll).
  t.linear() = (Eigen::AngleAxisd(rpy.z(), Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(rpy.y(), Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(rpy.x(), Eigen::Vector3d::UnitX())).toRotationMatrix();
  t.translation() = xyz;
  return t;
}

}  // namespace

UrdfModel UrdfModel::load(const std::string& path) {
  tinyxml2::XMLDocument doc;
  if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error("URDF: cannot read " + path + ": " + doc.ErrorStr());
  }
  const tinyxml2::XMLElement* robot = doc.FirstChildElement("robot");
  if (robot == nullptr) throw std::runtime_error("URDF: missing <robot> element");

  UrdfModel model;
  model.path_ = path;
  std::unordered_map<std::string, int> link_ids;
  for (auto* e = robot->FirstChildElement("link"); e != nullptr; e = e->NextSiblingElement("link")) {
    const char* name = e->Attribute("name");
    if (name == nullptr) throw std::runtime_error("URDF: <link> without name");
    if (link_ids.count(name)) throw std::runtime_error(std::string("URDF: duplicate link ") + name);
    link_ids[name] = static_cast<int>(model.links_.size());
    model.links_.push_back(UrdfLink{name, -1, {}});
  }

  std::vector<UrdfJoint> unordered;
  for (auto* e = robot->FirstChildElement("joint"); e != nullptr; e = e->NextSiblingElement("joint")) {
    UrdfJoint joint;
    const char* name = e->Attribute("name");
    const char* type = e->Attribute("type");
    if (name == nullptr || type == nullptr) throw std::runtime_error("URDF: <joint> without name/type");
    joint.name = name;
    const std::string kind = type;
    if (kind == "revolute") {
      joint.revolute = true;
    } else if (kind == "fixed") {
      joint.revolute = false;
    } else {
      throw std::runtime_error("URDF: unsupported joint type '" + kind + "' on " + joint.name);
    }
    const auto* parent = e->FirstChildElement("parent");
    const auto* child = e->FirstChildElement("child");
    if (parent == nullptr || child == nullptr || parent->Attribute("link") == nullptr ||
        child->Attribute("link") == nullptr) {
      throw std::runtime_error("URDF: joint " + joint.name + " lacks parent/child");
    }
    auto parent_it = link_ids.find(parent->Attribute("link"));
    auto child_it = link_ids.find(child->Attribute("link"));
    if (parent_it == link_ids.end() || child_it == link_ids.end()) {
      throw std::runtime_error("URDF: joint " + joint.name + " references an unknown link");
    }
    joint.parent_link = parent_it->second;
    joint.child_link = child_it->second;
    joint.origin = parse_origin(e->FirstChildElement("origin"));
    if (joint.revolute) {
      const auto* axis = e->FirstChildElement("axis");
      joint.axis = parse_xyz(axis ? axis->Attribute("xyz") : nullptr, Eigen::Vector3d::UnitZ());
      const double norm = joint.axis.norm();
      if (!(norm > 0.0)) throw std::runtime_error("URDF: zero axis on joint " + joint.name);
      joint.axis /= norm;
      const auto* limit = e->FirstChildElement("limit");
      if (limit == nullptr || limit->QueryDoubleAttribute("lower", &joint.lower) != tinyxml2::XML_SUCCESS ||
          limit->QueryDoubleAttribute("upper", &joint.upper) != tinyxml2::XML_SUCCESS) {
        throw std::runtime_error("URDF: revolute joint " + joint.name + " needs <limit lower upper>");
      }
      if (!(joint.lower < joint.upper)) throw std::runtime_error("URDF: empty limits on joint " + joint.name);
    }
    if (e->FirstChildElement("mimic") != nullptr) {
      throw std::runtime_error("URDF: <mimic> is not supported; express couplings in the retargeter config");
    }
    unordered.push_back(joint);
  }

  // Root: the one link that is never a child.
  std::vector<bool> is_child(model.links_.size(), false);
  for (const auto& joint : unordered) {
    if (is_child[joint.child_link]) throw std::runtime_error("URDF: link has two parent joints");
    is_child[joint.child_link] = true;
  }
  for (size_t i = 0; i < model.links_.size(); ++i) {
    if (!is_child[i]) {
      if (model.root_link_ != -1) throw std::runtime_error("URDF: more than one root link");
      model.root_link_ = static_cast<int>(i);
    }
  }
  if (model.root_link_ == -1) throw std::runtime_error("URDF: no root link (cycle?)");

  // Breadth-first topological order; assign q indices in document order of revolute joints.
  std::vector<std::vector<int>> children(model.links_.size());
  for (size_t j = 0; j < unordered.size(); ++j) children[unordered[j].parent_link].push_back(static_cast<int>(j));
  std::vector<int> q_of_unordered(unordered.size(), -1);
  for (size_t j = 0; j < unordered.size(); ++j) {
    if (unordered[j].revolute) {
      q_of_unordered[j] = model.dof_++;
      model.q_names_.push_back(unordered[j].name);
    }
  }
  model.q_to_joint_.assign(model.dof_, -1);
  std::deque<int> queue{model.root_link_};
  while (!queue.empty()) {
    const int link = queue.front();
    queue.pop_front();
    for (int j : children[link]) {
      UrdfJoint joint = unordered[j];
      joint.q_index = q_of_unordered[j];
      const int ordered_index = static_cast<int>(model.joints_.size());
      model.joints_.push_back(joint);
      if (joint.q_index >= 0) model.q_to_joint_[joint.q_index] = ordered_index;
      UrdfLink& child = model.links_[joint.child_link];
      child.parent_joint = ordered_index;
      child.ancestor_q = model.links_[link].ancestor_q;
      if (joint.q_index >= 0) child.ancestor_q.push_back(joint.q_index);
      queue.push_back(joint.child_link);
    }
  }
  if (model.joints_.size() != unordered.size()) throw std::runtime_error("URDF: joints not reachable from root");
  return model;
}

int UrdfModel::joint_index(const std::string& name) const {
  for (int i = 0; i < dof_; ++i) {
    if (q_names_[i] == name) return i;
  }
  throw std::runtime_error("URDF: unknown or fixed joint '" + name + "'");
}

int UrdfModel::link_index(const std::string& name) const {
  for (size_t i = 0; i < links_.size(); ++i) {
    if (links_[i].name == name) return static_cast<int>(i);
  }
  throw std::runtime_error("URDF: unknown link '" + name + "'");
}

Eigen::VectorXd UrdfModel::lower_limits() const {
  Eigen::VectorXd v(dof_);
  for (int i = 0; i < dof_; ++i) v[i] = joint_of_q(i).lower;
  return v;
}

Eigen::VectorXd UrdfModel::upper_limits() const {
  Eigen::VectorXd v(dof_);
  for (int i = 0; i < dof_; ++i) v[i] = joint_of_q(i).upper;
  return v;
}

void UrdfModel::forward(const Eigen::VectorXd& q, FkState& state) const {
  if (q.size() != dof_) throw std::invalid_argument("forward: q has wrong size");
  state.link.resize(links_.size());
  state.axis_w.resize(dof_);
  state.origin_w.resize(dof_);
  state.link[root_link_] = Eigen::Isometry3d::Identity();
  for (const auto& joint : joints_) {
    const Eigen::Isometry3d frame = state.link[joint.parent_link] * joint.origin;
    if (joint.revolute) {
      state.axis_w[joint.q_index] = frame.linear() * joint.axis;
      state.origin_w[joint.q_index] = frame.translation();
      state.link[joint.child_link] = frame * Eigen::AngleAxisd(q[joint.q_index], joint.axis);
    } else {
      state.link[joint.child_link] = frame;
    }
  }
}

void UrdfModel::point_jacobian(const FkState& state, int link, Eigen::Ref<Eigen::Matrix3Xd> jacobian) const {
  point_jacobian(state, link, Eigen::Vector3d::Zero(), jacobian);
}

void UrdfModel::point_jacobian(const FkState& state, int link, const Eigen::Vector3d& offset,
                               Eigen::Ref<Eigen::Matrix3Xd> jacobian) const {
  if (jacobian.rows() != 3 || jacobian.cols() != dof_) throw std::invalid_argument("point_jacobian: wrong shape");
  jacobian.setZero();
  const Eigen::Vector3d p = state.link.at(link) * offset;
  for (int k : links_.at(link).ancestor_q) {
    jacobian.col(k) = state.axis_w[k].cross(p - state.origin_w[k]);
  }
}

}  // namespace dexrt
