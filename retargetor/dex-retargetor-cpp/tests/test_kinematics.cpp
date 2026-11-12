// Forward kinematics against a Pinocchio reference (tests/data/fk_reference_left.csv,
// produced by python/gen_fk_reference.py) and analytic vs finite-difference Jacobians.
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "aac_config.hpp"
#include "check.hpp"
#include "dexrt/kinematics.hpp"

namespace {

std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream in(line);
  std::string item;
  while (std::getline(in, item, ',')) out.push_back(item);
  return out;
}

}  // namespace

int main() {
  const dexrt::UrdfModel model = dexrt::UrdfModel::load(dexrt_test::urdf_path());
  CHECK(model.dof() == 21);
  CHECK(model.joint_index("thumb_joint1") == 0);
  CHECK_THROWS(model.joint_index("fixed_joint1"));
  CHECK_THROWS(model.link_index("no_such_link"));

  // Reference comparison.
  std::ifstream file(std::string(DEXRT_SOURCE_DIR) + "/tests/data/fk_reference_left.csv");
  CHECK(file.good());
  std::string header;
  std::getline(file, header);
  const auto columns = split(header);
  struct Column { int q = -1; int link = -1; int axis = -1; };
  std::vector<Column> layout(columns.size());
  for (size_t i = 0; i < columns.size(); ++i) {
    const std::string& c = columns[i];
    if (c.rfind("q:", 0) == 0) {
      layout[i].q = model.joint_index(c.substr(2));
    } else if (c.rfind("p:", 0) == 0) {
      const size_t colon = c.rfind(':');
      layout[i].link = model.link_index(c.substr(2, colon - 2));
      layout[i].axis = c.substr(colon + 1) == "x" ? 0 : c.substr(colon + 1) == "y" ? 1 : 2;
    }
  }
  int rows = 0;
  double worst = 0;
  dexrt::FkState state;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    const auto values = split(line);
    CHECK(values.size() == columns.size());
    Eigen::VectorXd q = Eigen::VectorXd::Zero(model.dof());
    for (size_t i = 0; i < values.size(); ++i) {
      if (layout[i].q >= 0) q[layout[i].q] = std::stod(values[i]);
    }
    model.forward(q, state);
    for (size_t i = 0; i < values.size(); ++i) {
      if (layout[i].link < 0) continue;
      const double expected = std::stod(values[i]);
      const double actual = state.link[layout[i].link].translation()[layout[i].axis];
      worst = std::max(worst, std::fabs(expected - actual));
    }
    ++rows;
  }
  CHECK(rows >= 10);
  CHECK_NEAR(worst, 0.0, 1e-9);
  std::printf("FK reference rows %d, worst |dp| = %.3e m\n", rows, worst);

  // Jacobian check by central differences at a random interior configuration.
  const Eigen::VectorXd lo = model.lower_limits(), hi = model.upper_limits();
  Eigen::VectorXd q = lo + 0.37 * (hi - lo);
  for (int k = 0; k < model.dof(); ++k) q[k] = lo[k] + (0.2 + 0.6 * ((k * 7919) % 97) / 97.0) * (hi[k] - lo[k]);
  model.forward(q, state);
  const char* links[] = {"thumb_tip", "index_tip", "middle_pip", "ring_mcp", "pinky_tip", "palm_down"};
  const double h = 1e-6;
  double worst_j = 0;
  Eigen::Matrix3Xd J(3, model.dof());
  dexrt::FkState plus, minus;
  for (const char* name : links) {
    const int link = model.link_index(name);
    model.point_jacobian(state, link, J);
    for (int k = 0; k < model.dof(); ++k) {
      Eigen::VectorXd qp = q, qm = q;
      qp[k] += h;
      qm[k] -= h;
      model.forward(qp, plus);
      model.forward(qm, minus);
      const Eigen::Vector3d numeric = (plus.link[link].translation() - minus.link[link].translation()) / (2 * h);
      worst_j = std::max(worst_j, (numeric - J.col(k)).norm());
    }
  }
  CHECK_NEAR(worst_j, 0.0, 1e-6);
  std::printf("Jacobian worst |dJ| = %.3e\n", worst_j);

  // Fixed root: palm_down must not move with any joint.
  model.point_jacobian(state, model.link_index("palm_down"), J);
  CHECK_NEAR(J.norm(), 0.0, 0.0);
  return dexrt_test::finish("test_kinematics");
}
