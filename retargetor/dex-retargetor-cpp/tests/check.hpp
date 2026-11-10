// Tiny assertion helper so the tests have no framework dependency.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace dexrt_test {

inline int& failures() {
  static int count = 0;
  return count;
}

inline void check(bool ok, const char* expression, const char* file, int line, const std::string& detail = "") {
  if (ok) return;
  ++failures();
  std::fprintf(stderr, "FAILED %s:%d: %s %s\n", file, line, expression, detail.c_str());
}

inline int finish(const char* name) {
  if (failures() == 0) {
    std::printf("%s: all checks passed\n", name);
    return 0;
  }
  std::fprintf(stderr, "%s: %d check(s) failed\n", name, failures());
  return 1;
}

}  // namespace dexrt_test

#define CHECK(expr) dexrt_test::check((expr), #expr, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol) \
  dexrt_test::check(std::fabs(double(a) - double(b)) <= (tol), #a " ~ " #b, __FILE__, __LINE__, \
                    "(" + std::to_string(double(a)) + " vs " + std::to_string(double(b)) + ")")
#define CHECK_THROWS(stmt)                                      \
  do {                                                          \
    bool thrown = false;                                        \
    try {                                                       \
      stmt;                                                     \
    } catch (...) {                                             \
      thrown = true;                                            \
    }                                                           \
    dexrt_test::check(thrown, "throws: " #stmt, __FILE__, __LINE__); \
  } while (0)
