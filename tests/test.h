// Copyright (C) Jason Lynch

// A minimal GPU-free unit-test harness.
//
//   TEST(name) { CHECK(condition); CHECK_EQ(actual, expected); }

#pragma once

#include <sstream>
#include <string>
#include <vector>

namespace testing {

struct Case {
  const char* name;
  void (*fn)();
};

std::vector<Case>& registry();

struct Registrar {
  Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

void fail(const char* file, int line, const std::string& what);

}  // namespace testing

#define TEST(name)                                                                                                     \
  static void test_##name();                                                                                           \
  static const testing::Registrar registrar_##name{#name, test_##name};                                                \
  static void test_##name()

#define CHECK(cond)                                                                                                    \
  do {                                                                                                                 \
    if (!(cond)) { testing::fail(__FILE__, __LINE__, #cond); }                                                         \
  } while (0)

#define CHECK_EQ(actual, expected)                                                                                     \
  do {                                                                                                                 \
    auto const& actual_ = (actual);                                                                                    \
    auto const& expected_ = (expected);                                                                                \
    if (!(actual_ == expected_)) {                                                                                     \
      std::ostringstream os_;                                                                                          \
      os_ << #actual " == " #expected ": got " << actual_ << ", expected " << expected_;                               \
      testing::fail(__FILE__, __LINE__, os_.str());                                                                    \
    }                                                                                                                  \
  } while (0)
