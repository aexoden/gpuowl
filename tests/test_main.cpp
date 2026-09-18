// Copyright (C) Jason Lynch

// Runs every TEST, or those whose name contains argv[1]. Exit status 0 when every check passed.

#include "test.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

namespace testing {

std::vector<Case>& registry() {
  static std::vector<Case> cases;
  return cases;
}

namespace {
int failures = 0;
const char* current = "";
}  // namespace

void fail(const char* file, int line, const std::string& what) {
  ++failures;
  fprintf(stderr, "%s:%d: in %s: %s\n", file, line, current, what.c_str());
}

}  // namespace testing

int main(int argc, char** argv) {
  const char* const filter = argc > 1 ? argv[1] : "";
  int run = 0;
  int failedCases = 0;

  for (const testing::Case& c : testing::registry()) {
    if (!strstr(c.name, filter)) { continue; }
    testing::current = c.name;

    int const before = testing::failures;

    try {
      c.fn();
    } catch (const char* mes) {
      testing::fail(__FILE__, __LINE__, std::string("threw \"") + mes + "\"");
    } catch (const std::string& mes) {
      testing::fail(__FILE__, __LINE__, "threw \"" + mes + "\"");
    } catch (const std::exception& e) { testing::fail(__FILE__, __LINE__, std::string("threw ") + e.what()); }

    bool const ok = testing::failures == before;
    failedCases += !ok;
    ++run;
    printf("%-44s %s\n", c.name, ok ? "ok" : "FAILED");
  }

  printf("%d test(s), %d failed\n", run, failedCases);
  return (failedCases || !run) ? 1 : 0;
}
