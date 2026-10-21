// Copyright (C) Jason Lynch

// Runs every TEST, or those whose name contains the filter. Exit status 0 when every check passed.
//
//   prpll-test [-j N] [filter]
//
// Where the platform can fork, each test runs in a process of its own, N at a time (by default one per hardware
// thread), and only a failing test's output is shown. With -j 1, or where there is no fork, they run one after another
// in this process with all their output, which is the way to run one under a debugger.

#include "test.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <string_view>
#include <thread>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>

#include <map>
#include <memory>
#include <optional>
#endif

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

namespace {

bool run(const testing::Case& c) {
  testing::current = c.name;
  int const before = testing::failures;

  try {
    c.fn();
  } catch (const char* mes) {
    testing::fail(__FILE__, __LINE__, std::string("threw \"") + mes + "\"");
  } catch (const std::string& mes) {
    testing::fail(__FILE__, __LINE__, "threw \"" + mes + "\"");
  } catch (const std::exception& e) { testing::fail(__FILE__, __LINE__, std::string("threw ") + e.what()); }

  return testing::failures == before;
}

void report(const testing::Case& c, bool ok) { printf("%-44s %s\n", c.name, ok ? "ok" : "FAILED"); }

int runInProcess(const std::vector<const testing::Case*>& cases) {
  int failed = 0;
  for (const testing::Case* c : cases) {
    bool const ok = run(*c);
    failed += !ok;
    report(*c, ok);
  }
  return failed;
}

#ifndef _WIN32

struct FileCloser {
  void operator()(FILE* f) const { fclose(f); }
};
using TempFile = std::unique_ptr<FILE, FileCloser>;

struct Outcome {
  bool ok = false;
  std::string output;
};

std::string readAll(FILE* f) {
  std::string text;
  rewind(f);
  char buf[4096];
  for (size_t n = 0; (n = fread(buf, 1, sizeof buf, f)) > 0;) { text.append(buf, n); }
  return text;
}

std::string describe(int status) {
  if (WIFSIGNALED(status)) {
    return "killed by signal " + std::to_string(WTERMSIG(status)) + " (" + strsignal(WTERMSIG(status)) + ")\n";
  }
  if (WIFEXITED(status) && WEXITSTATUS(status) > 1) {
    return "exited with status " + std::to_string(WEXITSTATUS(status)) + "\n";
  }
  return {};
}

// Results are reported in the order the tests were registered, whichever finishes first, so that two runs compare
// line for line; a failure brings what its test wrote, and how its process ended where that was not by returning.
int runForked(const std::vector<const testing::Case*>& cases, unsigned jobs) {
  std::vector<std::optional<Outcome>> outcomes(cases.size());
  std::map<pid_t, std::pair<size_t, TempFile>> running;
  size_t started = 0;
  size_t reported = 0;
  int failed = 0;

  while (reported < cases.size()) {
    while (started < cases.size() && running.size() < jobs) {
      TempFile out{tmpfile()};
      if (!out) {
        perror("tmpfile");
        exit(2);
      }
      // Whatever is buffered here would otherwise be written again by the child.
      fflush(nullptr);
      pid_t const pid = fork();
      if (pid < 0) {
        perror("fork");
        exit(2);
      }
      if (pid == 0) {
        dup2(fileno(out.get()), STDOUT_FILENO);
        dup2(fileno(out.get()), STDERR_FILENO);
        // So that what it printed and what it failed on stay in the order they happened.
        setvbuf(stdout, nullptr, _IOLBF, 0);
        bool const ok = run(*cases[started]);
        fflush(nullptr);
        _exit(ok ? 0 : 1);
      }
      running.emplace(pid, std::pair{started, std::move(out)});
      ++started;
    }

    int status = 0;
    pid_t const pid = waitpid(-1, &status, 0);
    if (pid < 0) {
      perror("waitpid");
      exit(2);
    }
    auto node = running.extract(pid);
    if (node.empty()) { continue; }
    auto& [index, out] = node.mapped();
    bool const ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    outcomes[index] = Outcome{.ok = ok, .output = ok ? std::string{} : readAll(out.get()) + describe(status)};

    for (; reported < cases.size() && outcomes[reported]; ++reported) {
      const Outcome& o = *outcomes[reported];
      fputs(o.output.c_str(), stdout);
      report(*cases[reported], o.ok);
      failed += !o.ok;
    }
  }
  return failed;
}

#endif

}  // namespace

int main(int argc, char** argv) {
  [[maybe_unused]] unsigned jobs = std::max(1u, std::thread::hardware_concurrency());
  const char* filter = "";

  for (int i = 1; i < argc; ++i) {
    std::string_view const arg = argv[i];
    if (arg == "-j" && i + 1 < argc) {
      jobs = unsigned(std::max(1, atoi(argv[++i])));
    } else if (arg.starts_with("-j") && arg.size() > 2) {
      jobs = unsigned(std::max(1, atoi(argv[i] + 2)));
    } else {
      filter = argv[i];
    }
  }

  std::vector<const testing::Case*> cases;
  for (const testing::Case& c : testing::registry()) {
    if (strstr(c.name, filter)) { cases.push_back(&c); }
  }

#ifndef _WIN32
  int const failed = jobs > 1 ? runForked(cases, jobs) : runInProcess(cases);
#else
  int const failed = runInProcess(cases);
#endif

  printf("%zu test(s), %d failed\n", cases.size(), failed);
  return (failed || cases.empty()) ? 1 : 0;
}
