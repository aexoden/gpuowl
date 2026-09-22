// Copyright (C) Jason Lynch

#include "Restart.h"

#include "fs.h"
#include "log.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <system_error>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace restart {

namespace {

constexpr const char* CARRY = "PRPLL_RESTART_CARRY";

constexpr u32 DEFAULT_MAX_RESTARTS = 16;

std::vector<std::string> savedArgv;

fs::path startDir;

u32 readGeneration() {
  const char* const text = getenv(CARRY);
  return text ? parseCarry(text) : 0;
}

u32 env(const char* name, u32 fallback) {
  const char* const text = getenv(name);
  if (!text) { return fallback; }
  char* end = nullptr;
  unsigned long const value = strtoul(text, &end, 10);
  return end == text ? fallback : u32(value);
}

}  // namespace

u32 parseCarry(const char* text) {
  if (!text) { return 0; }
  char* end = nullptr;
  unsigned long const value = strtoul(text, &end, 10);
  return end == text ? 0 : u32(value);
}

void init(int argc, char** argv) {
  savedArgv.assign(argv, argv + argc);

  std::error_code ec;
  startDir = fs::current_path(ec);
  if (ec) { startDir.clear(); }

  // Read once, before this generation sets the value it will pass on.
  (void)generation();
}

u32 generation() {
  static u32 const value = readGeneration();
  return value;
}

u32 maxRestarts() { return env("PRPLL_MAX_RESTARTS", DEFAULT_MAX_RESTARTS); }

string reexec() {
#ifdef _WIN32
  return "this platform has no exec";
#else
  u32 const next = generation() + 1;
  if (next > maxRestarts()) {
    return "already restarted " + to_string(generation()) +
      " time(s), which is the limit (raise PRPLL_MAX_RESTARTS to allow more)";
  }
  if (savedArgv.empty()) { return "the command line was not recorded"; }
  if (setenv(CARRY, to_string(next).c_str(), 1)) { return "could not pass the generation count on"; }

  fflush(nullptr);

  if (!startDir.empty()) {
    std::error_code ec;
    fs::current_path(startDir, ec);
    if (ec) { log("restart: could not return to '%s' (%s)\n", startDir.string().c_str(), ec.message().c_str()); }
  }

  std::vector<char*> av;
  av.reserve(savedArgv.size() + 1);
  for (string& arg : savedArgv) { av.push_back(arg.data()); }
  av.push_back(nullptr);

  execv("/proc/self/exe", av.data());
  execvp(av[0], av.data());
  return "exec failed: "s + strerror(errno);
#endif
}

}  // namespace restart
