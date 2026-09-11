// Copyright (C) Mihai Preda

#include "Args.h"
#include "Background.h"
#include "Queue.h"
#include "Signal.h"
#include "Task.h"
#include "Worktodo.h"
#include "version.h"
#include "AllocTrac.h"
#include "typeName.h"
#include "log.h"
#include "Context.h"
#include "TrigBufCache.h"
#include "GpuCommon.h"
#include "Gpu.h"
#include "tune.h"

#include <filesystem>
#include <thread>
#include <utility>
// #include <format> from GCC-13 onwards

namespace {

constexpr int EXIT_OK = 0;
constexpr int EXIT_FAILED = 1;
constexpr int EXIT_USAGE = 2;

} // namespace

static void gpuWorker(GpuCommon shared, i32 instance) {
  // LogContext context{(instance ? shared.args->tailDir() : ""s) + to_string(instance) + ' '};
  // log("Starting worker %d\n", instance);
  if (instance > 0) {
    initLog(("gpuowl-"s + to_string(instance) + ".log").c_str());
    log("PRPLL %s, instance %d\n", VERSION, instance);
  }

  try {
    while (auto task = Worktodo::getTask(*shared.args, instance)) { task->execute(shared, instance); }
  } catch (const char *mes) {
    log("Exception \"%s\"\n", mes);
  } catch (const string& mes) {
    log("Exception \"%s\"\n", mes.c_str());
  } catch (const std::exception& e) {
    log("Exception %s: %s\n", typeName(e), e.what());
  }
}


#if defined(__MINGW32__) || defined(__MINGW64__) || defined(__MSYS__) // for Windows
extern int putenv(char *);
#endif

int main(int argc, char **argv) {
//!MSVC version support
#ifdef _MSC_VER
  _set_printf_count_output(1);    // I'm not sure what this does (it's from CrazeTheDragon)
#endif

#ifdef __MSYS__
  // I was unable to get putenv to link in MSYS2
#elif defined(__MINGW32__) || defined(__MINGW64__)
  putenv("ROC_SIGNAL_POOL_SIZE=32");
#elif defined(_WIN32)
  _putenv_s("ROC_SIGNAL_POOL_SIZE", "32");  // For MSVC
#else
  // Required to work around a ROCm bug when using multiple queues
  setenv("ROC_SIGNAL_POOL_SIZE", "32", 0);
#endif

  int exitCode = EXIT_OK;
  Args args;

  try {
    string const mainLine = Args::mergeArgs(argc, argv);
    {
      Args first{true};
      first.parse(mainLine);

      // "-h", "-version" and "-info" are handled by the first Args instance.
      if (first.printedAndDone) { return EXIT_OK; }
      if (!first.dir.empty()) {
        fs::current_path(first.dir);
      }
    }

    fs::path poolDir;
    {
      Args second{true};
      second.readConfig("config.txt");
      second.parse(mainLine);
      poolDir = second.masterDir;
    }

    initLog("gpuowl-0.log");
    log("PRPLL %s starting\n", VERSION);

    if (!poolDir.empty()) { args.readConfig(poolDir / "config.txt"); }
    args.readConfig("config.txt");
    args.parse(mainLine);
  } catch (const char *mes) {
    log("Exiting because \"%s\"\n", mes);
    return EXIT_USAGE;
  } catch (const string& mes) {
    log("Exiting because \"%s\"\n", mes.c_str());
    return EXIT_USAGE;
  } catch (const std::exception& e) {
    log("Exiting because an argument value could not be read (%s)\n", e.what());
    return EXIT_USAGE;
  }

  try {
    // Opens the device, so it's done in the run try block.
    args.setDefaults();

    if (args.maxAlloc) { AllocTrac::setMaxAlloc(args.maxAlloc); }

    Context context(getDevice(args.device));
    Signal const signal;
    Background background;
    GpuCommon shared;
    shared.context = &context;
    shared.args = &args;
    TrigBufCache bufCache{&context};
    shared.bufCache = &bufCache;
    shared.background = &background;

    if (args.doCtune || args.doTune || args.doZtune || args.carryTune) {
      Tune tune{shared};

      if (args.doCtune) {
        tune.ctune();
      } else if (args.doTune) {
        tune.tune();
      } else if (args.doZtune) {
        tune.ztune();
      } else if (args.carryTune) {
        tune.carryTune();
      }
    } else {
      {
        vector<jthread> threads;
        for (int i = 1; std::cmp_less(i, args.workers); ++i) {
          threads.emplace_back(gpuWorker, shared, i);
        }
        gpuWorker(shared, 0);
      }

      // log("No more work. Add work to worktodo.txt , see -h for details.\n");
    }
  } catch (const char *mes) {
    log("Exiting because \"%s\"\n", mes);
    exitCode = EXIT_FAILED;
  } catch (const string& mes) {
    log("Exiting because \"%s\"\n", mes.c_str());
    exitCode = EXIT_FAILED;
  } catch (const std::exception& e) {
    log("Exiting because %s: %s\n", typeName(e), e.what());
    exitCode = EXIT_FAILED;
  }

  log("Bye\n");
  return exitCode;
}
