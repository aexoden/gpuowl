// Copyright (C) Mihai Preda

#include "Args.h"
#include "FFTVariants.h"
#include "Measure.h"
#include "Tuner.h"
#include "File.h"
#include "clwrap.h"
#include "gpuid.h"
#include "Proof.h"
#include "version.h"

#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <cctype>
#include <iterator>
#include <sstream>
#include <algorithm>
#include <charconv>

// This is a copy of the args.verbose level.  It allows the CUDA wrapper to access the value.
int prpll_verbose = 0;

int Args::value(const string& key, int valNotFound) const {
  auto it = flags.find(key);
  if (it == flags.end()) { return valNotFound; }
  return atoi(it->second.c_str());
}

string Args::mergeArgs(int argc, char **argv) {
  string ret;
  for (int i = 1; i < argc; ++i) {
    ret += argv[i];
    ret += " ";
  }
  return ret;
}

vector<KeyVal> Args::splitArgLine(const string& inputLine) {
  vector<KeyVal> ret;

  string prev;
  for (const string& s : split(inputLine, ' ')) {
    if (s.empty()) { continue; }

    if (prev.empty()) {
      if (s[0] != '-') {
        log("Args: expected '-' before '%s'\n", s.c_str());
        throw "Argument syntax";
      }

      prev = s;
    } else {
      // A token such as "-5" is a negative value for the preceding option (e.g. -od -5), not a new option.
      bool const isNegativeNumber = s[0] == '-' && s.size() > 1 && (isdigit((unsigned char) s[1]) || s[1] == '.');
      if (s[0] == '-' && !isNegativeNumber) {
        ret.push_back({prev, {}});
        prev = s;
      } else {
        ret.emplace_back(prev, s);
        prev.clear();
      }
    }
  }
  if (!prev.empty()) {
    assert(prev[0] == '-');
    ret.push_back({prev, {}});
  }
  return ret;
}

// Aliases for -use keys, accepted so a variant spelling doesn't silently do nothing. George has
// repeatedly told users on the forum to "Try -use NOASM" (no underscore); keep that working by
// mapping it to the real key NO_ASM before it reaches -use validation or the OpenCL -D defines.
static const std::map<string, string> useKeyAliases = {
  {"NOASM", "NO_ASM"},
};

// Splits a string of the form "Foo=bar,C,D=1" into key=value pairs, with value defaulting to "1".
vector<KeyVal> Args::splitUses(string ss) { // pass by value is intentional
  vector<KeyVal> ret;
  std::ranges::replace(ss, ',', ' ');
  std::istringstream iss{ss};
  vector<string> const uses{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};
  for (const string &s : uses) {
    auto pos = s.find('=');
    string key = (pos == string::npos) ? s : s.substr(0, pos);
    string const val = (pos == string::npos) ? "1"s : s.substr(pos+1);
    if (auto it = useKeyAliases.find(key); it != useKeyAliases.end()) {
      log("-use %s taken as %s\n", key.c_str(), it->second.c_str());
      key = it->second;
    }
    ret.emplace_back(key, val);
  }
  return ret;
}

// Checks the comma separated -tune options up front: Tune::tune() silently skips anything it does not recognise,
// so a typo such as "maxexponent=" would otherwise tune the default exponent range for hours without a word.
static void checkTuneOptions(const string& options) {
  for (const string& s : split(options, ',')) {
    if (s.empty() || s == "noconfig" || s == "fp64" || s == "ntt" || s == "fp6431" || s == "nofp32" || s == "inplace") { continue; }
    auto pos = s.find('=');
    string const key = s.substr(0, pos);
    if (pos != string::npos && (key == "quick" || key == "minexp" || key == "maxexp")) {
      string const val = s.substr(pos + 1);
      u64 n = 0;
      auto [end, ec] = std::from_chars(val.data(), val.data() + val.size(), n);
      if (val.empty() || ec != std::errc{} || end != val.data() + val.size() || (key == "quick" && (n < 1 || n > 10))) {
        log("-tune %s expects %s (found '%s')\n", key.c_str(), key == "quick" ? "a value from 1 to 10" : "a whole number, e.g. 5000000000", val.c_str());
        throw "-tune option value";
      }
      continue;
    }
    log("-tune option '%s' not understood; valid options are noconfig, inplace, fp64, ntt, nofp32, fp6431, minexp=<val>, maxexp=<val>, quick=<val>\n", s.c_str());
    throw "-tune option";
  }
}

void Args::readConfig(const fs::path& path) {
  if (File file = File::openRead(path)) {
    file.allowUnterminatedLastLine();
    for (string line : file) {
      line = rstripNewline(line);
      parse(line, true);
    }
  }
}

u32 Args::getProofPow(u64 exponent) const {
  if (proofPow == -1) { return ProofSet::bestPower(exponent); }
  assert(proofPow >= 0);  // 0 == proof generation disabled
  return proofPow;
}

string Args::tailDir() const { return fs::path{dir}.filename().string(); }

bool Args::hasFlag(const string& key) const { return flags.contains(key); }

void Args::printHelp() {
  printf(R"(
PRPLL is "PRobable Prime and Lucas-Lehmer Categorizer", AKA "Purple-cat"

PRPLL is an OpenCL/CUDA (GPU) program for primality testing Mersenne numbers (of the form 2^n - 1).

To check that OpenCL is installed correctly use the command "clinfo". If clinfo does not find any
devices or otherwise fails, this program will not run.

This program is tested on Linux/ROCm (AMD GPUs); it also runs on Windows and on Nvidia GPUs.

For information about Mersenne primes search see https://www.mersenne.org/

Run "prpll -h"; If this displays a list of OpenCL devices, it means that PRPLL is detecting the GPUs
and should be able to run.


Worktodo:
PRPLL keeps the active tasks in per-worker files worktodo-0.txt, worktodo-1.txt etc in the local directory.
These per-worker files are supplied from the global worktodo.txt file if -pool is used.
In turn the work files can be supplied through AutoPrimeNet, located at https://download.mersenne.ca/AutoPrimeNet

It is also possible to manually add exponents by adding lines of the form "PRP=118063003" to worktodo-<N>.txt


The configuration options listed below can be passed on the command line or can be put in a file
named "config.txt" in the prpll run directory.

The "config.txt" file also supports per-FFT configuration lines of the form "! <fft-selector> <key=value,...>". The
selector is an FFT spec cut short after any part, optionally prefixed by prp: or ll:
   ! 1                  every FFT of type 1        ! 512:15:512:101     one shape and variant
   ! 512:15:512         one FP64 shape             ! ll:512:15:512      that shape, LL tests only
A kind-qualified line beats an unqualified one, then the more specific selector wins, then last line read. A '!' line
takes precedence over a plain -use line in config.txt, but -use on the command line still applies. Variant and carry
settings that compile the same kernels name a single FFT (256:2:256:111 is 256:2:256:101). A selector ending in a carry
(:0 or :1) matches only an FFT whose spec pins that carry, such as an FP64 entry in tune.txt whose 32-bit carry limits
its reach.


-h                 : print general help, list of FFTs, list of devices
-info <fft>        : print detailed information about the given FFT; e.g. -info 1K:13:256
-options [<fft>]   : print the -use options this build knows, resolved for this GPU and <fft> (default 512:15:512),
                     with the tuner's groups and combo clusters, then check the option table (exit 1 if it fails)
-measure <fft>[,<k>=<v>...] : time <fft> repeatedly in production-sized blocks and report whether the error bar it
                     declares describes how far its readings move, plus its per-call construction cost, its residue
                     and its rounding error, at the exponent given by -prp or, without one, the top of its range.
                     Settings: n=<calls> (8), blocks=<per call>, block=<iterations>, exp=<E>, anchor=<fft> (time a
                     second configuration alternately and correct for its drift, which turns off the scheduled one),
                     roe=0|1, drain=0|1, drift=0|1 (time the session's drift anchor, on by default)
-dir <folder>      : specify local work directory (containing worktodo-<N>.txt, results-<N>.txt, config.txt,
                     gpuowl-<N>.log)
-pool <dir>        : specify a directory with the shared (pooled) worktodo.txt and config.txt
                     Multiple PRPLL instances, each in its own directory, can share a pool of assignments.
                     Results are still written locally, to results-<N>.txt in each instance's own directory.
-verbose           : print more log, useful for developers
-version           : print only the version and exit
-user <name>       : specify the mersenne.org user name (for result reporting)
-workers <N>       : specify the number of parallel PRP tests to run (default 1)

-fft <spec>        : specify FFT or FFTs to use:
                     - a specific configuration: 256:13:1K
                     - a FFT size: 6.5M
                     - a size range: 7M-8M
                     - a list: 256:13:1K,8M
                     See the list of FFTs at the end.

-od <value>        : Overdrive the FFT range (ROE, CARRY32 limits). This allows to use a lower FFT for a given
                     exponent (thus faster), but increases the risk of errors. The presence of errors is detected,
                     but the errors are nevertheless costly computationally and better avoided.
                     A <value> of 1 extends the range by 0.1%% (and this would be acceptable); a value of 10
                     extends the range by 1%% (and this would be quite too much WRT errors).

-block <value>     : PRP block size, one of: 1000, 500, 200. Default 1000.
-carry long|short  : force carry type. Short carry may be faster, but requires high bits/word.
-prp <exponent>    : run a single PRP test and exit, ignoring worktodo.txt
-ll <exponent>     : run a single LL test and exit, ignoring worktodo.txt
-verify <file>     : verify PRP-proof contained in <file>
-smallest          : work on smallest exponent in worktodo.txt rather than the first exponent in worktodo.txt    
-proof <power>     : generate proof of power <power> (default: optimal depending on exponent).
                     A lower power reduces disk space requirements but increases the verification cost.
                     A higher power increases disk usage a lot.
                     e.g. proof power 10 for a 120M exponent uses about %.0fGB of disk space.
                     -proof 0 disables proof generation: the PRP result is reported without a proof.
-autoverify <power>: self-verify a generated proof only when its power is at least <power>. Default 0,
                     i.e. verify every proof. Self-verification costs about E/2^power iterations, so it
                     is cheap at high proof powers and expensive at low ones; it must also complete in
                     the same run that generated the proof. Use e.g. -autoverify 9 to skip it for the
                     low powers, or a value above the maximum power to skip it entirely.
-iters <N>         : run next PRP test for <N> iterations and exit.
-save <N>          : specify the number of savefiles to keep (default %u).
-noclean           : do not delete data after the test is complete.
-cache             : use binary kernel cache; useful with repeated use of -roeTune and -tune
-roe               : measure the Round-Off Error (Z) for more iterations (slow)
-time              : collect and print a per-kernel GPU timing profile
-log <N>           : log progress and checkpoint every <N> iterations (positive multiple of 1000; default 20000)

-use <define>      : comma separated list of defines for configuring openCL code, such as:
  -use FAST_BARRIER: on AMD Radeon VII and older AMD GPUs, use a faster barrier().  This option
                     may not work on Nvidia GPUs.  It is ignored on RDNA and on MI200 and later
                     AMD GPUs, where the faster barrier gives wrong results.
  -use NO_ASM      : do not use __asm() blocks (inline assembly); also accepted as NOASM
  -use TAIL_KERNELS=<val> : change how tailSquare and tailMul operate according to <val>:
                     0 = single wide, single kernel
                     1 = single wide, two kernels
                     2 = double wide, single kernel
                     3 = double wide, two kernels
  -use TAIL_TRIGS=<val> : change how tailSquare computes final trig values according to <val>:
                     2 = calculate from scratch, no memory read
                     1 = calculate using one complex multiply from cached memory and uncached memory
                     0 = read trig values from memory
  -use INPLACE=n   : Perform tranforms in-place.  Great if the reduced memory usage fits in the GPU's L2 cache.
                     0 = not in-place, 1 = nVidia friendly access pattern, 2 = AMD friendly access pattern.
  -use PAD=<val>   : insert pad bytes to possibly improve memory access patterns.  Val is number bytes to pad.
  -use MIDDLE_IN_LDS_TRANSPOSE=0|1  : Transpose values in local memory before writing to global memory
  -use MIDDLE_OUT_LDS_TRANSPOSE=0|1 : Transpose values in local memory before writing to global memory
  -use TABMUL_CHAIN=<val>: Controls how trig values are obtained in WIDTH and HEIGHT when FFT-spec is 1.
                     0 = Read one trig value and compute the next 3 or 7.
                     1 = All trig values are pre-computed and read from memmory.

  -use DEBUG       : enable asserts in OpenCL kernels (slow, developers)
  -use STATS=<val> : enable carry statistics collection & logging (developers), for the kernel according to <val>:
                     1 = CarryFused, 2 = CarryFusedMul, 4 = CarryA, 8 = CarryMul

-tune <options>    : Looks for best settings to include in config.txt.  Times many FFTs to find fastest one to test exponents -- written to tune.txt.
                     An -fft <spec> can be given on the command line to limit which FFTs are timed.
                     Options are not required.  If present, the options are a comma separated list from below.
                         noconfig     - Skip timings to find best config.txt settings.
                         inplace      - Skip timings for not-in-place FFTs and NTTs.  All nVidia GPUs seem to prefer in-place FFTs and NTTs.
                         fp64         - Tune for settings that affect FP64 FFTs.  Time FP64 FFTs for tune.txt.
                         ntt          - Tune for settings that affect integer NTTs.  Time integer NTTs for tune.txt.
                         nofp32       - Do not tune for settings that affect FP32 FFTs.  Some openCL compilers have trouble with FP32.
                         minexp=<val> - Time FFTs to find the best one for exponents greater than <val>.  Default 75000000.
                         maxexp=<val> - Time FFTs to find the best one for exponents less than <val>.  Default 350000000.
                                        Without an -fft <spec>, only FFTs in [minexp, maxexp] are timed, so tuning
                                        for a small exponent (e.g. PRP-CF at 18M) needs both ends set low, e.g.
                                        -tune minexp=10000000,maxexp=20000000
                         fp6431       - Time FP64+M31 FFTs for tune.txt.  Only GPUs with great FP64 performance will find this beneficial.
                         quick=<val>  - Use higher values for a quicker, potentially less accurate tune.  Val ranges from 1 to 10.
                     These subcommands open no device, so they run on a machine that has none.  Those working on
                     the measurement database, without an env= (into= for adopt), act on the one env whose rows were
                     measured against the kernels this binary carries.
                         emit[,env=<id>]               - write selection.txt from what the database supports
                         reset[,env=<id>][,fft=<spec>] - drop what was measured, for an env or for one of its FFTs
                         adopt[,into=<id>][,from=<id>] - take an earlier env's rows as the current kernels' own
                         compact                       - fold duplicate rows, and drop option sets nothing names
                         scope[,env=<id>]              - report the exponents a tuning run would work over, and the
                                                         expected iteration time over them that the database supports
                     scope and emit take the settings that bound a run, each defaulting from the pending worktodo
                     (emit, so that it finds the races a run held at the probe and writes the lines they decided):
                         workload=<lo>-<hi>  - the exponents worth covering, e.g. workload=100M-400M
                         probe=<E>           - the exponent that matters most, rounded to the prime at or below it
                         probeWeight=<0..1>  - how much of the weight the probe carries on its own (0.5)
                         kinds=prp|ll|prp+ll - which test kinds to tune for (prp)
                     Given only those settings, or nothing, -tune runs the new tuner on the device instead: it
                     times what is worth timing for the workload into tunedb.txt, publishes selection.txt after every
                     measurement, and stops cleanly on Ctrl-C; a re-run resumes. It tunes prp only (kinds=prp).
                     It first races the -use options of each FFT type worth tuning on one FFT at the probe, and
                     publishes the winners as selection.txt's default lines; bootstrap=0 skips that.
                     Then each FFT it has measured is searched one step at a time from its best option set, where
                     strategy= says what a step is: groups of related options together (hybrid, groups), one
                     option at a time (single), or every combination of named options (permute:PAD+IN_SIZEX).
                     hybrid then combines the best comboTop=<N> (3) answers of each group, over comboTiers=<1..3>
                     (3) tiers: groups alone, groups that share kernels, everything; comboTiers=1 is groups.
                     The option words above (noconfig, fp64, quick=, ...) still select the previous tuner.
                         accuracy[,fft=<spec>][,groups=<Group>+...] - on the device: read the rounding error of
                             every value of every -use option against the set it moved from, on each FFT type's
                             bootstrap FFT (or on fft=) at the probe, recording the readings in tunedb.txt, and say
                             which options change the rounding. Takes workload= and probe= as a run does.
-device <N>        : select the GPU at position N in the list of devices
-uid    <UID>      : select the GPU with the given UID (on ROCm/AMDGPU, Linux)
-pci    <BDF>      : select the GPU with the given PCI BDF, e.g. "0c:00.0"

Device selection : use one of -uid <UID>, -pci <BDF>, -device <N>, see the list below

)", ProofSet::diskUsageGB(120000000, 10), nSavefiles);

  vector<cl_device_id> deviceIds = getAllDeviceIDs();
  if (!deviceIds.empty()) {
    printf(" N  : PCI BDF |   UID            |   Driver                 |    Device\n");
  }
  for (unsigned i = 0; i < deviceIds.size(); ++i) {
    cl_device_id id = deviceIds[i];
    string const bdf = getBdfFromDevice(id);
    printf("%2u  : %7s | %16s | %-24s | %s | %s\n",
           i,
           bdf.c_str(),
           getUidFromBdf(bdf).c_str(),
           getDriverVersion(id).c_str(),
           getDeviceName(id).c_str(),
           getBoardName(id).c_str()
           );

  }
  printf("\nFFT Configurations (specify with -fft <type>:<width>:<middle>:<height> from the set below):\n");

  vector<FFTShape> const configs = FFTShape::allShapes();
  for (auto [type, name] : {pair{FFT64, "FP64"}, {FFT3161, "M31+M61 NTT"}, {FFT3261, "FP32+M61"}, {FFT61, "M61 NTT"},
                            {FFT323161, "FP32+M31+M61"}, {FFT6431, "FP64+M31"}}) {
    printf("\nFFT type %d: %s\n"
           " Size   MaxExp   BPW    FFT\n", type, name);
    u32 activeSize = 0;
    float maxBpw = 0;
    string variants;
    auto flush = [&]() {
      if (variants.empty()) { return; }
      printf("%5s  %7.2fM  %.2f  %s\n",
             numberK(activeSize).c_str(),
             // activeSize * FFTShape::MIN_BPW / 1'000'000,
             activeSize * maxBpw / 1'000'000.0,
             maxBpw,
             variants.c_str());
      variants.clear();
    };
    for (const FFTShape& c : configs) {
      if (c.fft_type != type) continue;
      if (c.size() != activeSize) {
        flush();
        activeSize = c.size();
        maxBpw = 0;
      }
      maxBpw = max(maxBpw, c.maxBpw());
      if (!variants.empty()) { variants.push_back(','); }
      variants += c.spec();
    }
    flush();
  }
}

void Args::parse(const string& line, bool fromConfigFile) {
  if (line.empty() || line[0] == '#') { return; }

  if (line[0] == '!') {
    // conditional defines predicated on a FFT
    perFftConfig.push_back(tune::parseUseLine(line));
    return;
  }

  if (!silent) { log("config: %s\n", line.c_str()); }

  auto args = splitArgLine(line);

  for (const auto& [key, s] : args) {
    // log("key '%s'\n", key.c_str());
    if (key == "-h" || key == "--help") {
      printHelp();
      throw "help";
    } if (key == "-version") {
      // Plain stdout, no log prefix: the flag exists for scripts and launchers
      // that record which build wrote a result (Task.cpp reports VERSION to
      // PrimeNet), so the one line must be the version and nothing else.
      printf("%s\n", (VERSION[0] == 'v') ? VERSION + 1 : VERSION);
      fflush(stdout);
      throw "version";
    } if (key == "-info") {
      if (s.empty()) {
        log("-info expects an FFT spec, e.g. -info 1K:13:256\n");
        throw "-info <fft>";
      }
      log(" FFT              | BPW   | Max exp (M)\n");
      for (const FFTShape& shape : FFTShape::multiSpec(s)) {
        for (u32 const variant : tune::allVariants(shape)) {
          FFTConfig const fft{shape, variant, CARRY_AUTO};
          log("%12s | %.2f | %5.1f\n", fft.spec().c_str(), fft.maxBpw(), fft.maxExp() / 1'000'000.0);
        }
      }
      throw "info";
    } if (key == "-od") {
      double od = stod(s);
      fftOverdrive = 1 + od / 1000;
    } else if (key == "-roe") {
      assert(s.empty());
      logROE = true;
    } else if (key == "-tune") {
      doTune = true;
      if (!s.empty()) { tune = s; }
      // The database-only subcommands are dispatched by main() before a device exists; validated here, so a mistyped
      // setting is a usage error rather than a silent fall-through to the tuner that takes the same flag.
      if (!tune::parseTuneCommand(tune)) { checkTuneOptions(tune); }
    } else if (key == "-measure") {
      // Resolving the options and building a Gpu need the device, so main() does this once a
      // context exists.
      if (s.empty()) { throw "-measure needs an FFT spec"; }
      doMeasure = true;
      measureSpec = s;
      (void)tune::parseMeasureArgs(measureSpec);
    } else if (key == "-options") {
      // Resolving the table needs the GPU, so main() does this once a context exists.
      dumpOptions = true;
      if (!s.empty()) { optionsFft = s; }
      (void) FFTConfig{optionsFft};
//    } else if (key == "-ctune") {
//      doCtune = true;
//      if (!s.empty()) { ctune.push_back(s); }
    } else if (key == "-ztune") {
      doZtune = true;
    } else if (key == "-carryTune") {
      carryTune = true;
    } else if (key == "-verbose" || key == "-v") {
      if (s.empty()) verbose = 1;
      else verbose = stoi(s);
      prpll_verbose = verbose;
    } else if (key == "-time") {
      profile = true;
    } else if (key == "-workers") {
      if (s.empty()) {
        log("-workers expects <N>\n");
        throw "-workers <N>";
      }
      workers = stoi(s);
      if (workers < 1 || workers > 4) {
        throw "Number of workers must be between 1 and 4";
      }
    } else if (key == "-cache") {
      useCache = true;
    } else if (key == "-noclean") {
      clean = false;
    } else if (key == "-proof") {
      int power = 0;
      if (s.empty() || (power = stoi(s)) < 0 || power > 13) {
        log("-proof expects <power> 0-13 (found '%s')\n", s.c_str());
        throw "-proof <power>";
      }
      proofPow = power;
      assert(proofPow >= 0);
    } else if (key == "-autoverify") {
      int power = 0;
      if (s.empty() || (power = stoi(s)) < 0 || power > 14) {
        log("-autoverify expects <power> 0-14 (found '%s')\n", s.c_str());
        throw "-autoverify <power>";
      }
      proofVerify = power;
    } else if (key == "-keep") {
      if (s != "proof") {
        log("-keep requires 'proof'\n");
        throw "-keep without proof";
      }
      keepProof = true;
    } else if (key == "-verify") {
      if (s.empty()) {
        log("-verify needs <proof-file>\n");
        throw "-verify without proof-file";
      }
      verifyPath = s;
    }
    else if (key == "-pool") {
      masterDir = s;
      if (!masterDir.is_absolute()) {
        log("-pool <path> requires an absolute path\n");
        throw("-pool <path> requires an absolute path");
      }
    }
    else if (key == "-maxAlloc" || key == "-maxalloc") {                // DEPRECATED, was only used for P-1 buffers.  Parsing left in place so previous users do not get an error.
      if (s.empty()) {                                                  // s.back() below would be undefined
        log("-maxAlloc expects a value, e.g. -maxAlloc 4G\n");
        throw "-maxAlloc <size>";
      }
      u32 multiple = (s.back() == 'G') ? (1u << 30) : (1u << 20);
      maxAlloc = size_t(stod(s) * multiple + .5);
    }
    // DEPRECATED options from old gpuowl config.txt files that no longer affect anything PRPLL does.
    // Accepted (rather than "not understood") so migrated configs keep running; see forum #474/#480.
    else if (key == "-yield") {          // was a work-around for Nvidia's CUDA busy-wait eating a CPU core; PRPLL has no such busy-wait to work around.
      log("-yield is deprecated and ignored (CUDA busy-wait work-around no longer applies)\n");
    }
    else if (key == "-nospin") {         // used to silence the "-\\|/" progress spinner, which no longer exists.
      log("-nospin is deprecated and ignored (there is no progress spinner to silence)\n");
    }
    else if (key == "-cpu") {            // used to label results with a machine name; PRPLL derives that label from the last segment of -dir instead.
      log("-cpu is deprecated and ignored (results are now labeled from -dir instead)\n");
    }
    else if (key == "-results") {        // used to rename results.txt; PRPLL always writes results-<worker>.txt.
      log("-results is deprecated and ignored (results are always written to results-<N>.txt)\n");
    }
    else if (key == "-tmpDir" || key == "-tmpdir") {   // used to redirect proof checkpoint scratch space.
      log("-tmpDir is deprecated and ignored (proof checkpoints are always kept under -dir)\n");
    }
    else if (key == "-binary") {         // used to load a precompiled kernel binary from a given file.
      log("-binary is deprecated and ignored; use -cache for a persistent kernel cache instead\n");
    }
    // DEPRECATED: P-1 factoring (and its second-stage mprime interop) was removed along with the GMP
    // dependency, not for cost (see PR history). These options would silently change what gets tested,
    // so unlike the no-ops above they must not be swallowed quietly.
    else if (key == "-B1" || key == "-b1" || key == "-B2" || key == "-b2" || key == "-rB2" ||
             key == "-pm1" || key == "-mprimeDir" || key == "-D") {
      log("%s: P-1 factoring is no longer supported; remove it from config.txt\n", key.c_str());
      throw "P-1 no longer supported";
    }
    else if (key == "-from") {           // used to resume at a specific iteration instead of the latest checkpoint.
      log("-from is no longer supported; PRPLL always resumes from the most recent checkpoint in -dir\n");
      throw "-from no longer supported";
    }
    else if (key == "-iters") { iters = stoi(s); assert(iters > 0); }   // any positive count; release never enforced the old multiple-of-10000 rule
    else if (key == "-prp" || key == "-PRP") { prpExp = stoll(s); }
    else if (key == "-ll" || key == "-LL") { llExp = stoll(s); }
    else if (key == "-smallest") { smallest = true; }
    else if (key == "-fft") {
      // Old gpuowl also accepted a "+N"/"-N" relative offset ("nudge the auto-selected FFT by N steps"),
      // which PRPLL never implemented; passed through as a literal spec it hits FFTConfig's opaque
      // "FFT spec" parse failure. "+0"/"-0" always meant "no change" regardless of version, so accept
      // that one case as if -fft were not given; any other offset picks an unspecified FFT, so reject
      // it with a clear message instead of that opaque failure.
      bool const isOffset = s.size() >= 2 && (s[0] == '+' || s[0] == '-') &&
        s.find_first_not_of("0123456789", 1) == string::npos;
      if (isOffset && stoi(s) == 0) {
        log("-fft %s ignored (relative FFT size offsets are not supported; auto-selecting FFT)\n", s.c_str());
      } else if (isOffset) {
        log("-fft %s not supported: relative FFT size offsets (+N/-N) no longer exist; "
            "specify an explicit FFT size or spec (e.g. -fft 6.5M), or omit -fft to auto-select\n", s.c_str());
        throw "-fft offset not supported";
      } else {
        fftSpec = s;
      }
    }
    else if (key == "-user") { user = s; }
    else if (key == "-device" || key == "-d") { device = stoi(s); }
    else if (key == "-uid") { device = getPosFromUid(s); }
    else if (key == "-pci") { device = getPosFromBdf(s); }
    else if (key == "-dir") { dir = s; }
    else if (key == "-carry") {
      if (s == "short" || s == "long") {
        carry = s == "short" ? CARRY_32 : CARRY_64;
      } else {
        log("-carry expects short|long\n");
        throw "-carry expects short|long";
      }
    } else if (key == "-block") {
      blockSize = stoi(s);
      if (blockSize != 1000 && blockSize != 500 && blockSize != 200) {
        log("-block must be one of 1000, 500, 200\n");
        throw "invalid block size";
      }
    } else if (key == "-log") {
      logStep = stoi(s);
      if (logStep == 0 || logStep % 1000 != 0) {       // 0 would divide by zero in the PRP loop
        log("-log must be a positive multiple of 1000\n");
        throw "invalid log size";
      }
    } else if (key == "-use") {
      for (const auto& [key, val] : splitUses(s)) {
        auto it = flags.find(key);
        if (it != flags.end() && it->second != val) {
          log("warning: -use %s=%s overrides %s=%s\n", key.c_str(), val.c_str(), it->first.c_str(), it->second.c_str());
        }
        flags[key] = val;
        if (!fromConfigFile) { cliKeys.insert(key); }
      }
    } else if (key == "-unsafeMath") {                                  // DEPRECATED, not in -help.  The flag has not reached the compiler since 424a54e,
      safeMath = false;                                                 // and measured on gfx1100 -cl-unsafe-math-optimizations gives no speedup and a lower
                                                                        // roundoff margin (reassoc folds fancyMul's fma).  Parsing left in place so previous
                                                                        // users do not get an error; safeMath kept in case a developer wants to try it again.
    } else if (key == "-save") {
      int const n = stoi(s);
      if (n < 1) {                                     // 0 makes Saver::trimFiles index v[-1]
        log("-save must be at least 1\n");
        throw "invalid -save value";
      }
      nSavefiles = n;
    } else {
      log("Argument '%s' '%s' not understood\n", key.c_str(), s.c_str());
      throw "args";
    }
  }
}

void Args::setDefaults() {
  uid = getUidFromPos(device);
  cl_device_id dev = getDevice(device);
  log("device %d, OpenCL %s, %s, unique id '%s'\n", device, getDriverVersionByPos(device).c_str(),
      isAmdGpu(dev) ? getBoardName(dev).c_str() : getDeviceName(dev).c_str(), uid.c_str());
  
  if (!masterDir.empty()) {
    assert(masterDir.is_absolute());
    for (filesystem::path* p : {&proofResultDir, &proofToVerifyDir, &cacheDir}) {
      if (p->is_relative()) { *p = masterDir / *p; }
    }
  }

  for (auto& p : {proofResultDir, proofToVerifyDir, cacheDir}) { fs::create_directory(p); }
}
