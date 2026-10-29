// Copyright (C) Jason Lynch

#include "Telemetry.h"

#include "gpuid.h"
#include "timeutil.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#ifdef __linux__
#include <dlfcn.h>
#endif

namespace tune {

namespace {

// The first line of a sysfs file, or nothing.
std::optional<std::string> readFile(const std::filesystem::path& path) {
  std::ifstream in{path};
  if (!in) { return {}; }
  std::stringstream text;
  text << in.rdbuf();
  return text.str();
}

std::optional<double> readNumber(const std::filesystem::path& path) {
  std::optional<std::string> const text = readFile(path);
  if (!text) { return {}; }
  double v = 0;
  auto const [end, ec] = std::from_chars(text->data(), text->data() + text->size(), v);
  if (ec != std::errc{}) { return {}; }
  return v;
}

#ifdef __linux__

// The few NVML entry points read here, as nvml.h declares them; looked up at run time.
using NvmlDevice = struct NvmlDeviceSt*;
using NvmlInit = int (*)();
using NvmlByBusId = int (*)(const char*, NvmlDevice*);
using NvmlCount = int (*)(unsigned*);
using NvmlByIndex = int (*)(unsigned, NvmlDevice*);
using NvmlClock = int (*)(NvmlDevice, int, unsigned*);
using NvmlTemperature = int (*)(NvmlDevice, int, unsigned*);
using NvmlPower = int (*)(NvmlDevice, unsigned*);
using NvmlReasons = int (*)(NvmlDevice, unsigned long long*);

// nvmlFieldValue_t, for nvmlDeviceGetFieldValues.
struct NvmlFieldValue {
  unsigned fieldId = 0;
  unsigned scopeId = 0;
  long long timestamp = 0;
  long long latencyUsec = 0;
  int valueType = 0;   // nvmlValueType_t
  int nvmlReturn = 0;  // nvmlReturn_t
  union {
    double d;
    unsigned ui;
    unsigned long ul;
    unsigned long long ull;
    long long sll;
    int si;
  } value{};
};
using NvmlFields = int (*)(NvmlDevice, int, NvmlFieldValue*);

constexpr int NVML_SUCCESS = 0;
constexpr int NVML_CLOCK_SM = 1;
constexpr int NVML_CLOCK_MEM = 2;
constexpr int NVML_TEMPERATURE_GPU = 0;

// The power drawn now, in milliwatts, rather than nvmlDeviceGetPowerUsage's figure, which on many cards is an average
// over the last second or more.  Drivers from about R530.
constexpr unsigned NVML_FI_DEV_POWER_INSTANT = 186;

class Nvml final : public Telemetry {
public:
  static std::unique_ptr<Telemetry> open(const std::string& bdf, std::string& why) {
    void* const lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
      why = "libnvidia-ml.so.1 could not be loaded";
      return nullptr;
    }
    auto sym = [&](const char* name) { return dlsym(lib, name); };
    auto init = NvmlInit(sym("nvmlInit_v2"));
    auto byBusId = NvmlByBusId(sym("nvmlDeviceGetHandleByPciBusId_v2"));
    auto count = NvmlCount(sym("nvmlDeviceGetCount_v2"));
    auto byIndex = NvmlByIndex(sym("nvmlDeviceGetHandleByIndex_v2"));
    auto out = std::unique_ptr<Nvml>(new Nvml);
    out->clock_ = NvmlClock(sym("nvmlDeviceGetClockInfo"));
    out->temperature_ = NvmlTemperature(sym("nvmlDeviceGetTemperature"));
    out->power_ = NvmlPower(sym("nvmlDeviceGetPowerUsage"));
    out->cap_ = NvmlPower(sym("nvmlDeviceGetEnforcedPowerLimit"));
    // Renamed from throttle reasons to clock event reasons; drivers keep both names.
    out->reasons_ = NvmlReasons(sym("nvmlDeviceGetCurrentClocksEventReasons"));
    if (!out->reasons_) { out->reasons_ = NvmlReasons(sym("nvmlDeviceGetCurrentClocksThrottleReasons")); }
    // Millijoules since the driver last loaded; Volta and later.  Older cards answer only power readings.
    out->energy_ = NvmlReasons(sym("nvmlDeviceGetTotalEnergyConsumption"));
    out->fields_ = NvmlFields(sym("nvmlDeviceGetFieldValues"));
    if (!init || !byBusId || !count || !byIndex || !out->clock_) {
      why = "libnvidia-ml.so.1 lacks the functions read here";
      return nullptr;
    }
    if (init() != NVML_SUCCESS) {
      why = "NVML did not initialise";
      return nullptr;
    }
    bool found = false;
    if (!bdf.empty()) {
      for (const char* domain : {"00000000:", "0000:"}) {
        out->busId_ = domain + bdf;
        if (byBusId(out->busId_.c_str(), &out->device_) == NVML_SUCCESS) {
          found = true;
          break;
        }
      }
    } else if (unsigned n = 0; count(&n) == NVML_SUCCESS && n == 1) {
      found = byIndex(0, &out->device_) == NVML_SUCCESS;
      out->busId_ = "the only NVIDIA card";
    }
    if (!found) {
      why = bdf.empty() ? "the card's PCI address is unknown and there is more than one NVIDIA card"
                        : "NVML has no card at " + bdf;
      return nullptr;
    }
    out->instant_ = out->instantPowerW().has_value();
    return out;
  }

  // The instantaneous power field, where the driver answers it.
  std::optional<double> instantPowerW() const {
    if (!fields_) { return {}; }
    NvmlFieldValue f;
    f.fieldId = NVML_FI_DEV_POWER_INSTANT;
    if (fields_(device_, 1, &f) != NVML_SUCCESS || f.nvmlReturn != NVML_SUCCESS) { return {}; }
    switch (f.valueType) {
    case 0: return f.value.d / 1000;
    case 1: return f.value.ui / 1000.0;
    case 2: return double(f.value.ul) / 1000;
    case 3: return double(f.value.ull) / 1000;
    case 4: return double(f.value.sll) / 1000;
    case 5: return f.value.si / 1000.0;
    default: return {};
    }
  }

  std::optional<CardSample> read() override {
    CardSample s;
    unsigned v = 0;
    bool any = false;
    if (clock_(device_, NVML_CLOCK_SM, &v) == NVML_SUCCESS) {
      s.smMHz = v;
      any = true;
    }
    if (clock_(device_, NVML_CLOCK_MEM, &v) == NVML_SUCCESS) { s.memMHz = v; }
    if (temperature_ && temperature_(device_, NVML_TEMPERATURE_GPU, &v) == NVML_SUCCESS) { s.tempC = v; }
    if (std::optional<double> const now = instant_ ? instantPowerW() : std::nullopt) {
      s.powerW = *now;
    } else if (power_ && power_(device_, &v) == NVML_SUCCESS) {
      s.powerW = v / 1000.0;
    }
    if (cap_ && cap_(device_, &v) == NVML_SUCCESS) { s.capW = v / 1000.0; }
    if (unsigned long long bits = 0; reasons_ && reasons_(device_, &bits) == NVML_SUCCESS) { s.held = bits; }
    if (unsigned long long mj = 0; energy_ && energy_(device_, &mj) == NVML_SUCCESS) { s.energyJ = double(mj) / 1000; }
    if (!any) { return {}; }
    return s;
  }

  std::string source() const override {
    return "NVML, " + busId_ + (instant_ ? ", instantaneous power" : ", averaged power") +
      (energy_ ? ", energy counter" : "");
  }
  bool saysWhy() const override { return reasons_ != nullptr; }
  bool instantPower() const override { return instant_; }

private:
  Nvml() = default;

  NvmlDevice device_ = nullptr;
  std::string busId_;
  NvmlClock clock_ = nullptr;
  NvmlTemperature temperature_ = nullptr;
  NvmlPower power_ = nullptr;
  NvmlPower cap_ = nullptr;
  NvmlReasons reasons_ = nullptr;
  NvmlReasons energy_ = nullptr;
  NvmlFields fields_ = nullptr;
  bool instant_ = false;
};

#endif

class AmdSysfs final : public Telemetry {
public:
  static std::unique_ptr<Telemetry> open(const std::string& bdf, std::string& why) {
    int const card = bdf.empty() ? -1 : getSysfsFromBdf(bdf);
    if (card < 0) {
      why = bdf.empty() ? "the card's PCI address is unknown" : "no amdgpu card in sysfs at " + bdf;
      return nullptr;
    }
    auto out = std::unique_ptr<AmdSysfs>(new AmdSysfs);
    out->card_ = card;
    out->device_ = "/sys/class/drm/card" + std::to_string(card) + "/device";
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(out->device_ / "hwmon", ec)) {
      out->hwmon_ = entry.path();
      break;
    }
    if (!out->read()) {
      why = "card" + std::to_string(card) + " has no pp_dpm_sclk to read";
      return nullptr;
    }
    return out;
  }

  std::optional<CardSample> read() override {
    CardSample s;
    std::optional<std::string> const sclk = readFile(device_ / "pp_dpm_sclk");
    std::optional<double> const sm = sclk ? currentDpmMHz(*sclk) : std::nullopt;
    if (!sm) { return {}; }
    s.smMHz = *sm;
    if (std::optional<std::string> const mclk = readFile(device_ / "pp_dpm_mclk")) {
      s.memMHz = currentDpmMHz(*mclk).value_or(NAN);
    }
    if (!hwmon_.empty()) {
      if (auto t = readNumber(hwmon_ / "temp1_input")) { s.tempC = *t / 1000; }
      auto p = readNumber(hwmon_ / "power1_average");
      if (!p) { p = readNumber(hwmon_ / "power1_input"); }
      if (p) { s.powerW = *p / 1e6; }
      if (auto c = readNumber(hwmon_ / "power1_cap")) { s.capW = *c / 1e6; }
      if (auto e = readNumber(hwmon_ / "energy1_input")) { s.energyJ = *e / 1e6; }  // microjoules, where amdgpu has it
    }
    return s;
  }

  std::string source() const override { return "amdgpu sysfs, card" + std::to_string(card_); }

private:
  AmdSysfs() = default;

  int card_ = -1;
  std::filesystem::path device_;
  std::filesystem::path hwmon_;
};

double median(std::vector<double> v) {
  if (v.empty()) { return NAN; }
  std::ranges::sort(v);
  size_t const n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

// NVML's clock event reason bits, most telling first.  The idle bit (1) is left out.
constexpr std::pair<u64, const char*> HELD[] = {
  {0x4, "power cap"},           {0x20, "thermal"},           {0x40, "hardware thermal"}, {0x8, "hardware slowdown"},
  {0x80, "power brake"},        {0x2, "application clocks"}, {0x10, "sync boost"},       {0x100, "display clock"},
};

}  // namespace

std::unique_ptr<Telemetry> openTelemetry(const std::string& bdf, bool nvidia, bool amd, std::string& why) {
#ifdef __linux__
  if (nvidia) { return Nvml::open(bdf, why); }
  if (amd) { return AmdSysfs::open(bdf, why); }
  why = "the card is neither NVIDIA nor AMD";
#else
  (void)bdf, (void)nvidia, (void)amd;
  why = "card readings are only taken on Linux";
#endif
  return nullptr;
}

std::unique_ptr<Telemetry> openTelemetryFor(cl_device_id id, std::string& why) {
  return openTelemetry(pciAddressOf(id), isNvidiaGpu(id), isAmdGpu(id), why);
}

std::string pciAddressOf(cl_device_id id) {
  std::string bdf = getBdfFromDevice(id);
  if (!bdf.empty()) { return bdf; }
  constexpr cl_device_info PCI_BUS_ID_NV = 0x4008;
  constexpr cl_device_info PCI_SLOT_ID_NV = 0x4009;
  u32 bus = 0, slot = 0;  // cl_uint
  if (clGetDeviceInfo(id, PCI_BUS_ID_NV, sizeof(bus), &bus, nullptr) == CL_SUCCESS &&
      clGetDeviceInfo(id, PCI_SLOT_ID_NV, sizeof(slot), &slot, nullptr) == CL_SUCCESS) {
    char buf[16];
    snprintf(buf, sizeof(buf), "%02x:%02x.%x", bus, slot >> 3, slot & 7);
    return buf;
  }
  return {};
}

Sampler::Sampler(Telemetry* telemetry, double period, double keep)
  : telemetry_{telemetry}, keep_{keep}, start_{std::chrono::steady_clock::now()} {
  if (!telemetry_) { return; }
  thread_ = std::jthread{[this, period](std::stop_token stop) {
    while (!stop.stop_requested()) {
      if (std::optional<CardSample> s = telemetry_->read()) {
        s->t = now();
        std::lock_guard lock{mutex_};
        samples_.push_back(*s);
        // Dropped a batch at a time, so the vector is not shifted on every sample.
        if (keep_ > 0 && samples_.front().t < s->t - 2 * keep_) {
          std::erase_if(samples_, [&](const CardSample& old) { return old.t < s->t - keep_; });
        }
      }
      std::this_thread::sleep_for(std::chrono::duration<double>(period));
    }
  }};
}

Sampler::~Sampler() {
  if (thread_.joinable()) {
    thread_.request_stop();
    thread_.join();
  }
}

double Sampler::now() const { return at(std::chrono::steady_clock::now()); }

double Sampler::at(std::chrono::steady_clock::time_point t) const {
  return std::chrono::duration<double>(t - start_).count();
}

std::vector<CardSample> Sampler::between(double from, double to) const {
  std::lock_guard lock{mutex_};
  std::vector<CardSample> out;
  for (const CardSample& s : samples_) {
    if (s.t >= from && s.t <= to) { out.push_back(s); }
  }
  return out;
}

bool Sampler::waitPast(double t, double most) const {
  if (!telemetry_) { return false; }
  for (Timer waited; waited.at() < most; std::this_thread::sleep_for(std::chrono::milliseconds(10))) {
    std::lock_guard lock{mutex_};
    if (!samples_.empty() && samples_.back().t > t) { return true; }
  }
  return false;
}

std::optional<WindowPower> powerOver(std::span<const CardSample> samples, double from, double to, bool instant) {
  if (!(to > from) || samples.empty()) { return {}; }
  WindowPower out;

  // The energy counter at `t`, between the samples either side of it.
  auto energyAt = [&](double t) -> std::optional<double> {
    const CardSample* before = nullptr;
    const CardSample* after = nullptr;
    for (const CardSample& s : samples) {
      if (std::isnan(s.energyJ)) { continue; }
      if (s.t <= t) {
        before = &s;
      } else {
        after = &s;
        break;
      }
    }
    if (!before || !after) { return {}; }
    return before->energyJ + (after->energyJ - before->energyJ) * (t - before->t) / (after->t - before->t);
  };
  std::optional<double> const e0 = energyAt(from);
  std::optional<double> const e1 = energyAt(to);

  // The samples inside the window, or the nearest one where none is.
  std::vector<const CardSample*> inside;
  for (const CardSample& s : samples) {
    if (s.t >= from && s.t <= to) { inside.push_back(&s); }
  }
  bool const between = inside.empty();
  if (between) {
    double const mid = (from + to) / 2;
    inside.push_back(&*std::ranges::min_element(samples, {}, [&](const CardSample& s) { return std::abs(s.t - mid); }));
  }

  if (e0 && e1 && *e1 >= *e0) { out.counterWatts = (*e1 - *e0) / (to - from); }
  // The median, not the mean: the first readings of a window can still be the card's before it, as a 4070 SUPER's
  // instantaneous power read 136-219 W over calls whose readings sat at 216 W but for one or two.
  std::vector<double> watts;
  for (const CardSample* s : inside) {
    if (!std::isnan(s->powerW)) { watts.push_back(s->powerW); }
  }
  out.readingWatts = median(watts);
  out.readings = between ? 0 : u32(watts.size());

  bool const byReadings = std::isnan(out.counterWatts) || (instant && out.readings >= INSTANT_READINGS);
  if (byReadings && std::isnan(out.readingWatts)) { return {}; }
  out.counted = !byReadings;
  out.watts = byReadings ? out.readingWatts : out.counterWatts;
  out.joules = out.watts * (to - from);
  std::vector<double> caps, clocks;
  for (const CardSample* s : inside) {
    if (!std::isnan(s->capW)) { caps.push_back(s->capW); }
    if (!std::isnan(s->smMHz)) { clocks.push_back(s->smMHz); }
  }
  out.capW = median(caps);
  out.smMHz = median(clocks);
  return out;
}

std::optional<WindowPower> powerOver(const Sampler& sampler, std::chrono::steady_clock::time_point from,
                                     std::chrono::steady_clock::time_point to) {
  double const a = sampler.at(from);
  double const b = sampler.at(to);
  std::vector<CardSample> const samples = sampler.between(a - 1, b + 1);
  return powerOver(samples, a, b, sampler.instantPower());
}

std::optional<double> currentDpmMHz(std::string_view ppDpm) {
  while (!ppDpm.empty()) {
    size_t const eol = std::min(ppDpm.find('\n'), ppDpm.size());
    std::string_view const line = ppDpm.substr(0, eol);
    ppDpm.remove_prefix(std::min(eol + 1, ppDpm.size()));
    if (line.find('*') == std::string_view::npos) { continue; }
    size_t const colon = line.find(':');
    if (colon == std::string_view::npos) { continue; }
    std::string_view rest = line.substr(colon + 1);
    while (!rest.empty() && rest.front() == ' ') { rest.remove_prefix(1); }
    double mhz = 0;
    auto const [end, ec] = std::from_chars(rest.data(), rest.data() + rest.size(), mhz);
    if (ec == std::errc{}) { return mhz; }
  }
  return {};
}

std::string heldText(u64 bits) {
  std::string out;
  for (auto const& [bit, name] : HELD) {
    if (bits & bit) { out += (out.empty() ? "" : ", ") + std::string{name}; }
  }
  return out;
}

CardSummary summarizeCard(std::span<const CardSample> samples) {
  CardSummary out;
  out.n = u32(samples.size());
  if (samples.empty()) { return out; }
  std::vector<double> sm, mem, temp, power, cap;
  for (const CardSample& s : samples) {
    for (auto [v, to] : {std::pair{s.smMHz, &sm}, {s.memMHz, &mem}, {s.tempC, &temp}, {s.powerW, &power},
                         {s.capW, &cap}}) {
      if (!std::isnan(v)) { to->push_back(v); }
    }
  }
  if (!sm.empty()) {
    out.smMin = std::ranges::min(sm);
    out.smMax = std::ranges::max(sm);
    out.smMedian = median(sm);
  }
  out.memMedian = median(mem);
  out.tempMedian = median(temp);
  out.tempMax = temp.empty() ? NAN : std::ranges::max(temp);
  out.powerMedian = median(power);
  out.capW = median(cap);
  for (auto const& [bit, name] : HELD) {
    u32 const n = u32(std::ranges::count_if(samples, [bit](const CardSample& s) { return (s.held & bit) != 0; }));
    if (n) { out.held.emplace_back(name, double(n) / double(samples.size())); }
  }
  return out;
}

std::string toString(const CardSummary& s, bool saysWhy) {
  if (s.n == 0) { return {}; }
  char buf[256];
  std::string out;
  if (!std::isnan(s.smMedian)) {
    snprintf(buf, sizeof(buf), "SM %.0f MHz (%.0f-%.0f)", s.smMedian, s.smMin, s.smMax);
    out += buf;
  }
  if (!std::isnan(s.memMedian)) {
    snprintf(buf, sizeof(buf), ", mem %.0f MHz", s.memMedian);
    out += buf;
  }
  if (!std::isnan(s.tempMedian)) {
    snprintf(buf, sizeof(buf), ", %.0f C (max %.0f)", s.tempMedian, s.tempMax);
    out += buf;
  }
  if (!std::isnan(s.powerMedian)) {
    if (!std::isnan(s.capW)) {
      snprintf(buf, sizeof(buf), ", %.0f/%.0f W", s.powerMedian, s.capW);
    } else {
      snprintf(buf, sizeof(buf), ", %.0f W", s.powerMedian);
    }
    out += buf;
  }
  if (saysWhy) {
    if (s.held.empty()) {
      out += "; never held back";
    } else {
      std::string held;
      for (auto const& [name, share] : s.held) {
        snprintf(buf, sizeof(buf), "%s%s %.0f%%", held.empty() ? "" : ", ", name.c_str(), share * 100);
        held += buf;
      }
      out += "; held back by " + held + " of samples";
    }
  }
  return out;
}

}  // namespace tune
