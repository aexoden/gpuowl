# The tuner's sources, unit tests and source checkers, included by the Makefile.  Kept apart from it so that upstream's
# own lists merge without conflicts.

TUNE_SRCS = \
  OptionSpace.cpp \
  FFTVariants.cpp \
  Occupancy.cpp \
  Restart.cpp \
  GpuFault.cpp \
  Stats.cpp \
  Eligibility.cpp \
  BuildId.cpp \
  Anchor.cpp \
  TuneDB.cpp \
  LLCheck.cpp \
  Measure.cpp \
  Axes.cpp \
  Model.cpp \
  MinSum.cpp \
  Branches.cpp \
  Coverage.cpp \
  Lanes.cpp \
  Explore.cpp \
  Audit.cpp \
  Optimize.cpp \
  Scope.cpp \
  TuneCommand.cpp \
  Readings.cpp \
  Gate.cpp \
  Candidates.cpp \
  Publish.cpp \
  Acquire.cpp \
  Tuner.cpp \
  Dashboard.cpp

TESTSRCS = \
  test_main.cpp \
  test_OptionSpace.cpp \
  test_FFTVariants.cpp \
  test_Occupancy.cpp \
  test_Restart.cpp \
  test_GpuFault.cpp \
  test_Stats.cpp \
  test_Eligibility.cpp \
  test_BuildId.cpp \
  test_Anchor.cpp \
  test_TuneDB.cpp \
  test_LLCheck.cpp \
  test_Measure.cpp \
  test_Model.cpp \
  test_MinSum.cpp \
  test_Optimize.cpp \
  test_Acquire.cpp \
  test_Coverage.cpp \
  test_Explore.cpp \
  test_Scope.cpp \
  test_TuneEntry.cpp \
  test_Gate.cpp \
  test_Publish.cpp \
  test_Tuner.cpp \
  test_Dashboard.cpp

# The prpll rules above were read with upstream's SRCS1 alone: their recipes link ${OBJS}, which is expanded when they
# run, so adding to SRCS1 here is enough for the link; the prerequisites are added below.
SRCS1 += $(TUNE_SRCS)
TUNE_OBJS = $(TUNE_SRCS:%.cpp=$(BIN)/%.o)

PYTHON ?= python3

ifeq ($(shell command -v $(PYTHON) >/dev/null 2>&1 && echo yes), yes)
 CHECKERS = $(BIN)/check-option-inventory.stamp $(BIN)/check-shufl-index.stamp
else
 CHECKERS =
 ifeq ($(filter clean,$(MAKECMDGOALS)),)
  $(info Note: $(PYTHON) not found; skipping the source checkers in tools/)
 endif
endif

$(BIN)/prpll $(BIN)/prpll-amd: $(TUNE_OBJS) | $(CHECKERS)

INVENTORY_SRCS = tools/check_option_inventory.py src/OptionSpace.cpp $(wildcard src/cl/*.cl) $(wildcard src/cuda/*.cuh) $(filter-out src/bundle.cpp,$(wildcard src/*.cpp)) $(wildcard src/*.h) $(wildcard src/cuda/*.cpp) $(wildcard src/cuda/*.h)
SHUFL_SRCS = tools/check_shufl_index.py src/cl/shufl.cl src/cl/fftbase.cl src/cl/base.cl src/cl/expand.cl src/FFTConfig.cpp src/FFTConfig.h

$(BIN)/check-option-inventory.stamp: $(INVENTORY_SRCS)
	$(PYTHON) tools/check_option_inventory.py
	@touch $@

$(BIN)/check-shufl-index.stamp: $(SHUFL_SRCS)
	$(PYTHON) tools/check_shufl_index.py
	@touch $@

.PHONY: check-source
check-source: $(CHECKERS)

.PHONY: check-tools
check-tools:
	$(PYTHON) -m unittest discover -s tools -p 'test_*.py'

# GPU-free unit tests (tests/).
TESTOBJS = $(TESTSRCS:%.cpp=$(BIN)/tests/%.o)

.PHONY: check
check: $(BIN)/prpll-test $(CHECKERS)
	$(BIN)/prpll-test

$(BIN)/prpll-test: $(filter-out $(BIN)/main.o,$(OBJS)) $(TESTOBJS) | $(CHECKERS)
	$(CXX) $(LDFLAGS) $(CXXFLAGS) -o $@ $^ $(LIBPATH) $(OPENCL_LIBS)

$(BIN)/tests/%.o : tests/%.cpp $(DEPDIR)/%.d
	@mkdir -p $(BIN)/tests
	$(COMPILE.cc) -Isrc $(OUTPUT_OPTION) $<
	$(POSTCOMPILE)

include $(wildcard $(patsubst %,$(DEPDIR)/%.d,$(basename $(TESTSRCS))))
