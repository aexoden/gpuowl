# The tuner's sources, included by CMakeLists.txt.  Kept apart from it so that upstream's own list merges without
# conflicts.
target_sources(prpll PRIVATE
  OptionSpace.cpp
  FFTVariants.cpp
  Occupancy.cpp
  Restart.cpp
  GpuFault.cpp
  Stats.cpp
  Eligibility.cpp
  BuildId.cpp
  Anchor.cpp
  TuneDB.cpp
  LLCheck.cpp
  Measure.cpp
  Axes.cpp
  Model.cpp
  MinSum.cpp
  Branches.cpp
  Optimize.cpp
  Scope.cpp
  TuneCommand.cpp
  Readings.cpp
  Gate.cpp
  Candidates.cpp
  Publish.cpp
  Acquire.cpp
  Tuner.cpp
  Dashboard.cpp
  )
