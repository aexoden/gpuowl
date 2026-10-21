# The tuner's sources, included by CMakeLists.txt.  Kept apart from it so that upstream's own list merges without
# conflicts.
target_sources(prpll PRIVATE
  OptionSpace.cpp
  FFTVariants.cpp
  Occupancy.cpp
  )
