set(EvoNet_sources  CACHE INTERNAL "This variable should hold all EvoNet sources at the end of the config step" )

## ATTENTION: The order of includes should be similar to the inclusion hierarchy
include(source/core/sources.cmake)
include(source/ml/sources.cmake)
include(source/io/sources.cmake)

set(EvoNet_sources_h  CACHE INTERNAL "This variable should hold all EvoNet sources at the end of the config step" )

## ATTENTION: The order of includes should be similar to the inclusion hierarchy
include(include/EvoNet/core/sources.cmake)
include(include/EvoNet/graph/sources.cmake)
include(include/EvoNet/ml/sources.cmake)
include(include/EvoNet/models/sources.cmake)
include(include/EvoNet/simulator/sources.cmake)
include(include/EvoNet/io/sources.cmake)

## add configured config.h&Co to source group
source_group("Header Files\\EvoNet" FILES ${EvoNet_configured_headers})
## merge all headers to sources (for source group view in VS)
list(APPEND EvoNet_sources ${EvoNet_sources_h} ${EvoNet_configured_headers})

# TODO track why the duplicate warnings are thrown for all (!) MOC sources
# Macro problem?
list(REMOVE_DUPLICATES EvoNet_sources)
