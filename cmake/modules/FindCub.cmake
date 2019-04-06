# 
# Try to find CUB  library  
# Once run this will define: 
# 
# CUB_FOUND
# CUB_INCLUDE_DIR 
#
# Use CUB_DIR to specify the directory where cub can be found
# --------------------------------

find_path(CUB_INCLUDE_DIR  
  cub/cub.cuh
  HINTS 
  ${CUB_DIR}
  ${CUB_INCLUDE_DIR}
  DOC "CUB headers"
 )

IF (CUB_INCLUDE_DIR)
    SET(CUB_FOUND TRUE)
ELSE (CUB_INCLUDE_DIR)
    MESSAGE("CUB include dir not found. Set CUB_DIR to find it.")
ENDIF(CUB_INCLUDE_DIR)

MARK_AS_ADVANCED(
  CUB_INCLUDE_DIR
)

include( FindPackageHandleStandardArgs ) 
find_package_handle_standard_args( CUB 
    REQUIRED_VARS 
        CUB_INCLUDE_DIR 
) 
