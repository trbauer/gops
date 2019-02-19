cmake_minimum_required(VERSION 3.0)

project(gops)


set(GOPS_VERSION_STRING "0.9.0")
add_definitions(-DGOPS_VERSION_STRING="${GOPS_VERSION_STRING}")

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  set(TARGET_MODIFIER "32")
  set(TARGET_MODIFIER_PATH "x86")
  add_definitions(-DGOPS_HOST_POINTER_SIZE=32)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(TARGET_MODIFIER "64")
  set(TARGET_MODIFIER_PATH "x64")
  add_definitions(-DGOPS_HOST_POINTER_SIZE=64)
else()
  message(FATAL_ERROR "unexpected platform")
endif()


###############################################################################
# gops##.exe
###############################################################################
file(GLOB GOPS_ROOT
  "src/*.hpp" "src/*.cpp"
)

source_group("Source" FILES ${GOPS_ROOT})
set(GOPS_SOURCES ${GOPS_ROOT})

add_executable("gops${TARGET_MODIFIER}"
  ${GOPS_SOURCES}
  )

###############################################################################
# enable parallel build
if (MSVC)
  target_compile_options("gops${TARGET_MODIFIER}" PRIVATE "/MP")
  add_compile_options("/w4")
endif()

###############################################################################
# Include OpenCL headers and stub libraries.
# Note, I'm using the latest Intel OpenCL SDK.
#
# TODO: need to ensure it's a 2.0 version when we start using SVM
# find_package(OpenCL REQUIRED)
# find_package(OpenCL REQUIRED HINTS "C:\\Intel\\OpenCL\\sdk")
# include_directories(${OpenCL_INCLUDE_DIRS})
# target_link_libraries("gops${TARGET_MODIFIER}" ${OpenCL_LIBRARIES})
#
# NOTE: the above aren't working because find_package(OpenCL ...) does
# find a OpenCL 2.0 version of the library
# (c.f. $CMAKE_ROOT/share/cmake-3.11/Modules/FindOpenCL.cmake)
# This could be because Intel moved their SDK paths around recently
# (I did have env $INTELOCLSDKROOT set).
# For now we just set the paths explicitly.
#
# Maybe it makes sense to not use find_package(...) because this tool has
# such sensitive needs (e.g. needs binding points for SVM and KHR stubs).
# Some SDKs do not have new enough libraries and miss some of these.
#
set(OpenCL_INCLUDE_DIRS "C:\\Intel\\OpenCL\\sdk\\include")
if(CMAKE_SIZEOF_VOID_P EQUAL 4)
set(OpenCL_LIBRARIES "C:\\Intel\\OpenCL\\sdk\\lib\\x86\\OpenCL.lib")
else()
set(OpenCL_LIBRARIES "C:\\Intel\\OpenCL\\sdk\\lib\\x64\\OpenCL.lib")
endif()
include_directories(${OpenCL_INCLUDE_DIRS})
target_link_libraries("gops${TARGET_MODIFIER}" ${OpenCL_LIBRARIES})

set_target_properties("gops${TARGET_MODIFIER}" PROPERTIES
  CXX_STANDARD 17
  OUTPUT_NAME  "gops${TARGET_MODIFIER}"
)
