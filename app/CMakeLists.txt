cmake_minimum_required(VERSION 3.16) # Helps to have this repeated in case you want this to be the root of seperate project

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set(FIRST_PRINCIPALS_NN_DEMO ${PROJECT_NAME_STR}_demo)

set(FIRST_PRINCIPALS_NN_DEMO_SRC 
    main.cpp
    )
    
add_executable(${FIRST_PRINCIPALS_NN_DEMO} ${FIRST_PRINCIPALS_NN_DEMO_SRC})

#--  inih  ------------------------------
find_path(INIH_INC_PATH INIReader.h) 
message("INIH_INC_PATH is " ${INIH_INC_PATH})
target_include_directories(${FIRST_PRINCIPALS_NN_DEMO} PUBLIC ${INIH_INC_PATH})
# Finding INIH debug and release versions was hard. INIH did not seem to have the necessary
# cmake support files so find_package(inih) though errors. I had to force find_library to
# look in certain directories and I had to set a system Envionment Variable VCPKG_ROOT
# so as not to have to hard code the full directory which obviously would not work on 
# a different machine
message("VCPKG_ROOT roos is " $ENV{VCPKG_ROOT})
FIND_LIBRARY(INIH_INC_LIB_DEBUG
  NAMES inih # Change name here if debug lib has different name.
  #PATHS E:/SD/Programming/C++/vcpkg/installed/x64-windows/debug/lib
  PATHS $ENV{VCPKG_ROOT}/installed/x64-windows/debug/lib
  NO_DEFAULT_PATH
  )
message("INIH_INC_LIB_DEBUG is " ${INIH_INC_LIB_DEBUG})
FIND_LIBRARY(INIH_INC_LIB_RELEASE
  NAMES inih
  #PATHS E:/SD/Programming/C++/vcpkg/installed/x64-windows/lib
  PATHS $ENV{VCPKG_ROOT}/installed/x64-windows/lib
  NO_DEFAULT_PATH
  )
message("INIH_INC_LIB_RELEASE is " ${INIH_INC_LIB_RELEASE})
#--  inih  ------------------------------


SET(INIH_INC_LIB
  debug ${INIH_INC_LIB_DEBUG}
  optimized ${INIH_INC_LIB_RELEASE}
  )
message("INIH_INC_LIB is " ${INIH_INC_LIB})
target_link_libraries(${FIRST_PRINCIPALS_NN_DEMO} PRIVATE ${INIH_INC_LIB})


#----------------------------------------

# The keyword 'PRIVATE' indicates that we are setting the build requirements between this <target> (monte_carlo_derivaitves_demo) 
# and the dependency (our library file). Use 'INTERFACE' for usage requirements and 'PUBLIC' for both build and usage requirements.
# You should not ommit one of these keywords, even though the macro allows you to do so - it may behave erratically
# find_library(${FIRST_PRINCIPALS_NN_LIBRARY})
message("My project libraries path is "  ${PROJECT_NAME_STR}_lib)
target_link_libraries(${FIRST_PRINCIPALS_NN_DEMO} PRIVATE  ${PROJECT_NAME_STR}_lib)