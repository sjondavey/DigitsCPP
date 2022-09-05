cmake_minimum_required(VERSION 3.16) # Helps to have this repeated in case you want this to be the root of seperate project

set(FIRST_PRINCIPALS_NN_LIBRARY ${PROJECT_NAME_STR}_lib)


set(FIRST_PRINCIPALS_NN_LIBRARY_LIB_HDR 
    neuralnetwork.h
    inputdatafilereader.h
    readwritematrixtocsv.h
    typedefs.h
    neuralNetworkMethods.h
    functions.h
    )

set(FIRST_PRINCIPALS_NN_LIBRARY_LIB_SRC 
    neuralnetwork.cpp
    inputdatafilereader.cpp
    readwritematrixtocsv.cpp
    neuralNetworkMethods.cpp
    functions.cpp
    )

# Making use of the target_sources(...) to add sources and headers rather than listing them in the add_library(...) macro. 
# target_sources allows sources to be declared as PRIVATE (necessary to build), INTERFACE (not required for build, 
# only use NOTE that headers are not needed for the build) and  PUBLIC (necessary of build and use). This control is 
# not possible in the add_library(...) macro
if(MSVC)
        add_library(${FIRST_PRINCIPALS_NN_LIBRARY} STATIC)
else()
        add_library(${FIRST_PRINCIPALS_NN_LIBRARY} SHARED)
endif()
target_sources(${FIRST_PRINCIPALS_NN_LIBRARY}
                PRIVATE ${FIRST_PRINCIPALS_NN_LIBRARY_LIB_SRC}
                INTERFACE ${FIRST_PRINCIPALS_NN_LIBRARY_LIB_HDR}) # Header not required for build, therefore 


#-- eigen3 ------------------------------
find_package(Eigen3 REQUIRED)
target_link_libraries(${FIRST_PRINCIPALS_NN_LIBRARY} PRIVATE Eigen3::Eigen)
#----------------------------------------
