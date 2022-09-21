cmake_minimum_required(VERSION 3.16) # Helps to have this repeated in case you want this to be the root of seperate project

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set(FIRST_PRINCIPALS_UNIT_TESTS ${PROJECT_NAME_STR}_tests)


set(MC_LIB_TEST_SRC
    test_main.cpp
    neuralnetworktest.cpp
    readwritematrixtocsvtest.cpp
    neuralNetworkMethodsTest.cpp
    inputdatafilereadertest.cpp
    functionstest.cpp
    neuralnetworklooptest.cpp
    )

add_executable(${FIRST_PRINCIPALS_UNIT_TESTS} ${MC_LIB_TEST_SRC})

# To find and use catch2
find_package(Catch2 3 REQUIRED)
target_link_libraries(${FIRST_PRINCIPALS_UNIT_TESTS} PRIVATE Catch2::Catch2WithMain)

#-- oneTBB ------------------------------
find_package(TBB CONFIG REQUIRED)
target_link_libraries(${FIRST_PRINCIPALS_UNIT_TESTS} PRIVATE TBB::tbb)
#----------------------------------------


# The keyword 'PRIVATE' indicates that we are setting the build requirements between this <target> (monte_carlo_derivaitves_tests)
# and the dependency (our library file). Use 'INTERFACE' for usage requirements and 'PUBLIC' for both build and usage requirements.
# You should not ommit one of these keywords, even though the macro allows you to do so - it may behave erratically

target_link_libraries(${FIRST_PRINCIPALS_UNIT_TESTS} PRIVATE ${PROJECT_NAME_STR}_lib)

add_test(NAME UnitTests COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${FIRST_PRINCIPALS_UNIT_TESTS})