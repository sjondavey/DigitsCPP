cmake_minimum_required(VERSION 3.16) # Helps to have this repeated in case you want this to be the root of seperate project

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
set(FIRST_PRINCIPALS_NN_DEMO ${PROJECT_NAME_STR}_demo)

set(FIRST_PRINCIPALS_NN_DEMO_SRC 
    main.cpp
    )
    
add_executable(${FIRST_PRINCIPALS_NN_DEMO} ${FIRST_PRINCIPALS_NN_DEMO_SRC})

# The keyword 'PRIVATE' indicates that we are setting the build requirements between this <target> (monte_carlo_derivaitves_demo) 
# and the dependency (our library file). Use 'INTERFACE' for usage requirements and 'PUBLIC' for both build and usage requirements.
# You should not ommit one of these keywords, even though the macro allows you to do so - it may behave erratically

# target_link_libraries(${FIRST_PRINCIPALS_NN_DEMO} PRIVATE ${PROJECT_NAME_STR}_lib)