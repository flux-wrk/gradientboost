cmake_minimum_required (VERSION 3.5)

add_library(tbb SHARED IMPORTED GLOBAL) # or STATIC instead of SHARED

if((UNIX AND NOT APPLE) AND (${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64"))
    set_target_properties(tbb PROPERTIES
            IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/linux/intel64/gcc4.7/libtbb.so.2"
            INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include")
endif()

if (APPLE)
    set_target_properties(tbb PROPERTIES
            IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/mac/libtbb.dylib"
            INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include")
endif ()
