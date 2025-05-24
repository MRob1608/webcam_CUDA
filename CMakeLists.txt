cmake_minimum_required(VERSION 3.10)
project(CUDA_webcam LANGUAGES C CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xlinker --as-needed")


find_package(X11 REQUIRED)

# Imposta la directory di output
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(CMAKE_CUDA_LINK_FLAGS "${CMAKE_CUDA_LINK_FLAGS} --no-warn")  #provvisorio


# Aggiunge la directory 'include' come directory di ricerca per i file header
include_directories(${X11_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Aggiunge la libreria (statica)
add_library(mylib STATIC
    capture_camera.c
    conversion.c
    interface.c
)

add_library(cudalib STATIC
    conversion_CUDA.cu
    image_manipulation.cu
)

set_target_properties(cudalib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 11
)

target_include_directories(cudalib PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Aggiunge l'eseguibile
add_executable(main_exec main.cu)



# Collega la libreria all'eseguibile
target_link_libraries(main_exec PRIVATE mylib cudalib X11::X11)


# Imposta la cartella include se hai header personalizzati
target_include_directories(main_exec PRIVATE ${CMAKE_SOURCE_DIR}/src)
#target_include_directories(mylib PRIVATE ${CMAKE_SOURCE_DIR}/src)

