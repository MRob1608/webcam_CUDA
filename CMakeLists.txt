cmake_minimum_required(VERSION 3.10)
project(CUDA_webcam C)

find_package(X11 REQUIRED)

# Imposta la directory di output
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Aggiunge la directory 'include' come directory di ricerca per i file header
include_directories(${X11_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Aggiunge la libreria (statica)
add_library(mylib STATIC
    capture_camera.c
    conversion.c
    interface.c
)

# Aggiunge l'eseguibile
add_executable(main_exec main.c)

# Collega la libreria all'eseguibile
target_link_libraries(main_exec PRIVATE mylib X11::X11)


# Imposta la cartella include se hai header personalizzati
target_include_directories(main_exec PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(mylib PRIVATE ${CMAKE_SOURCE_DIR}/src)
