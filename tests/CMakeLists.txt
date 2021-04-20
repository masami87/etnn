add_executable(cublas "cublas.cpp")
target_link_libraries(cublas ${PROJECT_NAME})

add_executable(dataset "dataset.cpp")
target_link_libraries(dataset ${PROJECT_NAME})

add_executable(initial "initial.cpp")
target_link_libraries(initial ${PROJECT_NAME})

add_executable(softmax "softmax.cpp")
target_link_libraries(softmax ${PROJECT_NAME})

add_executable(tensor "tensor.cpp")
target_link_libraries(tensor ${PROJECT_NAME})


add_executable(json_test "json.cpp")
target_link_libraries(json_test ${PROJECT_NAME})