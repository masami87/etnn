add_executable(mnist_mlp "mnist_mlp.cpp")
target_link_libraries(mnist_mlp ${PROJECT_NAME})

add_executable(mnist_mlp_distributed "mnist_mlp_distributed.cpp")
target_link_libraries(mnist_mlp_distributed ${PROJECT_NAME})

add_executable(json_mnist_mlp "json_mnist_mlp")
target_link_libraries(json_mnist_mlp ${PROJECT_NAME})