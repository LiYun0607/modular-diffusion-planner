cmake_minimum_required(VERSION 3.14)
project(autoware_diffusion_planner_onnx_split_cpp)

# === Compiler settings ===
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

# === Find ROS2 dependencies ===
find_package(ament_cmake REQUIRED)

# Check for Autoware messages (optional for node component)
find_package(rclcpp QUIET)
find_package(rclcpp_components QUIET)
find_package(autoware_planning_msgs QUIET)
find_package(autoware_perception_msgs QUIET)
find_package(autoware_vehicle_msgs QUIET)
find_package(autoware_map_msgs QUIET)
find_package(autoware_internal_planning_msgs QUIET)
find_package(autoware_vehicle_info_utils QUIET)

# Disable ROS node component for now due to API incompatibilities
# To enable, set -DENABLE_ROS_NODE=ON and ensure compatible autoware_diffusion_planner
option(ENABLE_ROS_NODE "Build ROS node component (requires compatible autoware_diffusion_planner)" OFF)

set(BUILD_ROS_NODE OFF)
if(ENABLE_ROS_NODE)
  find_package(autoware_diffusion_planner QUIET
    HINTS /opt/autoware/share/autoware_diffusion_planner/cmake
          ${autoware_diffusion_planner_DIR})
  if(rclcpp_FOUND AND rclcpp_components_FOUND AND
     autoware_planning_msgs_FOUND AND autoware_perception_msgs_FOUND AND
     autoware_vehicle_msgs_FOUND AND autoware_map_msgs_FOUND AND
     autoware_diffusion_planner_FOUND)
    set(BUILD_ROS_NODE ON)
    message(STATUS "All dependencies found - building ROS node component")
  else()
    message(STATUS "Missing dependencies - building core library only")
  endif()
else()
  message(STATUS "ROS node component disabled - building core library only")
endif()

# === ONNX Runtime Setup ===
set(ONNXRUNTIME_ROOT "/tmp/onnxruntime-linux-x64-gpu-1.23.2")

find_library(ONNXRUNTIME_LIBRARY onnxruntime
  HINTS
    ${ONNXRUNTIME_ROOT}/lib
    /usr/local/lib
    /usr/lib
    $ENV{ONNXRUNTIME_ROOT}/lib
)

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS
    ${ONNXRUNTIME_ROOT}/include
    /usr/local/include
    $ENV{ONNXRUNTIME_ROOT}/include
)

if(ONNXRUNTIME_LIBRARY AND ONNXRUNTIME_INCLUDE_DIR)
  message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIBRARY}")
  message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "ONNX Runtime not found! Please install it or set ONNXRUNTIME_ROOT")
endif()

# === Build Core Library ===
add_library(${PROJECT_NAME}_core SHARED
  src/diffusion_planner.cpp
)

target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME}_core
  ${ONNXRUNTIME_LIBRARY}
)

# === Build ROS2 Node Component (if dependencies available) ===
if(BUILD_ROS_NODE)
  find_package(std_msgs REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(visualization_msgs REQUIRED)
  find_package(autoware_utils_diagnostics REQUIRED)
  find_package(diagnostic_msgs REQUIRED)

  add_library(${PROJECT_NAME}_component SHARED
    src/diffusion_planner_node.cpp
  )

  target_include_directories(${PROJECT_NAME}_component PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${ONNXRUNTIME_INCLUDE_DIR}
  )

  target_link_libraries(${PROJECT_NAME}_component
    ${PROJECT_NAME}_core
    ${ONNXRUNTIME_LIBRARY}
  )

  ament_target_dependencies(${PROJECT_NAME}_component
    rclcpp
    rclcpp_components
    std_msgs
    geometry_msgs
    nav_msgs
    visualization_msgs
    autoware_planning_msgs
    autoware_perception_msgs
    autoware_vehicle_msgs
    autoware_map_msgs
    autoware_internal_planning_msgs
    autoware_vehicle_info_utils
    autoware_diffusion_planner  # Required for preprocessing utilities
    autoware_utils_diagnostics  # Required for auto mode!
    diagnostic_msgs
  )

  rclcpp_components_register_node(${PROJECT_NAME}_component
    PLUGIN "autoware::diffusion_planner_onnx_split::DiffusionPlannerNode"
    EXECUTABLE ${PROJECT_NAME}_node
  )

  install(
    TARGETS ${PROJECT_NAME}_component
    EXPORT export_${PROJECT_NAME}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )
endif()

# === Install ===
install(DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS ${PROJECT_NAME}_core
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

# === Testing ===
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
