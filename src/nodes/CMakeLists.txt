add_executable(object_detector_node object_detector_node.cpp)

target_link_libraries(object_detector_node
  lucrezio_semantic_perception_library
  ${catkin_LIBRARIES}
)

