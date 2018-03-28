add_library(lucrezio_semantic_perception_library SHARED
  detection.cpp detection.h
  model.cpp model.h
  image_utils.cpp image_utils.h
  object_detector.cpp object_detector.h
)

target_link_libraries(lucrezio_semantic_perception_library
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
)
