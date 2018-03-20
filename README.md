# lucrezio_semantic_perception

### Description

This ros package contains a node that simulates an object detector in a Gazebo simulation. 

It subscribes to the following topics:

* /gazebo/logical_camera_image: message produced by a Gazebo plugin that contains the set of objects currently seen by the robot
* /camera/rgb/image_raw: RGB image acquired with Xtion sensor
* /camera/depth/points: Depth point cloud acquired with Xtion sensor

It publishes to the following topics:

* /image_bounding_boxes: message containing the actual detected objects
* /camera/rgb/label_image: RGB image containing pixelwise annotations

### Usage

    rosrun lucrezio_semantic_perception object_detector_node

### TODO

* **Refactoring:** Remove `Detection` class to make smarter computations.
* **Debug:** Serialize data to run node offline. 
