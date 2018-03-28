#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <Eigen/Core>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <lucrezio_simulation_environments/LogicalImage.h>

#include "tf/tf.h"
#include "tf/transform_datatypes.h"

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <lucrezio_semantic_perception/ImageBoundingBoxesArray.h>

#include <lucrezio_semantic_perception/object_detector.h>

#include <gazebo_msgs/GetModelState.h>

using namespace lucrezio_semantic_perception;


class ObjectDetectorNode : public ObjectDetector{
public:
  ObjectDetectorNode(ros::NodeHandle nh_):
    _nh(nh_),
    _logical_image_sub(_nh,"/gazebo/logical_camera_image",1),
    _depth_image_sub(_nh,"/camera/depth/image_raw",1),
    _rgb_image_sub(_nh,"/camera/rgb/image_raw", 1),
    _synchronizer(FilterSyncPolicy(10),_logical_image_sub,_depth_image_sub,_rgb_image_sub),
    _it(_nh){

    _got_info = false;
    _camera_info_sub = _nh.subscribe("/camera/depth/camera_info",
                                     1000,
                                     &ObjectDetectorNode::cameraInfoCallback,
                                     this);

    _synchronizer.registerCallback(boost::bind(&ObjectDetectorNode::filterCallback, this, _1, _2, _3));

    _model_state_client = _nh.serviceClient<gazebo_msgs::GetModelState>("gazebo/get_model_state");

    _image_bounding_boxes_pub = _nh.advertise<lucrezio_semantic_perception::ImageBoundingBoxesArray>("/image_bounding_boxes", 1);
    _label_image_pub = _it.advertise("/camera/rgb/label_image", 1);

    ROS_INFO("Starting detection simulator node!");
  }

  void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg){
    sensor_msgs::CameraInfo camerainfo;
    camerainfo.K = camera_info_msg->K;

    Eigen::Matrix3f K;

    ROS_INFO("Got camera info!");
    K(0,0) = camerainfo.K.c_array()[0];
    K(0,1) = camerainfo.K.c_array()[1];
    K(0,2) = camerainfo.K.c_array()[2];
    K(1,0) = camerainfo.K.c_array()[3];
    K(1,1) = camerainfo.K.c_array()[4];
    K(1,2) = camerainfo.K.c_array()[5];
    K(2,0) = camerainfo.K.c_array()[6];
    K(2,1) = camerainfo.K.c_array()[7];
    K(2,2) = camerainfo.K.c_array()[8];
    std::cerr << K << std::endl;

    setK(K);

    _got_info = true;
    _camera_info_sub.shutdown();
  }

  void filterCallback(const lucrezio_simulation_environments::LogicalImage::ConstPtr &logical_image_msg,
                      const sensor_msgs::Image::ConstPtr &depth_image_msg,
                      const sensor_msgs::Image::ConstPtr &rgb_image_msg){

    if(_got_info && !logical_image_msg->models.empty()){

      ROS_INFO("--------------------------");
      ROS_INFO("Executing filter callback!");
      ROS_INFO("--------------------------");
      std::cerr << std::endl;

      //Save timestamp
      _last_timestamp = logical_image_msg->header.stamp;

      //Extract rgb and depth image from ROS messages
      cv_bridge::CvImageConstPtr rgb_cv_ptr,depth_cv_ptr;
      try{
        rgb_cv_ptr = cv_bridge::toCvShare(rgb_image_msg);
        depth_cv_ptr = cv_bridge::toCvShare(depth_image_msg);
      } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }

      cv::Mat rgb_image = rgb_cv_ptr->image.clone();
      int rgb_rows=rgb_image.rows;
      int rgb_cols=rgb_image.cols;
      std::string rgb_type=type2str(rgb_image.type());
      ROS_INFO("Got %dx%d %s image",rgb_cols,rgb_rows,rgb_type.c_str());

      cv::Mat depth_image = depth_cv_ptr->image.clone();
      int depth_rows=depth_image.rows;
      int depth_cols=depth_image.cols;
      std::string depth_type=type2str(depth_image.type());
      ROS_INFO("Got %dx%d %s image",depth_cols,depth_rows,depth_type.c_str());

      setImages(rgb_image,depth_image);

      //Listen to camera pose
      gazebo_msgs::GetModelState model_state;
      model_state.request.model_name = "robot";
      tf::StampedTransform robot_pose;
      if(_model_state_client.call(model_state)){
        ROS_INFO("Received robot model state!");
        tf::poseMsgToTF(model_state.response.pose,robot_pose);
      }else
        ROS_ERROR("Failed to call service gazebo/get_model_state");

      Eigen::Isometry3f rgbd_camera_pose = Eigen::Isometry3f::Identity();
      rgbd_camera_pose.translation() = Eigen::Vector3f(0.0,0.0,0.5);
      rgbd_camera_pose.linear() = Eigen::Quaternionf(0.5,-0.5,0.5,-0.5).toRotationMatrix();

      tf::StampedTransform logical_camera_pose;
      tf::poseMsgToTF(logical_image_msg->pose,logical_camera_pose);

      setCameraTransforms(tfTransform2eigen(robot_pose)*rgbd_camera_pose,
                          tfTransform2eigen(logical_camera_pose));

      //process models
      const std::vector<lucrezio_simulation_environments::Model> &camera_models = logical_image_msg->models;
      int num_models=camera_models.size();
      ModelVector models;
      models.resize(num_models);
      std::string model_type;
      tf::StampedTransform model_pose;
      Eigen::Isometry3f model_transform;
      Eigen::Vector3f model_min;
      Eigen::Vector3f model_max;
      for(size_t i=0; i < num_models; ++i){
        model_type = camera_models[i].type;
        tf::poseMsgToTF(camera_models[i].pose,model_pose);
        model_transform = tfTransform2eigen(model_pose);
        model_min = Eigen::Vector3f(camera_models[i].min.x,camera_models[i].min.y,camera_models[i].min.z);
        model_max = Eigen::Vector3f(camera_models[i].max.x,camera_models[i].max.y,camera_models[i].max.z);
        models[i] = Model(model_type,
                          model_transform,
                          model_min,
                          model_max);
      }

      setModels(models);

      compute();

      //      //publish image bounding boxes
      //      publishImageBoundingBoxes();

      sensor_msgs::ImagePtr label_image_msg = cv_bridge::CvImage(std_msgs::Header(),
                                                                 "bgr8",
                                                                 _label_image).toImageMsg();
      _label_image_pub.publish(label_image_msg);

      //      //            std::cerr << ".";
      //      //            _logical_image_sub.unsubscribe();
    }
  }

protected:
  ros::NodeHandle _nh;

  ros::Subscriber _camera_info_sub;
  bool _got_info;

  message_filters::Subscriber<lucrezio_simulation_environments::LogicalImage> _logical_image_sub;
  message_filters::Subscriber<sensor_msgs::Image> _depth_image_sub;
  message_filters::Subscriber<sensor_msgs::Image> _rgb_image_sub;
  typedef message_filters::sync_policies::ApproximateTime<lucrezio_simulation_environments::LogicalImage,
  sensor_msgs::Image,
  sensor_msgs::Image> FilterSyncPolicy;
  message_filters::Synchronizer<FilterSyncPolicy> _synchronizer;

  ros::ServiceClient _model_state_client;

  ros::Time _last_timestamp;
  ros::Publisher _image_bounding_boxes_pub;

  image_transport::ImageTransport _it;
  image_transport::Publisher _label_image_pub;

private:

  Eigen::Isometry3f tfTransform2eigen(const tf::Transform& p){
    Eigen::Isometry3f iso;
    iso.translation().x()=p.getOrigin().x();
    iso.translation().y()=p.getOrigin().y();
    iso.translation().z()=p.getOrigin().z();
    Eigen::Quaternionf q;
    tf::Quaternion tq = p.getRotation();
    q.x()= tq.x();
    q.y()= tq.y();
    q.z()= tq.z();
    q.w()= tq.w();
    iso.linear()=q.toRotationMatrix();
    return iso;
  }

  tf::Transform eigen2tfTransform(const Eigen::Isometry3f& T){
    Eigen::Quaternionf q(T.linear());
    Eigen::Vector3f t=T.translation();
    tf::Transform tft;
    tft.setOrigin(tf::Vector3(t.x(), t.y(), t.z()));
    tft.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
    return tft;
  }


  std::string type2str(int type) {
    std::string r;
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    switch ( depth ) {
      case CV_8U:  r = "8U"; break;
      case CV_8S:  r = "8S"; break;
      case CV_16U: r = "16U"; break;
      case CV_16S: r = "16S"; break;
      case CV_32S: r = "32S"; break;
      case CV_32F: r = "32F"; break;
      case CV_64F: r = "64F"; break;
      default:     r = "User"; break;
    }
    r += "C";
    r += (chans+'0');
    return r;
  }

  void publishImageBoundingBoxes(){
    //        std::cerr << "Publishing detections" << std::endl;
    lucrezio_semantic_perception::ImageBoundingBoxesArray image_bounding_boxes;
    image_bounding_boxes.header.frame_id = "camera_depth_optical_frame";
    image_bounding_boxes.header.stamp = _last_timestamp;
    lucrezio_semantic_perception::ImageBoundingBox image_bounding_box;
    for(int i=0; i < _detections.size(); ++i){
      //            std::cerr << "#" << i+1 << std::endl;
      image_bounding_box.type = _detections[i].type();
      image_bounding_box.top_left.r = _detections[i].topLeft().x();
      image_bounding_box.top_left.c = _detections[i].topLeft().y();
      image_bounding_box.bottom_right.r = _detections[i].bottomRight().x();
      image_bounding_box.bottom_right.c = _detections[i].bottomRight().y();
      lucrezio_semantic_perception::Pixel pixel;
      for(int j=0; j < _detections[i].pixels().size(); ++j){
        pixel.r = _detections[i].pixels()[j].x();
        pixel.c = _detections[i].pixels()[j].y();
        image_bounding_box.pixels.push_back(pixel);
      }
      image_bounding_boxes.image_bounding_boxes.push_back(image_bounding_box);
    }
    _image_bounding_boxes_pub.publish(image_bounding_boxes);
  }
};

int main(int argc, char** argv){
  ros::init(argc, argv, "detection_simulator");
  ros::NodeHandle nh;
  ObjectDetectorNode simulator(nh);

  //ros::spin();

  ros::Rate loop_rate(1);
  while(ros::ok()){
    ros::spinOnce();
    loop_rate.sleep();
  }


  return 0;
}
