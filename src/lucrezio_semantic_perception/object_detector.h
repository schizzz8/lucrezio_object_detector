#pragma once

#include "detection.h"
#include "model.h"

#include <iostream>
#include <fstream>
#include <iomanip>

#include "image_utils.h"

namespace lucrezio_semantic_perception{

  class ObjectDetector{
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::pair<Eigen::Vector3f,Eigen::Vector3f> BoundingBox3D;
    typedef std::vector<BoundingBox3D> BoundingBox3DVector;

    ObjectDetector(){}

    inline void setK(const Eigen::Matrix3f& K_){_K = K_;}

    void setImages(const RGBImage &rgb_image_,
                   const RawDepthImage &raw_depth_image_);

    inline void setCameraTransforms(const Eigen::Isometry3f &rgbd_camera_transform_,
                        const Eigen::Isometry3f &logical_camera_transform_){
      _rgbd_camera_transform = rgbd_camera_transform_;
      _logical_camera_transform = logical_camera_transform_;
    }

    inline void setModels(const ModelVector &models_){_models = models_;}

    void readData(char* filename);

    void compute();

    inline const Eigen::Matrix3f &K() const {return _K;}
    inline const Eigen::Isometry3f &rgbdCameraTransform() const {return _rgbd_camera_transform;}
    inline const Eigen::Isometry3f &logicalCameraTransform() const {return _logical_camera_transform;}
    inline const ModelVector &models() const {return _models;}
    inline const BoundingBox3DVector &boundingBoxes() const {return _bounding_boxes;}
    inline const DetectionVector &detections() const {return _detections;}
    inline const RGBImage &labelImage() const {return _label_image;}

  protected:
    RGBImage _rgb_image;
    RawDepthImage _raw_depth_image;
    int _rows;
    int _cols;
    Eigen::Matrix3f _K;
    Float3Image _points_image;

    Eigen::Isometry3f _rgbd_camera_transform;
    Eigen::Isometry3f _logical_camera_transform;
    ModelVector _models;

    BoundingBox3DVector _bounding_boxes;
    DetectionVector _detections;

    RGBImage _label_image;

  private:
    void computeWorldBoundingBoxes();

    inline bool inRange(const Eigen::Vector3f &point, const BoundingBox3D &bounding_box){
      return (point.x() >= bounding_box.first.x()-0.01 && point.x() <= bounding_box.second.x()+0.01 &&
              point.y() >= bounding_box.first.y()-0.01 && point.y() <= bounding_box.second.y()+0.01 &&
              point.z() >= bounding_box.first.z()-0.01 && point.z() <= bounding_box.second.z()+0.01);
    }

    void computeImageBoundingBoxes();

    cv::Vec3b type2color(std::string type);

    void computeLabelImage();

  };

}

