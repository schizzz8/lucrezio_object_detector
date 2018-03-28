#include "image_utils.h"

void convert_16UC1_to_32FC1(cv::Mat &dest, const cv::Mat &src, float scale){
  assert(src.type() == CV_16UC1 && "convert_16UC1_to_32FC1: source image of different type from 16UC1");
  const unsigned short* sptr = (const unsigned short*)src.data;
  int size = src.rows * src.cols;
  const unsigned short* send = sptr + size;
  dest.create(src.rows, src.cols, CV_32FC1);
  dest.setTo(cv::Scalar(0.0f));
  float* dptr = (float*)dest.data;
  while(sptr < send) {
    if(*sptr == 0) { *dptr = 1e9f; }
    else { *dptr = scale * (*sptr); }
    ++dptr;
    ++sptr;
  }
}

void initializePinholeDirections(Float3Image &directions,
                                 const Eigen::Matrix3f &camera_matrix,
                                 const UnsignedCharImage &mask){
  int rows=directions.rows;
  int cols=directions.cols;
  const Eigen::Matrix3f inverse_camera_matrix=camera_matrix.inverse();
  for (int r=0; r<rows; ++r) {
    cv::Vec3f* direction=directions.ptr<cv::Vec3f>(r);
    const unsigned char* masked=0;
    if (! mask.empty()) {
      masked=mask.ptr<const unsigned char>(r);
    }
    for (int c=0; c<cols; ++c, ++direction){
      *direction=cv::Vec3f(0.f,0.f,0.f);
      bool keep_point=(!masked || *masked);
      if (keep_point) {
        Eigen::Vector3f dir=inverse_camera_matrix*Eigen::Vector3f(c,r,1);
        *direction=cv::Vec3f(dir.x(), dir.y(), dir.z());
      }
      if (masked) ++masked;
    }
  }
}

void computePointsImage(Float3Image& points_image,
                        const Float3Image& directions,
                        const FloatImage&  depth_image,
                        const float min_distance,
                        const float max_distance){
  if (directions.size()!=depth_image.size())
    throw std::runtime_error("directions and depth image sizes should match");
  int rows=directions.rows;
  int cols=directions.cols;
  points_image.create(rows, cols);
  for (int r=0; r<rows; ++r) {
    cv::Vec3f* point=points_image.ptr<cv::Vec3f>(r);
    const cv::Vec3f* direction=directions.ptr<const cv::Vec3f>(r);
    const float* depth=depth_image.ptr<const float>(r);
    for (int c=0; c<cols; ++c, ++direction, ++depth, ++point){
      float d=*depth;
      if (d>max_distance||d<min_distance)
        d=0;
      *point=(*direction)*d;
    }
  }
}
