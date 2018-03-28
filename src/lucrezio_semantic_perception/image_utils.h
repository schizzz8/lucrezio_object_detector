#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

typedef cv::Mat_<unsigned char> UnsignedCharImage;
typedef cv::Mat_<float> FloatImage;
typedef cv::Mat_<cv::Vec3f> Float3Image;
typedef cv::Mat_<unsigned short> RawDepthImage;
typedef cv::Mat_<cv::Vec3b> RGBImage;


void convert_16UC1_to_32FC1(cv::Mat& dest, const cv::Mat& src, float scale = 0.001f);

void initializePinholeDirections(Float3Image& directions,
                                 const Eigen::Matrix3f& camera_matrix,
                                 const UnsignedCharImage& mask=UnsignedCharImage());

void computePointsImage(Float3Image& point_image,
                          const Float3Image& direction_image,
                          const FloatImage&  depth_image,
                          const float min_distance,
                          const float max_distance);
