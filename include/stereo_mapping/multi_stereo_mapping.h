#ifndef __MULTI_STEREO_MAPPPING_H__
#define __MULTI_STEREO_MAPPPING_H__

#include <cmath>
#include <iostream>

#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <yaml-cpp/yaml.h>

#include <gmm_msgs/Gaussian.h>
#include <gmm_msgs/GaussianMixture.h>
#include <std_msgs/Header.h>

#include "stereo_mapping/structures.h"

template <typename T>
class MultiStereoMapping {
 private:
  int patch_size_ = 4;
  StereoCalibration<T> stereo_pair_1_;
  StereoCalibration<T> stereo_pair_2_;
  MappingParameters<T> mapping_parameters_;
  SensorModel<T> sensor_model_1_;
  SensorModel<T> sensor_model_2_;
  bool calibration_set_ = false;
  std::map<std::string, ros::Publisher> publishers_;
  ros::NodeHandle nh_;

  std::vector<int> get_pixel_neighbors(const int V, const int U, const int v,
                                       const int u, const cv::Mat &image,
                                       const cv::Mat &mask);

 public:
  MultiStereoMapping() {
    ros::NodeHandle nh("~");
    nh_ = nh;
  }
  MultiStereoMapping(const StereoCalibration<T> &pair_1,
                     const StereoCalibration<T> &pair_2)
      : stereo_pair_1_(pair_1), stereo_pair_2_(pair_2) {
    calibration_set_ = true;
    sensor_model_1_.set_calibration(pair_1);
    sensor_model_2_.set_calibration(pair_2);
    ros::NodeHandle nh("~");
    nh_ = nh;
  };

  void read_mapping_parameters(const std::string &config_file);
  void update_map(const cv::Mat &depth_front_image,
                  const cv::Mat &depth_back_image, const Mat4D<T> &T_wb);
  std::vector<Model<T>> fit_model_on_image(const cv::Mat &image);
  void publish_model_as_gmm(const std::string &name,
                            const std::vector<Model<T>> &model);
  void publish_point_cloud(const std::string &name,
                           const pcl::PointCloud<pcl::PointXYZ> &cloud);

  inline void set_calibration(const StereoCalibration<T> &pair_1,
                              const StereoCalibration<T> &pair_2) {
    stereo_pair_1_   = pair_1;
    stereo_pair_2_   = pair_2;
    calibration_set_ = true;
    sensor_model_1_.set_calibration(pair_1);
    sensor_model_2_.set_calibration(pair_2);
  }
};

#endif  //__MULTI_STEREO_MAPPPING_H__
