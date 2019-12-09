#include <nav_msgs/Odometry.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include "stereo_mapping/multi_stereo_mapping.h"

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <cv_bridge/cv_bridge.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>

#include <boost/any.hpp>
#include <boost/foreach.hpp>
#include <boost/thread/thread.hpp>

template <class M>
class SubscriberInterface {
  virtual void new_message(const boost::shared_ptr<M const> &msg) = 0;
};

// for bagfile subscribers
#define foreach BOOST_FOREACH
template <class M>
class BagSubscriber : public message_filters::SimpleFilter<M> {
 public:
  void new_message(const boost::shared_ptr<M const> &msg) {
    this->signalMessage(msg);
  }
};

template <typename T>
class MultiStereoMappingNode {
 private:
  YAML::Node parameters_;
  YAML::Node calibration_data_;
  StereoCalibration<T> calibration_struct_pair_0_;
  StereoCalibration<T> calibration_struct_pair_1_;

  MultiStereoMapping<T> multi_stero_mapping_obj_;

 public:
  MultiStereoMappingNode(const std::string &config_file,
                         const std::string &calibration_file) {
    parameters_       = YAML::LoadFile(config_file);
    calibration_data_ = YAML::LoadFile(calibration_file);

    multi_stero_mapping_obj_.read_mapping_parameters(config_file);
    this->read_calibration();
    this->read_bag();
  }

  void read_calibration();
  void callback(const sensor_msgs::ImageConstPtr &depth_front_msg,
                const sensor_msgs::ImageConstPtr &depth_back_msg,
                const nav_msgs::OdometryConstPtr &body_odom_msg) {
    cv_bridge::CvImagePtr depth_front_ptr;
    try {
      depth_front_ptr = cv_bridge::toCvCopy(
          depth_front_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    } catch (cv_bridge::Exception &e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    cv_bridge::CvImagePtr depth_back_ptr;
    try {
      depth_back_ptr = cv_bridge::toCvCopy(
          depth_back_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    } catch (cv_bridge::Exception &e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    cv::Mat depth_front_image, depth_back_image;
    depth_front_image = depth_front_ptr->image;
    depth_back_image  = depth_back_ptr->image;

    cv::imshow("front", depth_front_image);
    cv::imshow("back", depth_back_image);
    cv::waitKey(1);

    Mat4D<T> T_wb = Mat4D<T>::Identity();
    Eigen::Quaternion<T> q =
        Eigen::Quaternion<T>(body_odom_msg->pose.pose.orientation.w,
                             body_odom_msg->pose.pose.orientation.x,
                             body_odom_msg->pose.pose.orientation.y,
                             body_odom_msg->pose.pose.orientation.z);
    Mat3D<T> R_wb          = q.toRotationMatrix();
    T_wb.block(0, 0, 3, 3) = R_wb;

    T_wb(0, 3) = body_odom_msg->pose.pose.position.x;
    T_wb(1, 3) = body_odom_msg->pose.pose.position.y;
    T_wb(2, 3) = body_odom_msg->pose.pose.position.z;

    if (ros::ok())
      multi_stero_mapping_obj_.update_map(depth_front_image, depth_back_image,
                                          T_wb);
  };

  void read_bag() {
    auto bagname = parameters_["input_params"]["bagname"].as<std::string>();
    rosbag::Bag bag;
    bag.open(bagname, rosbag::bagmode::Read);

    BagSubscriber<sensor_msgs::Image> depth_front_sub;
    BagSubscriber<sensor_msgs::Image> depth_back_sub;
    BagSubscriber<nav_msgs::Odometry> body_odom_sub;

    std::map<std::string, std::string> topic_map;
    std::vector<std::string> topics;

    auto sync_topics = parameters_["input_params"]["sync_messages"];
    for (auto it = sync_topics.begin(); it != sync_topics.end(); ++it) {
      auto msg_data = *it;
      topic_map[msg_data["topic"].as<std::string>()] =
          msg_data["type"].as<std::string>();

      topics.push_back(msg_data["topic"].as<std::string>());
    }

    rosbag::View view(bag, rosbag::TopicQuery(topics));

    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Image, sensor_msgs::Image, nav_msgs::Odometry>
        sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), depth_front_sub,
                                                 depth_back_sub, body_odom_sub);

    sync.registerCallback(
        boost::bind(&MultiStereoMappingNode::callback, this, _1, _2, _3));

    for (auto m : view) {
      std::string msg_topic = m.getTopic();
      if (topic_map[msg_topic] == "DepthImage1") {
        auto msg = m.instantiate<sensor_msgs::Image>();
        depth_front_sub.new_message(msg);
      } else if (topic_map[msg_topic] == "Odometry") {
        auto msg = m.instantiate<nav_msgs::Odometry>();
        body_odom_sub.new_message(msg);
      } else if (topic_map[msg_topic] == "DepthImage2") {
        auto msg = m.instantiate<sensor_msgs::Image>();
        depth_back_sub.new_message(msg);
      }
    }
  }
};

template <typename T>
void MultiStereoMappingNode<T>::read_calibration() {
  int w = calibration_data_["image_width"].as<int>();
  int h = calibration_data_["image_height"].as<int>();

  calibration_struct_pair_0_.w = w;
  calibration_struct_pair_1_.w = w;

  calibration_struct_pair_0_.h = h;
  calibration_struct_pair_1_.h = h;

  std::vector<T> T_cam0_imu =
      calibration_data_["T_cam0_imu"]["data"].as<std::vector<T>>();
  std::vector<T> T_cam1_imu =
      calibration_data_["T_cam1_imu"]["data"].as<std::vector<T>>();

  std::vector<T> T_cam2_imu =
      calibration_data_["T_cam2_imu"]["data"].as<std::vector<T>>();
  std::vector<T> T_cam3_imu =
      calibration_data_["T_cam3_imu"]["data"].as<std::vector<T>>();

  std::vector<T> cam0_intrinsics =
      calibration_data_["cam0_intrinsics"]["data"].as<std::vector<T>>();
  std::vector<T> cam1_intrinsics =
      calibration_data_["cam1_intrinsics"]["data"].as<std::vector<T>>();
  std::vector<T> cam2_intrinsics =
      calibration_data_["cam2_intrinsics"]["data"].as<std::vector<T>>();
  std::vector<T> cam3_intrinsics =
      calibration_data_["cam3_intrinsics"]["data"].as<std::vector<T>>();

  Mat4D<T> T_cam0_body = Mat4D<T>::Zero();
  Mat4D<T> T_cam1_body = Mat4D<T>::Zero();
  Mat4D<T> T_cam2_body = Mat4D<T>::Zero();
  Mat4D<T> T_cam3_body = Mat4D<T>::Zero();

  for (int i = 0; i < 16; ++i) {
    T_cam0_body(i / 4, i % 4) = T_cam0_imu[i];
    T_cam1_body(i / 4, i % 4) = T_cam1_imu[i];
    T_cam2_body(i / 4, i % 4) = T_cam2_imu[i];
    T_cam3_body(i / 4, i % 4) = T_cam3_imu[i];
  }

  calibration_struct_pair_0_.T_body_cam0 = T_cam0_body.inverse();
  calibration_struct_pair_0_.T_body_cam1 = T_cam1_body.inverse();
  calibration_struct_pair_1_.T_body_cam0 = T_cam2_body.inverse();
  calibration_struct_pair_1_.T_body_cam1 = T_cam3_body.inverse();

  calibration_struct_pair_0_.K0(0, 0) = cam0_intrinsics[0];
  calibration_struct_pair_0_.K0(1, 1) = cam0_intrinsics[1];
  calibration_struct_pair_0_.K0(0, 2) = cam0_intrinsics[2];
  calibration_struct_pair_0_.K0(1, 2) = cam0_intrinsics[3];
  calibration_struct_pair_0_.K0(2, 2) = 1;

  calibration_struct_pair_0_.K1(0, 0) = cam1_intrinsics[0];
  calibration_struct_pair_0_.K1(1, 1) = cam1_intrinsics[1];
  calibration_struct_pair_0_.K1(0, 2) = cam1_intrinsics[2];
  calibration_struct_pair_0_.K1(1, 2) = cam1_intrinsics[3];
  calibration_struct_pair_0_.K1(2, 2) = 1;

  calibration_struct_pair_1_.K0(0, 0) = cam2_intrinsics[0];
  calibration_struct_pair_1_.K0(1, 1) = cam2_intrinsics[1];
  calibration_struct_pair_1_.K0(0, 2) = cam2_intrinsics[2];
  calibration_struct_pair_1_.K0(1, 2) = cam2_intrinsics[3];
  calibration_struct_pair_1_.K0(2, 2) = 1;

  calibration_struct_pair_1_.K1(0, 0) = cam3_intrinsics[0];
  calibration_struct_pair_1_.K1(1, 1) = cam3_intrinsics[1];
  calibration_struct_pair_1_.K1(0, 2) = cam3_intrinsics[2];
  calibration_struct_pair_1_.K1(1, 2) = cam3_intrinsics[3];
  calibration_struct_pair_1_.K1(2, 2) = 1;

  multi_stero_mapping_obj_.set_calibration(calibration_struct_pair_0_,
                                           calibration_struct_pair_1_);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "geoviz_mapping_node");
  ros::NodeHandle nh("~");

  std::string config_file, calibration_file;
  if (!nh.hasParam("config_file")) {
    std::cerr << "No config file provided!\n";
    return EXIT_FAILURE;
  }
  if (!nh.hasParam("calibration_file")) {
    std::cerr << "No calibration file provided!\n";
    return EXIT_FAILURE;
  }
  nh.getParam("config_file", config_file);
  nh.getParam("calibration_file", calibration_file);

  auto mapping_node =
      MultiStereoMappingNode<float>(config_file, calibration_file);
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}
