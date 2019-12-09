#ifndef __STRUCTURES_H__
#define __STRUCTURES_H__

#include <Eigen/Core>

template <typename T>
using Mat4D = Eigen::Matrix<T, 4, 4, Eigen::RowMajor>;

template <typename T>
using MatxD = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename T>
using Vec4D = Eigen::Matrix<T, 4, 1>;

template <typename T>
using Mat3D = Eigen::Matrix<T, 3, 3, Eigen::RowMajor>;

template <typename T>
using Vec3D = Eigen::Matrix<T, 3, 1>;

template <typename T>
struct StereoCalibration {
  int w = 0;
  int h = 0;

  Mat3D<T> K0 = Mat3D<T>::Zero();
  Mat3D<T> K1 = Mat3D<T>::Zero();

  Mat4D<T> T_body_cam0 = Mat4D<T>::Identity();
  Mat4D<T> T_body_cam1 = Mat4D<T>::Identity();
};

template <typename T>
struct MappingParameters {
  int patch_size  = 0;
  T neighbor_dist = 0;
  T depth_scale   = 1;
  int min_points  = 0;
};

template <typename T>
class Model {
 public:
  Vec4D<T> mean;
  Mat3D<T> covariance;
  Mat3D<T> uncertainty;
  int num_points;
  T weight;
  Model() {
    mean        = Vec4D<T>::Zero();
    covariance  = Mat3D<T>::Zero();
    uncertainty = Mat3D<T>::Zero();
    num_points  = 0;
    weight      = 0;
  }

  void transform_model(const Mat4D<T> &Trans) {
    mean       = Trans * mean;
    Mat3D<T> R = Trans.block(0, 0, 3, 3);

    covariance  = R * covariance * R.transpose();
    uncertainty = R * uncertainty * R.transpose();
  }
};

template <typename T>
class SensorModel {
 public:
  StereoCalibration<T> calibration_;
  SensorModel(const StereoCalibration<T> &calib) : calibration_(calib) {}
  SensorModel() = default;
  void set_calibration(const StereoCalibration<T> &calib) {
    calibration_ = calib;
  }
  inline bool is_depth_valid(const T depth) {
    if (depth <= 1 || depth > 65000) return false;
    return true;
  }

  Mat3D<T> get_point_uncertainty(const int v, const int u, const T depth) {
    Mat3D<T> ret = Mat3D<T>::Identity() * 1e-4;
    return ret;
  }

  Vec4D<T> get_back_projected_h_point(const int v, const int u, const T depth) {
    Vec4D<T> ret = Vec4D<T>::Zero();
    ret(0)       = (u - calibration_.K0(0, 2)) / calibration_.K0(0, 0) * depth;
    ret(1)       = (v - calibration_.K0(1, 2)) / calibration_.K0(1, 1) * depth;
    ret(2)       = depth;
    ret(3)       = 1;
    return ret;
  }
};

template <typename T>
Vec3D<T> h_to_non_h(const Vec4D<T> &vec) {
  Vec3D<T> ret = Vec3D<T>::Zero();
  for (int i = 0; i < 3; ++i) ret(i) = vec(i);

  return ret;
}

#endif  //__STRUCTURES_H__
