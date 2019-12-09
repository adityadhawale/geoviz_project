#include "stereo_mapping/multi_stereo_mapping.h"
#include <cpp_utils/cpp_utils.h>
#include <pcl/common/transforms.h>

template class MultiStereoMapping<double>;
template class MultiStereoMapping<float>;

template <typename T>
void MultiStereoMapping<T>::read_mapping_parameters(
    const std::string &config_file) {
  auto params         = YAML::LoadFile(config_file);
  auto mapping_params = params["mapping_params"];

  mapping_parameters_.patch_size    = mapping_params["patch_size"].as<int>();
  mapping_parameters_.neighbor_dist = mapping_params["neighbor_dist"].as<T>();
  mapping_parameters_.depth_scale   = mapping_params["depth_scale"].as<T>();
  mapping_parameters_.min_points    = mapping_params["min_points"].as<int>();
}

template <typename T>
std::vector<Model<T>> MultiStereoMapping<T>::fit_model_on_image(
    const cv::Mat &image) {
  int row_patches   = std::ceil((T)image.rows / mapping_parameters_.patch_size);
  int col_patches   = std::ceil((T)image.cols / mapping_parameters_.patch_size);
  int total_patches = row_patches * col_patches;
  std::vector<Model<T>> regions(total_patches);

#pragma omp parallel for
  for (int v = 0; v < image.rows; v += mapping_parameters_.patch_size)
    for (int u = 0; u < image.cols; u += mapping_parameters_.patch_size) {
      cv::Mat mask(mapping_parameters_.patch_size,
                   mapping_parameters_.patch_size, CV_8UC1, cv::Scalar(0));
      Model<T> best_model;
      int best_points = 0;

      auto unassigned = cpp_utils::arange(
          0, mapping_parameters_.patch_size * mapping_parameters_.patch_size,
          1);

      while (unassigned.size() > 0 &&
             best_points < mapping_parameters_.min_points) {
        Model<T> model;
        int seed = cpp_utils::select_random_element(unassigned);
        int sv   = seed / mapping_parameters_.patch_size;
        int su   = seed % mapping_parameters_.patch_size;

        unassigned.erase(
            std::remove(unassigned.begin(), unassigned.end(), seed),
            unassigned.end());
        if (v + sv >= image.rows || u + su >= image.cols) continue;
        T depth = (T)image.at<ushort>(v + sv, u + su);
        if (!sensor_model_1_.is_depth_valid(depth)) continue;
        depth /= mapping_parameters_.depth_scale;

        Vec4D<T> point_h =
            sensor_model_1_.get_back_projected_h_point(v + sv, u + su, depth);
        model.mean       = point_h;
        Vec3D<T> point   = h_to_non_h(point_h);
        model.covariance = point * point.transpose();
        model.num_points = 1;
        model.uncertainty =
            sensor_model_1_.get_point_uncertainty(v + sv, u + su, depth);
        mask.at<uchar>(sv, su) = 1;

        auto neigh = this->get_pixel_neighbors(v, u, sv, su, image, mask);

        while (neigh.size() > 0) {
          std::vector<int> secondary;
          for (auto n : neigh) {
            int nv = n / mask.cols;
            int nu = n % mask.cols;
            if ((int)mask.at<uchar>(nv, nu) == 1) continue;
            T nd = (T)image.at<ushort>(v + nv, u + nu);
            nd /= mapping_parameters_.depth_scale;

            mask.at<uchar>(nv, nu) = 1;

            Vec4D<T> neigh_point_h =
                sensor_model_1_.get_back_projected_h_point(v + sv, u + su, nd);
            model.mean += neigh_point_h;
            Vec3D<T> neigh_point = h_to_non_h(neigh_point_h);
            model.covariance += neigh_point * neigh_point.transpose();
            model.uncertainty +=
                sensor_model_1_.get_point_uncertainty(v + sv, u + su, nd);
            model.num_points++;

            unassigned.erase(
                std::remove(unassigned.begin(), unassigned.end(), n),
                unassigned.end());
            auto nn = this->get_pixel_neighbors(v, u, nv, nu, image, mask);
            secondary.insert(secondary.end(), nn.begin(), nn.end());
          }
          neigh = secondary;
        }
        if (model.num_points > best_points) {
          best_points = model.num_points;
          best_model  = model;
        }
      }

      best_model.mean /= best_model.num_points;
      Vec3D<T> mean         = h_to_non_h(best_model.mean);
      best_model.covariance = best_model.covariance / best_model.num_points -
                              mean * mean.transpose();
      best_model.uncertainty /= best_points;

      int id = (v / mapping_parameters_.patch_size) *
                   (image.cols / mapping_parameters_.patch_size) +
               u / mapping_parameters_.patch_size;
      regions[id] = best_model;
    }

  std::cout << "Done Image\n";
  return regions;
}

template <typename T>
std::vector<int> MultiStereoMapping<T>::get_pixel_neighbors(
    const int V, const int U, const int sv, const int su, const cv::Mat &image,
    const cv::Mat &mask) {
  std::vector<int> ret;
  T seed_depth = (T)image.at<ushort>(V + sv, U + su);

  for (int i = -1; i < 2; ++i) {
    for (int j = -1; j < 2; ++j) {
      if (sv + i < 0 || sv + i >= mask.rows || su + j < 0 ||
          su + j >= mask.cols)
        continue;
      if ((int)mask.at<uchar>(sv + i, su + j) == 1) continue;
      T neigh_depth = (T)image.at<ushort>(V + sv + i, U + su + j);
      if (neigh_depth < 10) continue;
      if (!sensor_model_1_.is_depth_valid(neigh_depth)) continue;

      T dist = fabs(seed_depth - neigh_depth);
      dist /= mapping_parameters_.depth_scale;

      if (dist > 0.1) continue;
      int id = (sv + i) * mask.cols + (su + j);
      ret.push_back(id);
    }
  }

  return ret;
}

template <typename T>
void MultiStereoMapping<T>::update_map(const cv::Mat &depth_front_image,
                                       const cv::Mat &depth_back_image,
                                       const Mat4D<T> &T_wb) {
  auto front_regions = this->fit_model_on_image(depth_front_image);
  auto back_regions  = this->fit_model_on_image(depth_back_image);

  Mat4D<T> T_w_c_front = T_wb * stereo_pair_1_.T_body_cam0;
  Mat4D<T> T_w_c_back  = T_wb * stereo_pair_2_.T_body_cam0;

  Eigen::Matrix4f temp = Eigen::Matrix4f::Zero();
  for (int i = 0; i < 16; ++i) temp(i / 4, i % 4) = T_w_c_front(i / 4, i % 4);
  pcl::PointCloud<pcl::PointXYZ> cloud;
  for (int v = 0; v < depth_front_image.rows; v += 2) {
    for (int u = 0; u < depth_front_image.cols; u += 2) {
      T depth = depth_front_image.at<ushort>(v, u);
      if (depth < 1) continue;
      depth /= mapping_parameters_.depth_scale;
      pcl::PointXYZ p;
      p.x = (u - stereo_pair_1_.K0(0, 2)) / stereo_pair_1_.K0(0, 0) * depth;
      p.y = (v - stereo_pair_1_.K0(1, 2)) / stereo_pair_1_.K0(1, 1) * depth;
      p.z = depth;
      cloud.push_back(p);
    }
  }
  pcl::transformPointCloud(cloud, cloud, temp);

  for (int i = 0; i < 16; ++i) temp(i / 4, i % 4) = T_w_c_back(i / 4, i % 4);
  pcl::PointCloud<pcl::PointXYZ> back_cloud;
  for (int v = 0; v < depth_back_image.rows; v += 2) {
    for (int u = 0; u < depth_back_image.cols; u += 2) {
      T depth = depth_back_image.at<ushort>(v, u);
      if (depth < 1) continue;
      depth /= mapping_parameters_.depth_scale;
      pcl::PointXYZ p;
      p.x = (u - stereo_pair_2_.K0(0, 2)) / stereo_pair_2_.K0(0, 0) * depth;
      p.y = (v - stereo_pair_2_.K0(1, 2)) / stereo_pair_2_.K0(1, 1) * depth;
      p.z = depth;
      back_cloud.push_back(p);
    }
  }
  pcl::transformPointCloud(back_cloud, back_cloud, temp);
  cloud += back_cloud;
  this->publish_point_cloud("/geoviz_cloud", cloud);

  std::vector<Model<T>> regions;
  for (size_t r = 0; r < front_regions.size(); ++r) {
    front_regions[r].transform_model(T_w_c_front);
    // back_regions[r].transform_model(T_w_c_back);
  }
  regions = front_regions;
  // regions.insert(regions.end(), back_regions.begin(), back_regions.end());

  this->publish_model_as_gmm("/geoviz_model", regions);
}

template <typename T>
void MultiStereoMapping<T>::publish_model_as_gmm(
    const std::string &name, const std::vector<Model<T>> &model) {
  if (publishers_.find(name) == publishers_.end()) {
    ros::Publisher pub = nh_.advertise<gmm_msgs::GaussianMixture>(name, 10);
    publishers_[name]  = pub;
  }
  if (publishers_[name].getNumSubscribers() == 0) return;

  int valid_regs = 0;
  for (size_t r = 0; r < model.size(); ++r) {
    if (model[r].num_points > mapping_parameters_.min_points) ++valid_regs;
  }
  gmm_msgs::GaussianMixture gmm;
  gmm.components.resize(valid_regs);
  gmm.weights.resize(valid_regs);

  gmm.size = valid_regs;

  int count = 0;
  for (size_t r = 0; r < model.size(); ++r) {
    if (model[r].num_points <= mapping_parameters_.min_points) continue;
    gmm.components[count].num_points = model[r].num_points;
    gmm.components[count].mean.x     = model[r].mean(0);
    gmm.components[count].mean.y     = model[r].mean(1);
    gmm.components[count].mean.z     = model[r].mean(2);

    Mat3D<T> temp_cov = Mat3D<T>::Zero();
    for (int i = 0; i < 9; ++i)
      gmm.components[count].covariance[i] = model[r].covariance(i / 3, i % 3) +
                                            model[r].uncertainty(i / 3, i % 3);
    // gmm.components[count].covariance[0] += 1e-2;
    // gmm.components[count].covariance[4] += 1e-2;
    // gmm.components[count].covariance[8] += 1e-2;

    gmm.weights[count]             = model[r].num_points;
    gmm.components[count].color[0] = 0.1;
    gmm.components[count].color[1] = 0.1;
    gmm.components[count].color[2] = 0.1;
    ++count;
  }
  publishers_[name].publish(gmm);
}

template <typename T>
void MultiStereoMapping<T>::publish_point_cloud(
    const std::string &name, const pcl::PointCloud<pcl::PointXYZ> &cloud) {
  if (publishers_.find(name) == publishers_.end()) {
    ros::Publisher pub =
        nh_.advertise<pcl::PointCloud<pcl::PointXYZ>>(name, 10);
    publishers_[name] = pub;
  }
  if (publishers_[name].getNumSubscribers() == 0) return;
  publishers_[name].publish(cloud);
}
