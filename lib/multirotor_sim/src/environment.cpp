#include "environment.h"
#include "geometry/support.h"
#include "nanoflann_eigen/nanoflann_eigen.h"


Environment::Environment(int seed)
  : uniform_(-1.0, 1.0),
    generator_(seed)
{
    // Hard definition for points_ ****
    //    Vector3d new_point = t_I_c + depth * zeta_I;
    //    points_.pts.push_back(new_point);
//    Vector3d point0;
//    point0 << 0.0, 0.0, 0.0;
//    points_.pts.push_back(point0);

//    Vector3d point1;
//    point1 << -0.2, 0.2, 0.0;
//    points_.pts.push_back(point1);

//    Vector3d point2;
//    point2 << 0.2, 0.2, 0.0;
//    points_.pts.push_back(point2);

//    Vector3d point3;
//    point3 << 0.2, -0.2, 0.0;
//    points_.pts.push_back(point3);

//    Vector3d point4;
//    point4 << -0.2, -0.2, 0.0;
//    points_.pts.push_back(point4);
}

void Environment::load(string filename)
{
  // Load Landmarks
    std::vector<double> loaded_landmarks;
    int num_landmarks;
    if (get_yaml_node("landmarks", filename, loaded_landmarks))
    {
        num_landmarks = std::floor(loaded_landmarks.size()/3.0);
        for (int i = 0; i < num_landmarks; i++)
        {
            Vector3d point;
            point << loaded_landmarks[i * 3], loaded_landmarks[i * 3 + 1],
                     loaded_landmarks[i * 3 + 2];
            points_.pts.push_back(point);
//            std::cout << "adding point [" << point << "] to the environment." << std::endl;
        }
    }

  get_yaml_node("wall_max_offset", filename, max_offset_);
  get_yaml_node("points_move_stdev", filename, move_stdev_);
  get_yaml_eigen("image_size", filename, img_size_);
  Vector2d focal_len;
  get_yaml_eigen("focal_len", filename, focal_len);
  get_yaml_eigen("cam_center", filename, img_center_);
  finv_ = focal_len.cwiseInverse();


  point_idx_ = 0;
  floor_level_ = 0;
  kd_tree_ = new KDTree3d(3, points_, 10);
}

bool Environment::get_center_img_center_on_ground_plane(const Vector3d& t_I_c, const Quatd& q_I_c, Vector3d& point)
{
//  Vector3d zeta_I = q_I_c.rota(e_z);
//  double depth = -t_I_c(2) / zeta_I(2);
//  if (depth < 0.5 || depth > 100.0 )
//  {
//    return false;
//  }
//  else
//  {
//    point =  zeta_I * depth + t_I_c;
//    return true;
//  }
  return true;
}

int Environment::add_point(const Vector3d& t_I_c, const Quatd& q_I_c, Vector3d& zeta, Vector2d& pix, double& depth)
{
//  // Choose a random pixel (assume that image center is at center of camera)
//  pix.setRandom();
//  pix = 0.45 * pix.cwiseProduct(img_size_); // stay away from the edges of image

//  // Calculate the Unit Vector
//  zeta << pix.cwiseProduct(finv_), 1.0;
//  zeta /= zeta.norm();

//  pix += img_center_;

//  // Rotate the Unit vector into inertial coordatines
//  Vector3d zeta_I = q_I_c.rota(zeta);

//  // Find where zeta crosses the floor
//  double h = -1.0 * (t_I_c(2) + uniform_(generator_) * max_offset_);
//  depth = h / (zeta_I(2));
//  if (depth < 0.5 || depth > 100.0 )
//  {
//    return -1;
//  }
//  else
//  {
//    int idx = point_idx_++;
//    Vector3d new_point = t_I_c + depth * zeta_I;
//    points_.pts.push_back(new_point);
//    kd_tree_->addPoints(idx, idx);
//    return idx;
//  }
  return -1;
}

bool Environment::get_closest_points(const Vector3d& query_pt,
      int num_pts, double max_dist, vector<Vector3d, aligned_allocator<Vector3d>>& pts, vector<size_t>& ids)
{
//  std::vector<size_t> ret_index(num_pts);
//  std::vector<double> dist_sqr(num_pts);
//  nanoflann::KNNResultSet<double> resultSet(num_pts);
//  resultSet.init(&ret_index[0], &dist_sqr[0]);
//  kd_tree_->findNeighbors(resultSet, query_pt.data(), nanoflann::SearchParams(10));

//  pts.clear();
//  ids.clear();
//  for (int i = 0; i < resultSet.size(); i++)
//  {
//    if (dist_sqr[i] < max_dist*max_dist)
//    {
//      pts.push_back(points_.pts[ret_index[i]]);
//      ids.push_back(ret_index[i]);
//    }
//  }
//  return pts.size() > 0;
  return false;
}

//void Environment::move_point(int id)
//{
//  Vector3d move;
//  random_normal_vec(move, move_stdev_, normal_, generator_);
////  cout << "moving point by " << move.transpose();
//  points_.row(id) += move.transpose();
//}
