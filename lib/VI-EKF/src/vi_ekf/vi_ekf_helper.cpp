#include "vi_ekf.h"

#ifdef MC_SIM
#include "multirotor_sim/utils.h"
#endif

namespace vi_ekf
{

#ifdef MC_SIM
void VIEKF::load(std::string ekf_file, std::string common_file, bool use_logger, string prefix)
{
  Matrix<double, xMU, 1> x;
  double mu0;
  get_yaml_priority_eigen("ekf_x0", ekf_file, common_file, x);
  get_yaml_priority("mu0", ekf_file, common_file, mu0);
  Matrix<double, xZ,1> x0;
  x0.block<xMU,1>(0,0) = x;
  x0(xMU, 0) = mu0;
  
  Matrix<double, dxZ,1> P0, Qx, lambda;
  get_yaml_priority_eigen("P0", ekf_file, common_file, P0);
  get_yaml_priority_eigen("Qx", ekf_file, common_file, Qx);
  get_yaml_priority_eigen("lambda", ekf_file, common_file, lambda);

  double accel_init_stdev, gyro_init_stdev;
  get_yaml_priority("accel_init_stdev", ekf_file, common_file, accel_init_stdev);
  get_yaml_priority("gyro_init_stdev", ekf_file, common_file, gyro_init_stdev);
  double acc_init_var = accel_init_stdev * accel_init_stdev;
  double gyro_init_var = gyro_init_stdev * gyro_init_stdev;
  P0.block<3,1>(dxB_A,0).array() = acc_init_var;
  P0.block<3,1>(dxB_G,0).array() = gyro_init_var;

  double dt;
  get_yaml_priority("imu_update_rate", ekf_file, common_file, dt);
  dt = 1.0/dt;

  double accel_bias_walk_stdev, gyro_bias_walk_stdev;
  get_yaml_priority("accel_bias_walk", ekf_file, common_file, accel_bias_walk_stdev);
  get_yaml_priority("gyro_bias_walk", ekf_file, common_file, gyro_bias_walk_stdev);
  double acc_walk_var = accel_bias_walk_stdev * accel_bias_walk_stdev * dt * dt; // dt^2 for discrete propagation
  double gyro_walk_var = gyro_bias_walk_stdev * gyro_bias_walk_stdev * dt * dt; // dt^2 for discrete propagation
  Qx.block<3,1>(dxB_A,0).array() = acc_walk_var;
  Qx.block<3,1>(dxB_G,0).array() = gyro_walk_var;
  
  Vector3d P0_feat, Qx_feat, lambda_feat, p_b_c;
  Vector2d cam_center, focal_len;
  Vector4d q_b_c;
  get_yaml_priority_eigen("P0_feat", ekf_file, common_file, P0_feat);
  get_yaml_priority_eigen("Qx_feat", ekf_file, common_file, Qx_feat);
  get_yaml_priority_eigen("lambda_feat", ekf_file, common_file, lambda_feat);
  get_yaml_priority_eigen("p_b_c", ekf_file, common_file, p_b_c);
  get_yaml_priority_eigen("cam_center", ekf_file, common_file, cam_center);
  get_yaml_priority_eigen("focal_len", ekf_file, common_file, focal_len);
  get_yaml_priority_eigen("q_b_c", ekf_file, common_file, q_b_c);

  uVector Qu;
  double acc_stdev, gyro_stdev;
  get_yaml_priority("accel_noise_stdev", ekf_file, common_file, acc_stdev);
  get_yaml_priority("gyro_noise_stdev", ekf_file, common_file, gyro_stdev);
  double acc_var = acc_stdev * acc_stdev * dt * dt; // dt^2 for discrete propagation
  double gyro_var = gyro_stdev * gyro_stdev * dt * dt; // dt^2 for discrete propagation
  Qu << acc_var, acc_var, acc_var, gyro_var, gyro_var, gyro_var;
  
  std::string log_directory;
  double min_depth, keyframe_overlap;
  bool use_drag_term, partial_update, keyframe_reset;
  int cov_prop_skips;
  get_yaml_priority("min_depth", ekf_file, common_file, min_depth);
  get_yaml_priority("log_directory", ekf_file, common_file, log_directory);
  get_yaml_priority("use_drag_term", ekf_file, ekf_file, use_drag_term);
  get_yaml_priority("partial_update", ekf_file, ekf_file, partial_update);
  get_yaml_priority("keyframe_reset", ekf_file, ekf_file, keyframe_reset);
  get_yaml_priority("keyframe_overlap", ekf_file, common_file, keyframe_overlap);
  get_yaml_priority("cov_prop_skips", ekf_file, common_file, cov_prop_skips);
  if (!use_logger)
  {
    log_directory = "~"; // special character to disable the logger
  }
  
  init(x0, P0, Qx, lambda, Qu, P0_feat, Qx_feat, lambda_feat, cam_center,
       focal_len, q_b_c, p_b_c, min_depth, log_directory, use_drag_term, 
       partial_update, keyframe_reset, keyframe_overlap, cov_prop_skips, prefix);
}
#endif


void VIEKF::boxplus(const xVector& x, const dxVector& dx, xVector& out) const
{
  out.block<6,1>((int)xPOS, 0) = x.block<6,1>((int)xPOS, 0) + dx.block<6,1>((int)dxPOS, 0);
  out.block<4,1>((int)xATT, 0) = (Quatd(x.block<4,1>((int)xATT, 0)) + dx.block<3,1>((int)dxATT, 0)).elements();
  out.block<7,1>((int)xB_A, 0) = x.block<7,1>((int)xB_A, 0) + dx.block<7,1>((int)dxB_A, 0);
  for (int i = 0; i < len_features_; i++)
  {
    out.block<4,1>(xZ+i*5,0) = q_feat_boxplus(Quatd(x.block<4,1>(xZ+i*5,0)), dx.block<2,1>(dxZ+3*i,0)).elements();
    out(xZ+i*5+4) = x(xZ+i*5+4) + dx(dxZ+3*i+2);
  }
}

void VIEKF::boxminus(const xVector &x1, const xVector &x2, dxVector &out) const
{
  out.block<6,1>((int)dxPOS, 0) = x1.block<6,1>((int)xPOS, 0) - x2.block<6,1>((int)xPOS, 0);
  out.block<3,1>((int)dxATT, 0) = (Quatd(x1.block<4,1>((int)xATT, 0)) - Quatd(x2.block<4,1>((int)xATT, 0)));
  out.block<7,1>((int)dxB_A, 0) = x1.block<7,1>((int)xB_A, 0) - x2.block<7,1>((int)xB_A, 0);
  
  for (int i = 0; i < len_features_; i++)
  {
    out.block<2,1>(dxZ+i*3,0) = q_feat_boxminus(Quatd(x1.block<4,1>(xZ+i*5,0)), Quatd(x2.block<4,1>(xZ+i*5,0)));
    out(dxZ+i*3+2) = x1(xZ+i*5+4) - x2(xZ+i*5+4);
  }
}


int VIEKF::global_to_local_feature_id(const int global_id) const
{
  int dist = std::distance(current_feature_ids_.begin(), std::find(current_feature_ids_.begin(), current_feature_ids_.end(), global_id));
  if (dist < current_feature_ids_.size())
  {
    return dist;
  }
  else
  {
    return -1;
  }
}


void VIEKF::fix_depth()
{
  // Apply an Inequality Constraint per
  // "Avoiding Negative Depth in Inverse Depth Bearing-Only SLAM"
  // by Parsley and Julier
  for (int i = 0; i < len_features_; i++)
  {
    int xRHO_i = xZ + 5*i + 4;
    int dxRHO_i = dxZ + 3*i + 2;
    if (x_[i_](xRHO_i, 0) != x_[i_](xRHO_i, 0))
    {
      // if a depth state has gone NaN, reset it
      x_[i_](xRHO_i, 0) = 1.0/(2.0*min_depth_);
    }
    if (x_[i_](xRHO_i, 0) < 0.0)
    {
      // If the state has gone negative, reset it
      double err = 1.0/(2.0*min_depth_) - x_[i_](xRHO_i, 0);
      P_[i_](dxRHO_i, dxRHO_i) += err*err;
      x_[i_](xRHO_i, 0) = 1.0/(2.0*min_depth_);
    }
    else if (x_[i_](xRHO_i, 0) > 1e2)
    {
      // If the state has grown unreasonably large, reset it
      P_[i_](dxRHO_i, dxRHO_i) = P0_feat_(2,2);
      x_[i_](xRHO_i, 0) = 1.0/(2.0*min_depth_);
    }
  }
}

}
