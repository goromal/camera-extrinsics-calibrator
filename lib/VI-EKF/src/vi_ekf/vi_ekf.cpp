#include "vi_ekf.h"

namespace vi_ekf
{

VIEKF::VIEKF(){}

void VIEKF::init(Matrix<double, xZ,1>& x0, Matrix<double, dxZ,1> &P0, Matrix<double, dxZ,1> &Qx,
                 Matrix<double, dxZ,1> &lambda, uVector &Qu, Vector3d& P0_feat, Vector3d& Qx_feat,
                 Vector3d& lambda_feat, Vector2d &cam_center, Vector2d &focal_len, Vector4d &q_b_c,
                 Vector3d &p_b_c, double min_depth, std::string log_directory, bool use_drag_term, bool partial_update,
                 bool use_keyframe_reset, double keyframe_overlap, int cov_prop_skips, std::string prefix)
{
  x_.resize(LEN_STATE_HIST);
  P_.resize(LEN_STATE_HIST);
  t_.resize(LEN_STATE_HIST, -1);

  u_.clear();
  zbuf_.clear();

  i_ = 0;

  xp_.setZero();
  Qx_.setZero();
  x_[i_].block<(int)xZ, 1>(0,0) = x0;
  P_[i_].block<(int)dxZ, (int)dxZ>(0,0) = P0.asDiagonal();
  Qx_.block<(int)dxZ, (int)dxZ>(0,0) = Qx.asDiagonal();
  lambda_.block<(int)dxZ, 1>(0,0) = lambda;
  
  for (int i = 0; i < NUM_FEATURES; i++)
  {
    P_[i_].block<3,3>(dxZ+3*i, dxZ+3*i) = P0_feat.asDiagonal();
    Qx_.block<3,3>(dxZ+3*i, dxZ+3*i) = Qx_feat.asDiagonal();
    lambda_.block<3,1>(dxZ+3*i,0) = lambda_feat;
  }
  
  Qu_ = Qu.asDiagonal();
  P0_feat_ = P0_feat.asDiagonal();
  
  Lambda_ = dx_ones_ * lambda_.transpose() + lambda_*dx_ones_.transpose() - lambda_*lambda_.transpose();
  
  len_features_ = 0;
  next_feature_id_ = 0;
  
  current_feature_ids_.clear();
  
  // set cam-to-body
  p_b_c_ = p_b_c;
  q_b_c_ = Quatd(q_b_c);
  
  // set camera intrinsics
  cam_center_ = cam_center;
  cam_F_ << focal_len(0), 0, 0,
            0, focal_len(1), 0;
  
  use_drag_term_ = use_drag_term;
  partial_update_ = partial_update;
  keyframe_reset_ = use_keyframe_reset;
  cov_prop_skips_ = cov_prop_skips;
  start_t_ = NAN; // indicate that we need to initialize the filter
  
  min_depth_ = min_depth;
  
  current_node_global_pose_ = Xformd::Identity();
  global_pose_cov_.setZero();
  
  keyframe_overlap_threshold_ = keyframe_overlap;
  keyframe_features_.clear();
  edges_.clear();
  keyframe_reset_callback_ = nullptr;
  
  if (log_directory.compare("~") != 0)
    init_logger(log_directory);
  
  K_.setZero();
  H_.setZero();
}

void VIEKF::set_x0(const Matrix<double, xZ, 1>& _x0)
{
  x_[i_].topRows(xZ) = _x0;
}

void VIEKF::register_keyframe_reset_callback(std::function<void(void)> cb)
{
  keyframe_reset_callback_ = cb;
}


VIEKF::~VIEKF()
{
  if (log_)
  {
    for (std::vector<std::ofstream>::iterator it=log_->begin(); it!=log_->end(); ++it)
    {
      (*it) << endl;
      (*it).close();
    }
  }
}

void VIEKF::set_imu_bias(const Vector3d& b_g, const Vector3d& b_a)
{
  x_[i_].block<3,1>((int)xB_G,0) = b_g;
  x_[i_].block<3,1>((int)xB_A,0) = b_a;
}


const xVector& VIEKF::get_state() const
{
  return x_[i_];
}

const Xformd &VIEKF::get_current_node_global_pose() const
{
  return current_node_global_pose_;
}

const dxMatrix& VIEKF::get_covariance() const
{
  return P_[i_];
}

const dxVector VIEKF::get_covariance_diagonal() const
{
  dxVector ret = P_[i_].diagonal();
  return ret;
}



VectorXd VIEKF::get_depths() const
{
  VectorXd out(len_features_);
  for (int i = 0; i < len_features_; i++)
  {
    out[i] = 1.0/x_[i_]((int)xZ + 4 + 5*i);
  }
  return out;
}

MatrixXd VIEKF::get_zetas() const
{
  MatrixXd out(3, len_features_);
  for (int i = 0; i < len_features_; i++)
  {
    Vector4d qzeta = x_[i_].block<4,1>(xZ + 5*i,0);
    out.block<3,1>(0,i) = Quatd(qzeta).rota(e_z);
  }
  return out;
}

MatrixXd VIEKF::get_qzetas() const
{
  MatrixXd out(4, len_features_);
  for (int i = 0; i < len_features_; i++)
  {
    out.block<4,1>(0,i) = x_[i_].block<4,1>(xZ + 5*i,0);
  }
  return out;
}

VectorXd VIEKF::get_zeta(const int i) const
{
  Vector4d qzeta_i = x_[i_].block<4,1>(xZ + 5*i,0);
  return Quatd(qzeta_i).rota(e_z);
}

double VIEKF::get_depth(const int id) const
{
  int i = global_to_local_feature_id(id);
  return 1.0/x_[i_]((int)xZ + 4 + 5*i);
}

Vector2d VIEKF::get_feat(const int id) const
{
  int i = global_to_local_feature_id(id);
  Quatd q_zeta(x_[i_].block<4,1>(xZ+i*5, 0));
  Vector3d zeta = q_zeta.rota(e_z);
  double ezT_zeta = e_z.transpose() * zeta;
  return cam_F_ * zeta / ezT_zeta + cam_center_;
}

void VIEKF::propagate_state(const uVector &u, const double t, bool save_input)
{
  if (save_input)
  {
    input_t input;
    input.t = t;
    input.u = u;
    u_.push_front(input);
  }

  if (std::isnan(start_t_))
  {
    start_t_ = t;
    t_[i_] = t;
    return;
  }

  double dt = t - t_[i_];
  if (dt < 1e-6)
    return;

  NAN_CHECK;

  // Calculate Dynamics and Jacobians
  dynamics(x_[i_], u, true, true);

  NAN_CHECK;
  int ip = (i_ + 1) % LEN_STATE_HIST; // next state history index

  // Propagate State and Covariance
  boxplus(x_[i_], dx_*dt, x_[ip]);
  A_ = I_big_ + A_*dt;
  P_[ip] = A_ * P_[i_]* A_.transpose() + G_ * Qu_ * G_.transpose() + Qx_;
  t_[ip] = t;
  i_ = ip;

  NAN_CHECK;
  
  // Correct any impossible depth states
  fix_depth();
  
  NAN_CHECK;
  NEGATIVE_DEPTH;
  
  log_state(t, x_[i_], P_[i_].diagonal(), u - x_[i_].block<6,1>(xB_A, 0), dx_);
}

}








