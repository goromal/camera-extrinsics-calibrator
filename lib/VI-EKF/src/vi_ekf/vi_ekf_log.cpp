#include "vi_ekf.h"

namespace vi_ekf
{

void VIEKF::log_state(const double t, const xVector& x, const dxVector& P, const uVector& u, const dxVector& dx)
{
  if (log_)
  {
    (*log_)[LOG_STATE].write((char*)&t, sizeof(double));
    (*log_)[LOG_STATE].write((char*)x.data(), sizeof(double) * x.rows());
    (*log_)[LOG_STATE].write((char*)P.data(), sizeof(double) * P.rows());

    (*log_)[LOG_INPUT].write((char*)&t, sizeof(double));
    (*log_)[LOG_INPUT].write((char*)u.data(), sizeof(double) * u.rows());

    (*log_)[LOG_XDOT].write((char*)&t, sizeof(double));
    (*log_)[LOG_XDOT].write((char*)dx.data(), sizeof(double) * dx.rows());

    (*log_)[LOG_FEATURE_IDS].write((char*)&t, sizeof(double));
    for (int i = 0; i < NUM_FEATURES; i++)
    {
      double idd = -1.0;
      if (i < len_features_)
        idd = (double)current_feature_ids_[i];
      (*log_)[LOG_FEATURE_IDS].write((char*)&idd, sizeof(double));
    }
  }
}

void VIEKF::log_global_position(const Xformd &truth_global_transform) //Vector3d pos, const Vector4d att)
{ 
  if (log_)
  {
    Xformd global_pose = get_global_pose();

    double t = t_[i_] - start_t_;
    (*log_)[LOG_GLOBAL].write((char*)&t, sizeof(double));
    (*log_)[LOG_GLOBAL].write((char*)truth_global_transform.q_.arr_.data(), sizeof(double) * 4);
    (*log_)[LOG_GLOBAL].write((char*)truth_global_transform.t_.data(), sizeof(double) * 3);
    (*log_)[LOG_GLOBAL].write((char*)global_pose.q_.arr_.data(), sizeof(double) * 4);
    (*log_)[LOG_GLOBAL].write((char*)global_pose.t_.data(), sizeof(double) * 3);
  }
}

void VIEKF::log_measurement(const measurement_type_t type, const double t, const int dim, const MatrixXd& z, const MatrixXd& zhat, const bool active, const int id)
{
  if (log_ != NULL)
  {
    (*log_)[type].write((char*)&t, sizeof(double));
    (*log_)[type].write((char*)z.data(), sizeof(double) * dim);
    (*log_)[type].write((char*)zhat.data(), sizeof(double) * dim);
    double ac = 1.0 * active;
    (*log_)[type].write((char*)&ac, sizeof(double));
    if (type == FEAT || type == QZETA || type == DEPTH || type == INV_DEPTH || type == PIXEL_VEL)
    {
      double idd = 1.0 * id;
      (*log_)[type].write((char*)&idd, sizeof(double));
    }
  }
}

void VIEKF::disable_logger()
{
  for (auto i = log_->begin(); i != log_->end(); i++)
  {
    i->close();
  }
  delete log_;
  log_ = NULL;
}

void VIEKF::init_logger(string root_filename, string prefix)
{
  log_ = new std::vector<std::ofstream>;
  (*log_).resize(TOTAL_LOGS);

  // Make the directory
  int result = system(("mkdir -p " + root_filename).c_str());
  (void)result;

  // A logger for the results of propagation
  for (int i = 0; i < TOTAL_MEAS; i++)
  {
    (*log_)[i].open(root_filename + "/" + measurement_names[i] + ".bin",  std::ofstream::out | std::ofstream::trunc);
  }
  (*log_)[LOG_STATE].open(root_filename + "/" + prefix +"prop.bin", std::ofstream::out | std::ofstream::trunc);
  (*log_)[LOG_FEATURE_IDS].open(root_filename + "/" + prefix +"feat_id.bin", std::ofstream::out | std::ofstream::trunc);
  (*log_)[LOG_CONF].open(root_filename + "/" + prefix +"conf.txt", std::ofstream::out | std::ofstream::trunc);
  (*log_)[LOG_INPUT].open(root_filename + "/" + prefix +"input.bin", std::ofstream::out | std::ofstream::trunc);
  (*log_)[LOG_XDOT].open(root_filename + "/" + prefix +"xdot.bin", std::ofstream::out | std::ofstream::trunc);
  (*log_)[LOG_KF].open(root_filename + "/" + prefix +"kf.bin", std::ofstream::out | std::ofstream::trunc);
  (*log_)[LOG_DEBUG].open(root_filename + "/" + prefix +"debug.txt", std::ofstream::out | std::ofstream::trunc);

  // Save configuration
  (*log_)[LOG_CONF] << "Test Num: " << root_filename << "\n";
  (*log_)[LOG_CONF] << "x0" << x_[i_].block<(int)xZ, 1>(0,0).transpose() << "\n";
  (*log_)[LOG_CONF] << "P0: " << P_[i_].diagonal().block<(int)xZ, 1>(0,0).transpose() << "\n";
  (*log_)[LOG_CONF] << "P0_feat: " << P0_feat_.diagonal().transpose() << "\n";
  (*log_)[LOG_CONF] << "Qx: " << Qx_.diagonal().block<(int)dxZ, 1>(0,0).transpose() << "\n";
  (*log_)[LOG_CONF] << "Qx_feat: " << Qx_.diagonal().block<3, 1>((int)dxZ,0).transpose() << "\n";
  (*log_)[LOG_CONF] << "Qu: " << Qu_.diagonal().transpose() << "\n";
  (*log_)[LOG_CONF] << "q_b_c: " << q_b_c_.arr_.transpose() << "\n";
  (*log_)[LOG_CONF] << "p_b_c: " << p_b_c_.transpose() << "\n";
  (*log_)[LOG_CONF] << "lambda: " << lambda_.block<(int)dxZ,1>(0,0).transpose() << "\n";
  (*log_)[LOG_CONF] << "lambda_feat: " << lambda_.block<3,1>((int)dxZ,0).transpose() << "\n";
  (*log_)[LOG_CONF] << "partial_update: " << partial_update_ << "\n";
  (*log_)[LOG_CONF] << "keyframe reset: " << keyframe_reset_ << "\n";
  (*log_)[LOG_CONF] << "Using Drag Term: " << use_drag_term_ << "\n";
  (*log_)[LOG_CONF] << "keyframe overlap: " << keyframe_overlap_threshold_ << "\n";
  (*log_)[LOG_CONF] << "num features: " << NUM_FEATURES << "\n";
  (*log_)[LOG_CONF] << "min_depth: " << min_depth_ << std::endl;
}

}


