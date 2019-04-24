#include <vector>
#include <fstream>

#include "geometry/xform.h"
#include "multirotor_sim/simulator.h"

#include "sim.h"

MotionCaptureSimulator::MotionCaptureSimulator()
{
  imu_.clear();
  mocap_.clear();
  vo_.clear();
}


void MotionCaptureSimulator::run()
{
  Simulator multirotor(true);
  multirotor.load("../params/sim_params.yaml");

  ekf.load("../params/ekf_baseline_params.yaml", "../params/sim_params.yaml");
  ekf.propagate_state(multirotor.get_imu_prev(), 0.0);
  bool keyframe_reset;
  if (ekf.get_keyframe_reset())
    ekf.register_keyframe_reset_callback([&keyframe_reset]() mutable {keyframe_reset = true;});
  bool do_camera_update = true;

  std::vector<Simulator::measurement_t, Eigen::aligned_allocator<Simulator::measurement_t>> meas_list;

  imu_bias_.segment<3>(0) = multirotor.get_accel_bias();
  imu_bias_.segment<3>(3) = multirotor.get_gyro_bias();
  dt_ = multirotor.mocap_transmission_time_;
  mocap_cov_ = multirotor.get_mocap_noise_covariance();
  imu_cov_ = multirotor.get_imu_noise_covariance();
  xb2m_.t_ = multirotor.p_b_m_;
  xb2m_.q_ = multirotor.q_b_m_;
  xb2c_.t_ = multirotor.p_b_c_;
  xb2c_.q_ = multirotor.q_b_c_;
  xb2u_.t_ = multirotor.p_b_u_;
  xb2u_.q_ = multirotor.q_b_u_;

  int n = 0;
  int state_size = 7500; // <<<<<<<<<<<< NEED TO CHANGE THIS, AD HOC FOR NOW <<<<<<<<<<<<<<<<<<<<<<<<
  x_truth_.resize(8, state_size);
  x_est_.resize(8, state_size);

  while (multirotor.run())
  {
    truth_.push_back(std::pair<double, dynamics::xVector> {multirotor.t_, multirotor.dyn_.get_state()});
    multirotor.get_measurements(meas_list);
    ekf.propagate_state(multirotor.get_imu_prev(), multirotor.t_);

    Xformd mocap_meas, vo_meas;
    Vector2d feat_meas;
    double feat_depth;
    int feat_id;
    bool got_mocap = false;
    bool got_vo = false;
    bool got_feature = false;

    for (int i = 0; i < meas_list.size(); i++)
    {
      switch (meas_list[i].type)
      {
      case Simulator::ACC:
      {
        imu_.push_back(std::pair<double, Vector6d> {multirotor.t_, multirotor.get_imu_prev()});
        break;
      }
      case Simulator::ATT:
        got_mocap = true;
        mocap_meas.q_ = meas_list[i].z;
        ekf.add_measurement(multirotor.t_, meas_list[i].z, vi_ekf::VIEKF::ATT, meas_list[i].R, meas_list[i].active);
        ekf.handle_measurements();
        break;
      case Simulator::POS:
        mocap_meas.t_ = meas_list[i].z;
        got_mocap = true;
        ekf.add_measurement(multirotor.t_, meas_list[i].z, vi_ekf::VIEKF::POS, meas_list[i].R, meas_list[i].active);
        ekf.handle_measurements();
        break;
      case Simulator::VO:
        vo_meas = Xformd(meas_list[i].z);
        got_vo = true;
        break;
      case Simulator::FEAT:
      {
        feat_meas = meas_list[i].z;
        feat_depth = meas_list[i].depth;
        feat_id = meas_list[i].feature_id;
        features_.push_back(Feat{multirotor.t_, feat_id, feat_meas, feat_depth});
        got_feature = true;
        break;
      }
      default:
        break;
      }
    }

    if (got_mocap)
    {
      mocap_.push_back(std::pair<double, Xformd> {multirotor.t_, mocap_meas});
      dynamics::xVector quad_state = multirotor.dyn_.get_state();
      x_truth_.col(n) << multirotor.t_, quad_state(dynamics::PX), quad_state(dynamics::PY), quad_state(dynamics::PZ),
                   quad_state(dynamics::QW), quad_state(dynamics::QX), quad_state(dynamics::QY), quad_state(dynamics::QZ);

      xVector ekf_state = ekf.get_state();
      x_est_.col(n) << multirotor.t_, ekf_state(0), ekf_state(1), ekf_state(2), ekf_state(6), ekf_state(7), ekf_state(8),
                 ekf_state(9);
      n++;
    }

    if (got_vo)
    {
      vo_.push_back(std::pair<double, Xformd> {multirotor.t_, vo_meas});
    }
  }
  landmarks_ = multirotor.env_.get_points();
  x_truth_.conservativeResize(8, n);
  x_est_.conservativeResize(8, n);
}

void MotionCaptureSimulator::logTruth(const string &filename)
{
  std::ofstream f(filename);
  for (int i = 0; i < truth_.size(); i++)
  {
    f.write((char *)&truth_[i].first, sizeof(double));
    f.write((char *)&truth_[i].second, sizeof(double)*16);
  }
  f.close();
}

void MotionCaptureSimulator::logXTruth(const std::string &filename)
{
  std::ofstream f(filename);
  for (int i = 0; i < mocap_.size(); i++)
  {
    f.write((char *)&x_truth_(0, i), sizeof(double));
    f.write((char *)&x_truth_(1, i), sizeof(double));
    f.write((char *)&x_truth_(2, i), sizeof(double));
    f.write((char *)&x_truth_(3, i), sizeof(double));
    f.write((char *)&x_truth_(4, i), sizeof(double));
    f.write((char *)&x_truth_(5, i), sizeof(double));
    f.write((char *)&x_truth_(6, i), sizeof(double));
    f.write((char *)&x_truth_(7, i), sizeof(double));
  }
  f.close();
}

void MotionCaptureSimulator::logXEst(const std::string &filename)
{
  std::ofstream f(filename);
  for (int i = 0; i < mocap_.size(); i++)
  {
    f.write((char *)&x_est_(0, i), sizeof(double));
    f.write((char *)&x_est_(1, i), sizeof(double));
    f.write((char *)&x_est_(2, i), sizeof(double));
    f.write((char *)&x_est_(3, i), sizeof(double));
    f.write((char *)&x_est_(4, i), sizeof(double));
    f.write((char *)&x_est_(5, i), sizeof(double));
    f.write((char *)&x_est_(6, i), sizeof(double));
    f.write((char *)&x_est_(7, i), sizeof(double));
  }
  f.close();
}

void MotionCaptureSimulator::logFeat(const string &filename)
{
  std::ofstream f(filename);
  for (int i = 0; i < features_.size(); i++)
  {
    double id = features_[i].id;
    f.write((char *) &features_[i].t, sizeof(double));
    f.write((char *) &id, sizeof(double));
    f.write((char *) features_[i].z.data(), sizeof(double)*2);
  }
  f.close();
}

void MotionCaptureSimulator::logImu(const string &filename)
{
  std::ofstream f(filename);
  for (int i = 0; i < imu_.size(); i++)
  {
    f.write((char *)&imu_[i].first, sizeof(double));
    f.write((char *)imu_[i].second.data(), sizeof(double)*6);
  }

  f.close();
}

void MotionCaptureSimulator::logLandmarks(const string &filename)
{
  std::ofstream f(filename);
  for (int i = 0; i < landmarks_.size(); i++)
  {
    f.write((char *) landmarks_[i].data(), sizeof(double)*3);
  }
  f.close();
}

void MotionCaptureSimulator::logMocap(const string &filename)
{
  std::ofstream f(filename);
  for (int i = 0; i < mocap_.size(); i++)
  {
    f.write((char *)&mocap_[i].first, sizeof(double));
    f.write((char *)mocap_[i].second.data(), sizeof(double)*7);
    Vector6d xdot;
    if (i == 0)
      xdot = (mocap_[1].second - mocap_[0].second) / (mocap_[1].first - mocap_[0].first);
    else if (i == mocap_.size() - 1)
      xdot = (mocap_[i].second - mocap_[i-1].second) / (mocap_[i].first - mocap_[i-1].first);
    else
      xdot = (mocap_[i+1].second - mocap_[i-1].second) / (mocap_[i+1].first - mocap_[i-1].first);
    f.write((char *)xdot.data(), sizeof(double)*6);
  }
  f.close();
}
