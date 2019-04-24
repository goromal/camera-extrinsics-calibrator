#pragma once

#include <fstream>
#include <tuple>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include "geometry/xform.h"
#include "multirotor_sim/utils.h"
#include "XformLocalParam.h"
#include "FeatProj.h"
#include "math_helper.h"

//class FeatProj;
typedef ceres::AutoDiffCostFunction<FeatProj, 2, 3, 7, 2, 2, 1, 5, 1> FeatProjFactor;
using namespace xform;
using namespace Eigen;

class CamExtCalibrator
{

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CamExtCalibrator(const std::string &filename);
  void setup(MatrixXd &stateMeas, MatrixXd &featureMeas, MatrixXd &landmarks);
  bool solve();
  const Xformd &getEstXformB2C() const;
  void logxhatVOTC(const std::string &filename) const;
  void logPixelResiduals(const std::string &filename) const;

private:
  Xformd XB2C_ = Xformd::Identity();
  MatrixXd xhat_vo_;
  MatrixXd landmarks_;
  std::vector<bool> initialized_landmarks_;
  Vector2d cam_focal_len_;
  Vector2d cam_center_;
  Vector2d cam_image_size_;
  double cam_skew_;
  double cam_dt_;
  Vector5d cam_distortion_;
  ceres::Problem *problem_ = nullptr;
  std::vector<ceres::ResidualBlockId> feat_factors_;
};
