#include "cal.h"

CamExtCalibrator::CamExtCalibrator(const std::string &filename)
{
  // Load initial guesses and key parameters from parameter file
  Vector3d p_b2c0;
  Vector4d q_b2c0;

  get_yaml_eigen("p_b2c", filename, p_b2c0);
  get_yaml_eigen("q_b2c", filename, q_b2c0);
  get_yaml_eigen("focal_len", filename, cam_focal_len_);
  get_yaml_eigen("cam_center", filename, cam_center_);
  get_yaml_eigen("image_size", filename, cam_image_size_);

  q_b2c0.normalize();
  XB2C_.sett(p_b2c0);
  XB2C_.setq(Quatd(q_b2c0));

  cam_skew_ = 0;
  cam_dt_ = 0;
  cam_distortion_.setZero();
}

void CamExtCalibrator::setup(MatrixXd &stateMeas, MatrixXd &featureMeas, MatrixXd &landmarks)
{
  landmarks_ = landmarks; // true landmark positions

  // Ensure there's at least one state measurement before and after each feature measurement
  while (featureMeas.col(0)[0] < stateMeas.col(0)[0]) // compare start times
    removeColumn(featureMeas, 0);
  while (featureMeas.col(featureMeas.cols()-1)[0] > stateMeas.col(stateMeas.cols()-1)[0]) // compare end times
    removeColumn(featureMeas, featureMeas.cols()-1);

  // Align xhat_vo_ with feature measurements
  xhat_vo_.setZero(8, featureMeas.cols());
  int ni = 0; // Node index
  int max_feature = 0;

  for (int n = 0; n < featureMeas.cols(); ++n)
  {
    int feat_id = featureMeas.col(n)[1];
    max_feature = (feat_id > max_feature) ? feat_id : max_feature;
    while (stateMeas.col(ni)[0] <= featureMeas.col(n)[0] && ni < stateMeas.cols())
    {
      if (stateMeas.col(ni)[0] == featureMeas.col(n)[0])
      {
        xhat_vo_.col(n) = stateMeas.col(ni);
        break;
      }
      else if (stateMeas.col(ni)[0] < featureMeas.col(n)[0] &&
               stateMeas.col(ni+1)[0] > featureMeas.col(n)[0]) // interpolate
      {
        Xformd xhat1, xhat2;
        xhat1.arr_ = stateMeas.block<7, 1>(1, ni);
        xhat2.arr_ = stateMeas.block<7, 1>(1, ni+1);
        double dt = featureMeas.col(n)[0] - stateMeas.col(ni)[0];
        double DT = stateMeas.col(ni+1)[0] - stateMeas.col(ni)[0];
        Xformd xhat_interp = xhat1 + (xhat2 - xhat1) * dt / DT; // using boxplus and boxminus for quaternions
        xhat_vo_.block<1, 1>(0, n) = featureMeas.block<1, 1>(0, n);
        xhat_vo_.block<7, 1>(1, n) = xhat_interp.elements();
        break;
      }
      ++ni;
    }
  }

  // Setup problem
  if (problem_ != nullptr)
    delete problem_;
  problem_ = new ceres::Problem;
  feat_factors_.clear();

  // Parameters used in calculating the cost function (only XB2C is modifiable by the solver)
  problem_->AddParameterBlock(XB2C_.data(), 7, new XformLocalParam());
  problem_->AddParameterBlock(cam_focal_len_.data(), 2);
  problem_->SetParameterBlockConstant(cam_focal_len_.data());
  problem_->AddParameterBlock(cam_center_.data(), 2);
  problem_->SetParameterBlockConstant(cam_center_.data());
  problem_->AddParameterBlock(&cam_skew_, 1);
  problem_->SetParameterBlockConstant(&cam_skew_);
  problem_->AddParameterBlock(cam_distortion_.data(), 5);
  problem_->SetParameterBlockConstant(cam_distortion_.data());
  problem_->AddParameterBlock(&cam_dt_, 1);
  problem_->SetParameterBlockConstant(&cam_dt_);

  for (int id = 0; id < initialized_landmarks_.size(); id++)
  {
    problem_->AddParameterBlock(landmarks_.data()+3*id, 3);
    problem_->SetParameterBlockConstant(landmarks_.data()+3*id);
    initialized_landmarks_[id] = true;
  }

  // Add feature projection measurement residual blocks
  for (int n = 0; n < featureMeas.cols(); ++n)
  {
    Vector2d z = (Vector2d) featureMeas.block<2, 1>(2, n); // pixel measurement
    int feature_id = featureMeas.col(n)[1];
    FeatProjFactor *feat_factor = new FeatProjFactor(new FeatProj(z, Xformd(xhat_vo_.block<7, 1>(1, n))));
    ceres::ResidualBlockId id;
    id = problem_->AddResidualBlock(feat_factor, NULL, landmarks_.data()+3*feature_id,
                                    XB2C_.data(), cam_focal_len_.data(), cam_center_.data(), &cam_skew_,
                                    cam_distortion_.data(), &cam_dt_);
    feat_factors_.push_back(id);
  }
}

void CamExtCalibrator::logPixelResiduals(const std::string &filename) const
{
  std::vector<double> residuals;
  residuals.resize(2, feat_factors_.size());
  ceres::Problem::EvaluateOptions options;
  options.residual_blocks = feat_factors_;
  problem_->Evaluate(options, NULL, &residuals, NULL, NULL);

  ofstream f(filename);
  f.write((char *) residuals.data(), sizeof(double) * residuals.size());
  f.close();
}

bool CamExtCalibrator::solve()
{
  ceres::Solver::Options options;
#ifdef NDEBUG
  options.max_num_iterations = 1000;
#else
  options.max_num_iterations = 100;
#endif
  options.num_threads = 4;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;

  ceres::Solve(options, problem_, &summary);
  std::cout << summary.BriefReport() << std::endl << std::endl;
}

void CamExtCalibrator::logxhatVOTC(const std::string &filename) const
{
  std::ofstream f(filename);
  for (int i = 0; i < xhat_vo_.cols(); i++)
  {
    f.write((char *)(xhat_vo_.data()+8*i), sizeof(double)*8);
  }
  f.close();
}

const Xformd &CamExtCalibrator::getEstXformB2C() const
{
  return XB2C_;
}
