#include <tuple>
#include <Eigen/Core>
#include "geometry/xform.h"
#include "multirotor_sim/simulator.h"
#include "vi_ekf.h"

using namespace Eigen;
using namespace xform;

typedef std::vector<std::pair<double, Xformd>, aligned_allocator<std::pair<double, Xformd>>> MocapVec;
typedef std::vector<std::pair<double, Xformd>, aligned_allocator<std::pair<double, Xformd>>> VoVec;
typedef std::vector<std::pair<double, Vector6d>, aligned_allocator<std::pair<double, Vector6d>>> ImuVec;
struct Feat
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW double t;
  int id;
  Vector2d z;
  double depth;
};
typedef std::vector<Feat, aligned_allocator<Feat>> FeatVec;

class MotionCaptureSimulator
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MotionCaptureSimulator();

  const MatrixXd &getStateTruth()
  {
    return x_truth_;
  }
  const MatrixXd &getStateEst()
  {
    return x_est_;
  }
  const ImuVec &getImu() const
  {
    return imu_;
  }
  const MocapVec &getMocap() const
  {
    return mocap_;
  }
  const VoVec &getVO() const
  {
    return vo_;
  }
  const FeatVec &getFeat() const
  {
    return features_;
  }
  const Vector6d &getImuBias() const
  {
    return imu_bias_;
  }
  const Xformd &getXformB2M() const
  {
    return xb2m_;
  }
  const Xformd &getXformB2C() const
  {
    return xb2c_;
  }
  const Xformd &getXformB2U() const
  {
    return xb2u_;
  }
  const Quatd &getQuatB2U() const
  {
    return xb2u_.q_;
  }
  double getTimeOffset() const
  {
    return dt_;
  }
  const Matrix6d &getImuCov() const
  {
    return imu_cov_;
  }
  const Matrix6d &getMocapCov() const
  {
    return mocap_cov_;
  }
  const std::vector<Vector3d, aligned_allocator<Vector3d>> &getLandmarks() const
  {
    return landmarks_;
  }
  void logTruth(const std::string &filename);
  void logMocap(const std::string &filename);
  void logImu(const std::string &filename);
  void logFeat(const std::string &filename);
  void logLandmarks(const string &filename);
  void logXTruth(const std::string &filename);
  void logXEst(const std::string &filename);

  void run();

private:
  vi_ekf::VIEKF ekf;
  ImuVec imu_;
  MocapVec mocap_;
  VoVec vo_;
  std::vector<Vector3d, aligned_allocator<Vector3d>> landmarks_;
  FeatVec features_;
  std::vector<std::pair<double, dynamics::xVector>, aligned_allocator<std::pair<double, dynamics::xVector>>> truth_;
  Vector6d imu_bias_;
  double dt_;
  Matrix6d imu_cov_;
  Matrix6d mocap_cov_;
  Xformd xb2m_, xb2u_, xb2c_;
  MatrixXd x_truth_;
  MatrixXd x_est_;
};
