#pragma once

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/StdVector"
#include "unsupported/Eigen/MatrixFunctions"

#include <deque>
#include <set>
#include <map>
#include <functional>
#include <fstream>
#include <chrono>
#include <iostream>
#include <random>

#include "geometry/quat.h"
#include "geometry/xform.h"
#include "math_helper.h"

using namespace quat;
using namespace xform;
using namespace std;
using namespace Eigen;

#define NO_NANS(mat) (mat.array() == mat.array()).all()

#ifndef NDEBUG
#define NAN_CHECK if (NaNsInTheHouse()) { std::cout << "NaNs In The House at line " << __LINE__ << "!!!\n"; exit(0); }
#define NEGATIVE_DEPTH if (NegativeDepth()) std::cout << "Negative Depth " << __LINE__ << "!!!\n"
#define CHECK_MAT_FOR_NANS(mat) if ((K_.array() != K_.array()).any()) { std::cout << "NaN detected in " << #mat << " at line " << __LINE__ << "!!!\n" << mat << "\n"; exit(0); }
#else
#define NAN_CHECK {}
#define NEGATIVE_DEPTH {}
#define CHECK_MAT_FOR_NANS(mat) {}
#endif

#ifndef NUM_FEATURES
#ifndef NDEBUG
#define NUM_FEATURES 3
#else
#define NUM_FEATURES 12
#endif
#endif

#define MAX_X 17+NUM_FEATURES*5
#define MAX_DX 16+NUM_FEATURES*3

#define LEN_STATE_HIST 25
#define LEN_MEAS_HIST 20

typedef Matrix<double, MAX_X, 1> xVector;
typedef Matrix<double, MAX_DX, 1> dxVector;
typedef Matrix<double, MAX_X, MAX_X> xMatrix;
typedef Matrix<double, MAX_DX, MAX_DX> dxMatrix;
typedef Matrix<double, MAX_DX, 6> dxuMatrix;
typedef Matrix<double, 6, 1> uVector;
typedef Matrix<double, 6, 6> Matrix6d;
typedef Matrix<double, 4, 1> zVector;
typedef Matrix<double, 3, MAX_DX> hMatrix;

namespace vi_ekf
{

class VIEKF;

typedef void (VIEKF::*measurement_function_ptr)(const xVector& x, zVector& h, hMatrix& H, const int id) const;

static const Vector3d gravity = [] {
  Vector3d tmp;
  tmp << 0, 0, 9.80665;
  return tmp;
}();

static const Vector3d khat = [] {
  Vector3d tmp;
  tmp << 0, 0, 1.0;
  return tmp;
}();

class VIEKF
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW // http://eigen.tuxfamily.org/dox-devel/group__TopicStructHavingEigenMembers.html

  enum : int{
    xPOS = 0,
    xVEL = 3,
    xATT = 6,
    xB_A = 10,
    xB_G = 13,
    xMU = 16,
    xZ = 17
  };

  enum : int{
    uA = 0,
    uG = 3,
    uTOTAL = 6
  };

  enum : int {
    dxPOS = 0,
    dxVEL = 3,
    dxATT = 6,
    dxB_A = 9,
    dxB_G = 12,
    dxMU = 15,
    dxZ = 16
  };

  typedef enum {
    ACC,
    ALT,
    ATT,
    POS,
    VEL,
    QZETA,
    FEAT,
    PIXEL_VEL,
    DEPTH,
    INV_DEPTH,
    TOTAL_MEAS
  } measurement_type_t;

  typedef struct{
    Xformd transform;
    Matrix6d cov;
  } edge_t;

  typedef enum {
    MEAS_SUCCESS,
    MEAS_GATED,
    MEAS_NAN,
    MEAS_INVALID,
    MEAS_NEW_FEATURE
  } meas_result_t;

private:
  typedef enum {
    LOG_STATE = TOTAL_MEAS,
    LOG_FEATURE_IDS,
    LOG_INPUT,
    LOG_XDOT,
    LOG_GLOBAL,
    LOG_CONF,
    LOG_KF,
    LOG_DEBUG,
    TOTAL_LOGS
  } log_type_t;


  // State, Covariance History
  int i_;
  std::vector<xVector, aligned_allocator<xVector>> x_;
  std::vector<dxMatrix, aligned_allocator<dxMatrix>> P_;
  std::vector<double> t_;

  // Input Buffer
  typedef struct
  {
    double t;
    uVector u;
  } input_t;
  std::deque<input_t, aligned_allocator<input_t>> u_;

  // Measurement Buffer
  typedef struct
  {
    double t;
    measurement_type_t type;
    zVector z;
    Matrix3d R;
    bool active;
    int id;
    double depth;
    int zdim;
    int rdim;
    bool handled;
  } measurement_t;
  std::deque<measurement_t, aligned_allocator<measurement_t>> zbuf_;

  // State and Covariance and Process Noise Matrices
  dxMatrix Qx_;
  Matrix<double, 6, 6> Qu_;

  // Partial Update Gains
  dxVector lambda_;
  dxMatrix Lambda_;

  // Initial uncertainty on features
  Matrix3d P0_feat_;

  // Internal bookkeeping variables
  double start_t_;
  int len_features_;
  int next_feature_id_;
  std::vector<int> current_feature_ids_;
  std::vector<int> keyframe_features_;
  double keyframe_overlap_threshold_;

  std::deque<edge_t> edges_;

  // Matrix Workspace
  dxMatrix A_;
  dxuMatrix G_;
  dxVector dx_;
  const dxMatrix I_big_ = dxMatrix::Identity();
  const dxMatrix Ones_big_ = dxMatrix::Constant(1.0);
  const dxVector dx_ones_ = dxVector::Constant(1.0);
  xVector xp_;
  Matrix<double, MAX_DX, 3>  K_;
  zVector zhat_;
  hMatrix H_;

  // EKF Configuration Parameters
  bool use_drag_term_;
  bool keyframe_reset_;
  bool partial_update_;
  double min_depth_;
  int cov_prop_skips_;

  // Camera Intrinsics and Extrinsics
  Vector2d cam_center_;
  Matrix<double, 2, 3> cam_F_;
  Quatd q_b_c_;
  Vector3d p_b_c_;

  Matrix6d global_pose_cov_;
  Xformd current_node_global_pose_;
  std::default_random_engine generator_;
  std::normal_distribution<double> normal_;

  std::function<void(void)> keyframe_reset_callback_;

  // Log Stuff
  std::vector<std::ofstream>* log_ = nullptr;

public:

  VIEKF();
  ~VIEKF();
#ifdef MC_SIM
  void load(string ekf_file, string common_file, bool use_logger=true, string prefix="");
#endif
  void init(Matrix<double, xZ,1> &x0, Matrix<double, dxZ,1> &P0, Matrix<double, dxZ,1> &Qx,
            Matrix<double, dxZ,1> &lambda, uVector &Qu, Vector3d& P0_feat, Vector3d& Qx_feat,
            Vector3d& lambda_feat, Vector2d& cam_center, Vector2d& focal_len,
            Vector4d& q_b_c, Vector3d &p_b_c, double min_depth, std::string log_directory, bool use_drag_term,
            bool partial_update, bool use_keyframe_reset, double keyframe_overlap, int cov_prop_skips, string prefix="");

  inline double now() const
  {
    std::chrono::microseconds now = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch());
    return (double)now.count()*1e-6;
  }

  // Errors
  bool NaNsInTheHouse() const;
  bool BlowingUp() const;
  bool NegativeDepth() const;

  // Helpers
  int global_to_local_feature_id(const int global_id) const;
  const std::vector<int>& tracked_features() const;


  // Getters and Setters
  VectorXd get_depths() const;
  MatrixXd get_zetas() const;
  MatrixXd get_qzetas() const;
  VectorXd get_zeta(const int i) const;
  Vector2d get_feat(const int id) const;
  const Xformd &get_current_node_global_pose() const;
  const xVector &get_state() const;
  const dxMatrix &get_covariance() const;
  const dxVector get_covariance_diagonal() const;
  double get_depth(const int id) const;
  inline int get_len_features() const { return len_features_; }

  void set_x0(const Matrix<double, xZ, 1>& _x0);
  void set_imu_bias(const Vector3d& b_g, const Vector3d& b_a);
  void set_drag_term(const bool use_drag_term) {use_drag_term_ = use_drag_term;}
  bool get_drag_term() const {return use_drag_term_;}
  bool get_keyframe_reset() const {return keyframe_reset_;}

  bool init_feature(const Vector2d &l, const int id, const double depth=-1.0);
  void clear_feature(const int id);
  void keep_only_features(const std::vector<int> features);

  // State Propagation
  void boxplus(const xVector &x, const dxVector &dx, xVector &out) const;
  void boxminus(const xVector& x1, const xVector &x2, dxVector& out) const;
  void step(const uVector& u, const double t);
  void propagate_state(const uVector& u, const double t, bool save_input=true);
  void dynamics(const xVector &x, const uVector& u, dxVector& xdot, dxMatrix& dfdx, dxuMatrix& dfdu);
  void dynamics(const xVector &x, const uVector& u, bool state = true, bool jac = true);

  // Measurement Updates
  void handle_measurements(std::vector<int> *gated_feature_ids=nullptr);
  meas_result_t add_measurement(const double t, const VectorXd& z, const measurement_type_t& meas_type, const MatrixXd& R, bool active=false, const int id=-1, const double depth=NAN);
  meas_result_t update(measurement_t &meas);
  void h_acc(const xVector& x, zVector& h, hMatrix& H, const int id) const;
  void h_alt(const xVector& x, zVector& h, hMatrix& H, const int id) const;
  void h_att(const xVector& x, zVector& h, hMatrix& H, const int id) const;
  void h_pos(const xVector& x, zVector& h, hMatrix& H, const int id) const;
  void h_vel(const xVector& x, zVector& h, hMatrix& H, const int id) const;
  void h_qzeta(const xVector& x, zVector& h, hMatrix& H, const int id) const;
  void h_feat(const xVector& x, zVector& h, hMatrix& H, const int id) const;
  void h_depth(const xVector& x, zVector& h, hMatrix& H, const int id) const;
  void h_inv_depth(const xVector& x, zVector& h, hMatrix& H, const int id) const;
  void h_pixel_vel(const xVector& x, zVector& h, hMatrix& H, const int id) const;

  // Keyframe Reset
  void propagate_global_covariance(const Matrix6d &P_prev, const edge_t& edge, Matrix6d &P_new) const;
  void keyframe_reset(const xVector &xm, xVector &xp, dxMatrix &N);
  void keyframe_reset();
  void register_keyframe_reset_callback(std::function<void(void)> cb);
  Xformd get_global_pose() const;
  Matrix6d get_global_cov() const;

  // Logger
  void log_state(const double t, const xVector& x, const dxVector& P, const uVector& u, const dxVector& dx);
  void log_measurement(const measurement_type_t type, const double t, const int dim, const MatrixXd& z, const MatrixXd& zhat, const bool active, const int id);
  void init_logger(std::string root_filename, string prefix="");
  void disable_logger();
  void log_global_position(const Xformd& truth_global_transform);

  // Inequality Constraint on Depth
  void fix_depth();
};

static std::vector<std::string> measurement_names = [] {
  std::vector<std::string> tmp;
  tmp.resize(VIEKF::TOTAL_MEAS);
  tmp[VIEKF::ACC] = "ACC";
  tmp[VIEKF::ALT] = "ALT";
  tmp[VIEKF::ATT] = "ATT";
  tmp[VIEKF::POS] = "POS";
  tmp[VIEKF::VEL] = "VEL";
  tmp[VIEKF::QZETA] = "QZETA";
  tmp[VIEKF::FEAT] = "FEAT";
  tmp[VIEKF::DEPTH] = "DEPTH";
  tmp[VIEKF::INV_DEPTH] = "INV_DEPTH";
  tmp[VIEKF::PIXEL_VEL] = "PIXEL_VEL";
  return tmp;
}();

static std::vector<measurement_function_ptr> measurement_functions = [] {
  std::vector<measurement_function_ptr> tmp;
  tmp.resize(VIEKF::TOTAL_MEAS);
  tmp[VIEKF::ACC] = &VIEKF::h_acc;
  tmp[VIEKF::ALT] = &VIEKF::h_alt;
  tmp[VIEKF::ATT] = &VIEKF::h_att;
  tmp[VIEKF::POS] = &VIEKF::h_pos;
  tmp[VIEKF::VEL] = &VIEKF::h_vel;
  tmp[VIEKF::QZETA] = &VIEKF::h_qzeta;
  tmp[VIEKF::FEAT] = &VIEKF::h_feat;
  tmp[VIEKF::DEPTH] = &VIEKF::h_depth;
  tmp[VIEKF::INV_DEPTH] = &VIEKF::h_inv_depth;
  tmp[VIEKF::PIXEL_VEL] = &VIEKF::h_pixel_vel;
  return tmp;
}();

}




