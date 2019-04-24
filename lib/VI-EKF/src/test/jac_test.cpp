#include "gtest/gtest.h"
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include "vi_ekf.h"
#include <random>
#include <chrono>

using namespace quat;
using namespace vi_ekf;
using namespace Eigen;

#define NUM_ITERS 100

#define QUATERNION_EQUALS(q1, q2) \
  if (sign((q1).w()) != sign((q2).w())) \
{ \
  EXPECT_NEAR((-1.0*(q1).w()), (q2).w(), 1e-8); \
  EXPECT_NEAR((-1.0*(q1).x()), (q2).x(), 1e-8); \
  EXPECT_NEAR((-1.0*(q1).y()), (q2).y(), 1e-8); \
  EXPECT_NEAR((-1.0*(q1).z()), (q2).z(), 1e-8); \
  } \
  else\
{\
  EXPECT_NEAR((q1).w(), (q2).w(), 1e-8); \
  EXPECT_NEAR((q1).x(), (q2).x(), 1e-8); \
  EXPECT_NEAR((q1).y(), (q2).y(), 1e-8); \
  EXPECT_NEAR((q1).z(), (q2).z(), 1e-8); \
  }

#define VECTOR3_EQUALS(v1, v2) \
  EXPECT_NEAR((v1)(0,0), (v2)(0,0), 1e-8); \
  EXPECT_NEAR((v1)(1,0), (v2)(1,0), 1e-8); \
  EXPECT_NEAR((v1)(2,0), (v2)(2,0), 1e-8)

#define VECTOR2_EQUALS(v1, v2) \
  EXPECT_NEAR((v1)(0,0), (v2)(0,0), 1e-8); \
  EXPECT_NEAR((v1)(1,0), (v2)(1,0), 1e-8)

#define MATRIX_EQUAL(m1, m2, tol) {\
  for (int row = 0; row < m1.rows(); row++ ) \
{ \
  for (int col = 0; col < m1.cols(); col++) \
{ \
  EXPECT_NEAR((m1)(row, col), (m2)(row, col), tol); \
  } \
  } \
  }

#define CALL_MEMBER_FN(objectptr,ptrToMember) ((objectptr).*(ptrToMember))
#define HEADER "\033[95m"
#define OKBLUE "\033[94m"
#define OKGREEN "\033[92m"
#define WARNING "\033[93m"
#define FONT_FAIL "\033[91m"
#define ENDC "\033[0m"
#define BOLD "\033[1m"
#define UNDERLINE "\033[4m"

static std::map<std::string, std::vector<int>> indexes = [] {
  std::map<std::string, std::vector<int>> tmp;
  tmp["dxPOS"] = std::vector<int> {0,3};
  tmp["dxVEL"] = std::vector<int> {3,3};
  tmp["dxATT"] = std::vector<int> {6,3};
  tmp["dxB_A"] = std::vector<int> {9,3};
  tmp["dxB_G"] = std::vector<int> {12,3};
  tmp["dxMU"] = std::vector<int> {15,1};
  tmp["uA"] = std::vector<int> {0,3};
  tmp["uG"] = std::vector<int> {3,3};
  for (int i = 0; i < 50; i++)
  {
    tmp["dxZETA_" + std::to_string(i)] = {16 + 3*i, 2};
    tmp["dxRHO_" + std::to_string(i)] = {16 + 3*i+2, 1};
  }
  return tmp;
}();


bool check_block(std::string row_id, std::string col_id, MatrixXd analytical, MatrixXd fd, double tolerance=1e-3)
{
  MatrixXd error_mat = analytical - fd;
  std::vector<int> row = indexes[row_id];
  std::vector<int> col = indexes[col_id];
  if ((error_mat.block(row[0], col[0], row[1], col[1]).array().abs() > tolerance).any())
  {
    std::cout << FONT_FAIL << "Error in Jacobian " << row_id << ", " << col_id << "\n";
    std::cout << "BLOCK ERROR:\n" << error_mat.block(row[0], col[0], row[1], col[1]) << "\n";
    std::cout << "ANALYTICAL:\n" << analytical.block(row[0], col[0], row[1], col[1]) << "\n";
    std::cout << "FD:\n" << fd.block(row[0], col[0], row[1], col[1]) << ENDC << "\n";
    return true;
  }
  return false;
}

int check_all(MatrixXd analytical, MatrixXd fd, std::string name, double tol = 1e-3)
{
  MatrixXd error_mat = analytical - fd;
  if ((error_mat.array().abs() > tol).any())
  {
    std::cout << FONT_FAIL << "Error in total " << BOLD << name << ENDC << FONT_FAIL << " matrix" << ENDC << "\n";
    for (int row =0; row < error_mat.rows(); row ++)
    {
      for (int col = 0; col < error_mat.cols(); col++)
      {
        if(std::abs(error_mat(row, col)) > tol)
        {
          std::cout << BOLD << "error in (" << row << ", " << col << "):\tERR: " << error_mat(row,col) << "\tA: " << analytical(row, col) << "\tFD: " << fd(row,col) << ENDC << "\n";
        }
      }
    }
    return true;
  }
  return false;
}

VIEKF init_jacobians_test(xVector& x0, uVector& u0)
{
  // Configure initial State
  x0.setZero();
  x0(VIEKF::xATT) = 1.0;
  x0(VIEKF::xMU) = 0.2;
  x0.block<3,1>((int)VIEKF::xPOS, 0) += Vector3d::Random() * 100.0;
  x0.block<3,1>((int)VIEKF::xVEL, 0) += Vector3d::Random() * 10.0;
  x0.block<4,1>((int)VIEKF::xATT, 0) = (Quatd(x0.block<4,1>((int)VIEKF::xATT, 0)) + Vector3d::Random()).elements();
  x0.block<3,1>((int)VIEKF::xB_A, 0) += Vector3d::Random() * 1.0;
  x0.block<3,1>((int)VIEKF::xB_G, 0) += Vector3d::Random() * 0.5;
  x0((int)VIEKF::xMU, 0) += (static_cast <double> (rand()) / (static_cast <double> (RAND_MAX)))*0.05;
  
  // Create VIEKF
  VIEKF ekf;
  Matrix<double, vi_ekf::VIEKF::dxZ, 1> P0, Qx, gamma;
  P0.setOnes();
  Qx.setOnes();
  gamma.setOnes();
  uVector Qu;
  Qu.setOnes();
  Vector3d P0feat, Qxfeat, gammafeat;
  P0feat.setOnes();
  Qxfeat.setOnes();
  gammafeat.setOnes();
  Vector2d cam_center = Vector2d::Random();
  cam_center << 320-25+std::rand()%50, 240-25+std::rand()%50;
  Vector2d focal_len;
  focal_len << static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/100.0)),
      static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/100.0));
  Vector4d q_b_c = Quatd::Random().elements();
  Vector3d p_b_c = Vector3d::Random() * 0.5;
  Matrix<double, 17, 1> state_0;
  state_0 =x0.block<17, 1>(0,0); 
  ekf.init(state_0, P0, Qx, gamma, Qu, P0feat, Qxfeat, gammafeat, cam_center, focal_len, q_b_c, p_b_c, 2.0, "~", true, true, true, 0.0, 0);
  
  // Initialize Random Features
  for (int i = 0; i < NUM_FEATURES; i++)
  {
    Vector2d l;
    l << std::rand()%640, std::rand()%480;
    double depth = static_cast <double> (rand()) / (static_cast <double> (RAND_MAX/10.0));
    ekf.init_feature(l, i, depth);
  }
  // Recover the new state to return
  x0 = ekf.get_state();
  
  // Initialize Inputs
  u0.setZero();
  u0.block<3,1>((int)VIEKF::uA, 0) += Vector3d::Random() * 1.0;
  u0.block<3,1>((int)VIEKF::uG, 0) += Vector3d::Random() * 1.0;
  
  return ekf;
}

int htest(measurement_function_ptr fn, VIEKF& ekf, const VIEKF::measurement_type_t type, const int id, const int dim, double tol=1e-3)
{
  int num_errors = 0;
  xVector x0 = ekf.get_state();
  zVector z0;
  MatrixXd a_dhdx;
  a_dhdx.setZero(dim, MAX_DX);
  
  // Call the Measurement function
  hMatrix H;
  CALL_MEMBER_FN(ekf, fn)(x0, z0, H, id);
  a_dhdx = H.topRows(dim);

  
  MatrixXd d_dhdx;
  d_dhdx.setZero(dim, MAX_DX);
  
  Matrix<double, MAX_DX, MAX_DX> I = Matrix<double, MAX_DX, MAX_DX>::Identity();
  double epsilon = 1e-6;
  
  zVector z_prime;
  hMatrix dummy_H;
  xVector x_prime;
  for (int i = 0; i < a_dhdx.cols(); i++)
  {
    ekf.boxplus(ekf.get_state(), (I.col(i) * epsilon), x_prime);
    
    CALL_MEMBER_FN(ekf, fn)(x_prime, z_prime, dummy_H, id);
    
    if (type == VIEKF::QZETA)
      d_dhdx.col(i) = q_feat_boxminus(Quatd(z_prime), Quatd(z0))/epsilon;
    else if (type == VIEKF::ATT)
      d_dhdx.col(i) = (Quatd(z_prime) - Quatd(z0))/epsilon;
    else
      d_dhdx.col(i) = (z_prime.topRows(dim) - z0.topRows(dim))/epsilon;
  }
  
  MatrixXd error = a_dhdx - d_dhdx;
  double err_threshold = std::max(tol * a_dhdx.norm(), tol);
  
  for (std::map<std::string, std::vector<int>>::iterator it=indexes.begin(); it!=indexes.end(); ++it)
  {
    if(it->second[0] + it->second[1] > error.cols())
      continue;
    MatrixXd block_error = error.block(0, it->second[0], error.rows(), it->second[1]);    
    if ((block_error.array().abs() > err_threshold).any())
    {
      num_errors += 1;
      std::cout << FONT_FAIL << "Error in Measurement " << measurement_names[type] << "_" << id << ", " << it->first << ": (thresh = " << err_threshold << " mean = " << a_dhdx.norm() << ")\n";
      std::cout << "ERR:\n" << block_error << "\nA:\n" << a_dhdx.block(0, it->second[0], error.rows(), it->second[1]) << "\n";
      std::cout << "FD:\n" << d_dhdx.block(0, it->second[0], error.rows(), it->second[1]) << ENDC <<  "\n";
    }
  }
  return num_errors;
}

void XVECTOR_EQUAL(xVector& x1, xVector& x2)
{
  for (int i = 0; i < VIEKF::xATT; i++)
    EXPECT_NEAR(x1(i, 0), x2(i,0), 1e-8);
  
  QUATERNION_EQUALS(Quatd(x1.block<4,1>((int)VIEKF::xATT, 0)), Quatd(x2.block<4,1>((int)VIEKF::xATT, 0)));
  
  for (int i = VIEKF::xB_A; i < VIEKF::xZ; i++)
    EXPECT_NEAR(x1(i, 0), x2(i,0), 1e-8);
  
  for (int i = 0; i < NUM_FEATURES; i++)
  {
    VECTOR3_EQUALS(Quatd(x1.block<4,1>(VIEKF::xZ+i*5,0)).rota(e_z), Quatd(x2.block<4,1>(VIEKF::xZ+i*5,0)).rota(e_z));
    EXPECT_NEAR(x1(VIEKF::xZ+i*5+4), x1(VIEKF::xZ+i*5+4), 1e-8);
  }
}

void VIEKF_manifold()
{
  xVector x, x2, x3;
  uVector u;
  dxVector dx, dx1, dx2;
  for (int j = 0; j < NUM_ITERS; j++)
  {
    vi_ekf::VIEKF ekf = init_jacobians_test(x, u);
    vi_ekf::VIEKF dummyekf = init_jacobians_test(x2, u);
    dx.setZero();
    
    // (x [+] 0) == x 
    ekf.boxplus(x, dx, x3);
    MATRIX_EQUAL(x3, x, 1e-8);
    
    // (x [+] (x2 [-] x)) = x2
    ekf.boxminus(x2, x, dx2);
    ekf.boxplus(x, dx2, x3);
    XVECTOR_EQUAL(x3, x2);
    
    // ((x [+] dx) [-] x) == dx
    dx.setRandom();
    ekf.boxplus(x, dx, x3);
    ekf.boxminus(x3, x, dx2);
    MATRIX_EQUAL(dx2, dx, 1e-8);
    
    // ||(x [+] dx1) [-] (x [+] dx2)|| < || dx1 - dx2 ||
    dx1.setRandom();
    dx2.setRandom();
    ekf.boxplus(x, dx1, x2);
    ekf.boxplus(x, dx2, x3);
    ekf.boxminus(x2, x3, dx);
    ASSERT_LE(dx.norm(), (dx - dx2).norm());    
  }
}
TEST(VI_EKF, manifold){VIEKF_manifold();}

void VIEKF_dfdx_test()
{  
  xVector x0;
  uVector u0;
  dxMatrix dummydfdx;
  dxuMatrix dummydfdu;
  dxVector dxprime;
  xVector xprime;
  dxMatrix Idx = dxMatrix::Identity();
  double epsilon = 1e-6;
  
  dxMatrix d_dfdx;
  dxVector dx0;
  dxuMatrix a_dfdu;
  dxMatrix a_dfdx;
  for (int j = 0; j < NUM_ITERS; j++)
  {
    vi_ekf::VIEKF ekf = init_jacobians_test(x0, u0);
    
    // analytical differentiation
    ekf.dynamics(x0, u0, dx0, a_dfdx, a_dfdu);
    d_dfdx.setZero();
    
    // numeri
    for (int i = 0; i < d_dfdx.cols(); i++)
    {
      ekf.boxplus(x0, (Idx.col(i) * epsilon), xprime);
      ekf.dynamics(xprime, u0, dxprime, dummydfdx, dummydfdu);
      d_dfdx.col(i) = (dxprime - dx0) / epsilon;
    }
    
    ASSERT_FALSE(check_block("dxPOS", "dxVEL", a_dfdx, d_dfdx));
    ASSERT_FALSE(check_block("dxPOS", "dxATT", a_dfdx, d_dfdx));
    ASSERT_FALSE(check_block("dxVEL", "dxVEL", a_dfdx, d_dfdx));
    ASSERT_FALSE(check_block("dxVEL", "dxPOS", a_dfdx, d_dfdx));
    ASSERT_FALSE(check_block("dxVEL", "dxATT", a_dfdx, d_dfdx));
    ASSERT_FALSE(check_block("dxVEL", "dxB_A", a_dfdx, d_dfdx));
    ASSERT_FALSE(check_block("dxVEL", "dxB_G", a_dfdx, d_dfdx));
    ASSERT_FALSE(check_block("dxVEL", "dxMU", a_dfdx, d_dfdx));
    
    for (int i = 0; i < ekf.get_len_features(); i++)
    {
      std::string zeta_key = "dxZETA_" + std::to_string(i);
      std::string rho_key = "dxRHO_" + std::to_string(i);
      
      ASSERT_FALSE(check_block(zeta_key, "dxVEL", a_dfdx, d_dfdx));
      ASSERT_FALSE(check_block(zeta_key, "dxB_G", a_dfdx, d_dfdx));
      ASSERT_FALSE(check_block(zeta_key, zeta_key, a_dfdx, d_dfdx));
      ASSERT_FALSE(check_block(zeta_key, rho_key, a_dfdx, d_dfdx));
      ASSERT_FALSE(check_block(rho_key, "dxVEL", a_dfdx, d_dfdx));
      ASSERT_FALSE(check_block(rho_key, "dxB_G", a_dfdx, d_dfdx));
      ASSERT_FALSE(check_block(rho_key, zeta_key, a_dfdx, d_dfdx, 1e-2));
      ASSERT_FALSE(check_block(rho_key, rho_key, a_dfdx, d_dfdx));
    }
  }
}
TEST(VI_EKF, dfdx_test){VIEKF_dfdx_test();}

void VIEKF_dfdu_test()
{
  // Variation w.r.t. accel/gyro noise is the same as w.r.t. their biases, so just perturb those
  xVector x0;
  uVector u0;
  dxVector dxprime;
  xVector xprime;
  double epsilon = 1e-6;
  dxMatrix dfdx_dummy;
  dxuMatrix dfdu_dummy;
  dxuMatrix d_dfdu;
  Matrix<double, MAX_DX, 6> Iu;
  Iu.setZero();
  Iu.block<6,6>(VIEKF::dxB_A,0).setIdentity();
  dxVector dx0;
  dxMatrix a_dfdx;
  dxuMatrix a_dfdu;
  for (int j = 0; j < NUM_ITERS; j++)
  {
    vi_ekf::VIEKF ekf = init_jacobians_test(x0, u0);
    
    // Perform Analytical Differentiation
    ekf.dynamics(x0, u0, dx0, a_dfdx, a_dfdu);
    d_dfdu.setZero();
    
    // Perform Numerical Differentiation
    for (int i = 0; i < d_dfdu.cols(); i++)
    {
      ekf.boxplus(x0, (Iu.col(i) * epsilon), xprime);
      ekf.dynamics(xprime, u0, dxprime, dfdx_dummy, dfdu_dummy);
      d_dfdu.col(i) = (dxprime - dx0) / epsilon;
    }
    
    ASSERT_FALSE(check_block("dxVEL","uA", a_dfdu, d_dfdu));
    ASSERT_FALSE(check_block("dxVEL","uG", a_dfdu, d_dfdu));
    ASSERT_FALSE(check_block("dxATT", "uG", a_dfdu, d_dfdu));
    for (int i = 0; i < ekf.get_len_features(); i++)
    {
      std::string zeta_key = "dxZETA_" + std::to_string(i);
      std::string rho_key = "dxRHO_" + std::to_string(i);
      ASSERT_FALSE(check_block(zeta_key, "uG", a_dfdu, d_dfdu));
      ASSERT_FALSE(check_block(rho_key, "uG", a_dfdu, d_dfdu, 1e-1));
    }
  }
}
TEST(VI_EKF, dfdu_test){VIEKF_dfdu_test();}

void VI_EKF_h_test()
{
  xVector x0;
  uVector u0;
  for (int j = 0; j < NUM_ITERS; j++)
  {
    vi_ekf::VIEKF ekf = init_jacobians_test(x0, u0);
    
    ASSERT_FALSE(htest(&VIEKF::h_acc, ekf, VIEKF::ACC, 0, 2));
    ASSERT_FALSE(htest(&VIEKF::h_pos, ekf, VIEKF::POS, 0, 3));
    ASSERT_FALSE(htest(&VIEKF::h_vel, ekf, VIEKF::VEL, 0, 3));
    ASSERT_FALSE(htest(&VIEKF::h_alt, ekf, VIEKF::ALT, 0, 1));
    ekf.set_drag_term(true);
    ASSERT_FALSE(htest(&VIEKF::h_att, ekf, VIEKF::ATT, 0, 3));
    ekf.set_drag_term(false);
    ASSERT_FALSE(htest(&VIEKF::h_att, ekf, VIEKF::ATT, 0, 3));
    for (int i = 0; i < ekf.get_len_features(); i++)
    {
      EXPECT_FALSE(htest(&VIEKF::h_feat, ekf, VIEKF::FEAT, i, 2, 1e-1));
      EXPECT_FALSE(htest(&VIEKF::h_qzeta, ekf, VIEKF::QZETA, i, 2));
      EXPECT_FALSE(htest(&VIEKF::h_depth, ekf, VIEKF::DEPTH, i, 1));
      EXPECT_FALSE(htest(&VIEKF::h_inv_depth, ekf, VIEKF::INV_DEPTH, i, 1));
      //        ASSERT_EQ(htest(VIEKF::h_pixel_vel, ekf, VIEKF::PIXEL_VEL, i), 0); // Still needs to be implemented
    }
  }
}
TEST(VI_EKF, h_test) {VI_EKF_h_test();}


void VIEKF_KF_reset_test()
{
  uVector u0;
  dxMatrix d_dxpdxm;
  dxMatrix a_dxpdxm;
  xVector xm;
  xVector xp;
  dxMatrix dummy;
  dxMatrix I_dx = dxMatrix::Identity();
  xVector xm_prime;    
  xVector xp_prime;    
  dxVector d_xp;
  
  for (int j = 0; j < NUM_ITERS; j++)
  {
    vi_ekf::VIEKF ekf = init_jacobians_test(xm, u0);
    ekf.keyframe_reset(xm, xp, a_dxpdxm);
    Quatd qm(xm.block<4,1>((int)VIEKF::xATT, 0));
    Quatd qp(xp.block<4,1>((int)VIEKF::xATT, 0));
    
    ASSERT_NEAR(qm.roll(), qp.roll(), 1e-8);
    ASSERT_NEAR(qm.pitch(), qp.pitch(), 1e-8);
    ASSERT_NEAR(qp.yaw(), 0.0, 1e-8);    
    
    d_dxpdxm.setZero();
    double epsilon = 1e-6;
    
    // Perform Numerical Differentiation
    for (int i = 0; i < d_dxpdxm.cols(); i++)
    {
      ekf.boxplus(xm, (I_dx.col(i) * epsilon), xm_prime);
      ekf.keyframe_reset(xm_prime, xp_prime, dummy);
      ekf.boxminus(xp_prime, xp, d_xp);
      d_dxpdxm.col(i) =  d_xp / epsilon;
    }
    
    ASSERT_FALSE(check_block("dxPOS", "dxPOS", a_dxpdxm, d_dxpdxm));
    ASSERT_FALSE(check_block("dxATT", "dxATT", a_dxpdxm, d_dxpdxm, 1e-1));
  }
  ASSERT_FALSE(check_all(a_dxpdxm, d_dxpdxm, "dfdx", 1e-1));
}
TEST(VI_EKF, KF_reset_test){VIEKF_KF_reset_test();}

int main(int argc, char **argv) {
  srand(std::chrono::system_clock::now().time_since_epoch().count());
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


