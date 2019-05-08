#include "gtest/gtest.h"

#include <eigen3/Eigen/Geometry>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

#include "quat.h"
#include "math_helper.h"


using namespace quat;
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

void quat_rotation_direction()
{
  // Compare against a known active and passive rotation
  Vector3d v, beta, v_active_rotated, v_passive_rotated;
  v << 0, 0, 1;
  v_active_rotated << 0, -1.0*std::pow(0.5,0.5), std::pow(0.5,0.5);
  beta << 1, 0, 0;
  Quat q_x_45 = Quat::from_axis_angle(beta, 45.0*M_PI/180.0);
  Eigen::Vector3d v_x_45 = q_x_45.rota(v);
  
  Matrix3d R_true;
  R_true << 1.0000000,  0.0000000,  0.0000000,
      0.0000000,  0.70710678118654757,  0.70710678118654757,
      0.0000000,  -0.70710678118654757, 0.70710678118654757;
  Matrix3d qR = q_x_45.R();
  MATRIX_EQUAL(qR, R_true, 1e-6);
  VECTOR3_EQUALS(qR.transpose() * v, v_active_rotated);
  VECTOR3_EQUALS(R_true.transpose() * v, v_active_rotated);
  
  VECTOR3_EQUALS(v_x_45, v_active_rotated);
  
  v_passive_rotated << 0, std::pow(0.5, 0.5), std::pow(0.5, 0.5);
  Vector3d v_x_45_T = q_x_45.rotp(v);
  VECTOR3_EQUALS(v_x_45_T, v_passive_rotated);
  VECTOR3_EQUALS(qR * v, v_passive_rotated);
  VECTOR3_EQUALS(R_true * v, v_passive_rotated);
}
TEST(Quat, quat_rotation_direction) { quat_rotation_direction(); }

void quat_rot_invrot_R()
{
  Vector3d v;
  Quat q1 = Quat::Random();
  for (int i = 0; i < NUM_ITERS; i++)
  {
    v.setRandom();
    q1 = Quat::Random();
    
    // Check that rotations are inverses of each other
    VECTOR3_EQUALS(q1.rota(v), q1.R().transpose() * v);
    VECTOR3_EQUALS(q1.rotp(v), q1.R() * v);
  }
}
TEST(Quat, quat_rot_invrot_R){quat_rot_invrot_R();}

void quat_from_two_unit_vectors()
{
  Vector3d v1, v2;
  for (int i = 0; i < NUM_ITERS; i++)
  {
    v1.setRandom();
    v2.setRandom();
    v1 /= v1.norm();
    v2 /= v2.norm();
    
    VECTOR3_EQUALS(Quat::from_two_unit_vectors(v1, v2).rota(v1), v2);
    VECTOR3_EQUALS(Quat::from_two_unit_vectors(v2, v1).rotp(v1), v2);
  }
}
TEST(Quat, quat_from_two_unit_vectors){quat_from_two_unit_vectors();}

void quat_from_R()
{
  Vector3d v;
  for (int i = 0; i < NUM_ITERS; i++)
  {
    Quat q1 = Quat::Random();
    Matrix3d R = q1.R();
    Quat qR = Quat::from_R(R);
    v.setRandom();
    VECTOR3_EQUALS(qR.rota(v), R.transpose() * v);
    VECTOR3_EQUALS(q1.rota(v), R.transpose() * v);
    VECTOR3_EQUALS(qR.rotp(v), R * v);
    VECTOR3_EQUALS(q1.rotp(v), R * v);
    MATRIX_EQUAL(R, qR.R(), 1e-6);
    QUATERNION_EQUALS(qR, q1);
  }
}
TEST(Quat, from_R){quat_from_R();}

void quat_otimes()
{
  Quat q1 = Quat::Random();
  Quat qI = Quat::Identity();
  QUATERNION_EQUALS(q1 * q1.inverse(), qI);
}
TEST(Quat, otimes){quat_otimes();}

TEST(Quat, exp_log_axis_angle)
{
  // Check that qexp is right by comparing with matrix exp and axis-angle
  for (int i = 0; i < NUM_ITERS; i++)
  {
    Vector3d omega;
    omega.setRandom();
    Matrix3d R_omega_exp_T = Quat::skew(omega).exp();  // Why is this needed?
    Quat q_R_omega_exp = Quat::from_R(R_omega_exp_T.transpose());
    Quat q_omega = Quat::from_axis_angle(omega/omega.norm(), omega.norm());
    Quat q_omega_exp = Quat::exp(omega);
    QUATERNION_EQUALS(q_R_omega_exp, q_omega);
    QUATERNION_EQUALS(q_omega_exp, q_omega);
    
    // Check that exp and log are inverses of each otherprint_error
    VECTOR3_EQUALS(Quat::log(Quat::exp(omega)), omega);
    QUATERNION_EQUALS(Quat::exp(Quat::log(q_omega)), q_omega);
  }
}


TEST(Quat, boxplus_and_boxminus)
{
  Vector3d delta1, delta2, zeros;
  zeros.setZero();
  for (int i = 0; i < NUM_ITERS; i++)
  {
    Quat q = Quat::Random();
    Quat q2 = Quat::Random();
    delta1.setRandom();
    delta2.setRandom();
    
    QUATERNION_EQUALS(q + zeros, q);
    QUATERNION_EQUALS(q + (q2 - q), q2);
    VECTOR3_EQUALS((q + delta1) - q, delta1);
    ASSERT_LE(((q+delta1)-(q+delta2)).norm(), (delta1-delta2).norm());
  }
}

void quat_inplace_add_and_mul()
{
  Vector3d delta1, delta2, zeros;
  zeros.setZero();
  for (int i = 0; i < NUM_ITERS; i++)
  {
    Quat q = Quat::Random();
    Quat q2 = Quat::Random();
    Quat q_copy = q.copy();
    QUATERNION_EQUALS(q_copy, q);
    delta1.setRandom();
    delta2.setRandom();
    
    q_copy += delta1;
    QUATERNION_EQUALS(q_copy, q+delta1);
    
    q_copy = q.copy();
    q_copy *= q2;
    QUATERNION_EQUALS(q_copy, q*q2);
  }
}
TEST(Quat, inplace_add_and_mul){quat_inplace_add_and_mul();}

void quat_euler()
{
  for (int i =0; i < NUM_ITERS; i++)
  {
    double roll = random(-M_PI, M_PI);
    double pitch = random(-M_PI/2.0, M_PI/2.0);
    double yaw = random(-M_PI, M_PI);
    Quat q = Quat::from_euler(roll, pitch, yaw);
    ASSERT_NEAR(roll, q.roll(), 1e-8);
    ASSERT_NEAR(pitch, q.pitch(), 1e-8);
    ASSERT_NEAR(yaw, q.yaw(), 1e-8);    
    Quat q2 = Quat::from_euler(q.roll(), q.pitch(), q.yaw());
    QUATERNION_EQUALS(q, q2);
  }
}
TEST(Quat, euler){quat_euler();}

void quat_passive_derivative()
{
  Quat q0 = Quat::Random();
  Vector3d v;
  v.setRandom();
  
  Matrix3d a;
  Matrix3d fd;
  Matrix3d I = Matrix3d::Identity();
  double epsilon = 1e-8;
  
  a = skew(q0.rotp(v));  // [R(q)v]
  
  for (int i = 0; i < 3; i++)
  {
    Quat qi = q0 + (epsilon * (I.col(i)));
    fd.col(i) = (qi.rotp(v) - q0.rotp(v))/epsilon;
  }
  if ((fd - a).array().abs().sum() > 1e-6)
  {
    std::cout << "ERROR IN LINE " << __LINE__ << "\nA:\n" << a << "\nD:\nfd" << fd << std::endl;
  }
  ASSERT_LE((fd - a).array().abs().sum(), 1e-6);
}
TEST(Quat, passive_rotation_derivative){quat_passive_derivative();}

void quat_active_derivative()
{
  Quat q0 = Quat::Random();
  Vector3d v;
  v.setRandom();
  
  Matrix3d a;
  Matrix3d fd;
  Matrix3d I = Matrix3d::Identity();
  double epsilon = 1e-8;
  
  a = -q0.R().transpose() * skew(v);  // -R(q).T * [v]
  
  for (int i = 0; i < 3; i++)
  {
    Quat qi = q0 + (epsilon * (I.col(i)));
    fd.col(i) = (qi.rota(v) - q0.rota(v))/epsilon;
  }
  if ((fd - a).array().abs().sum() > 1e-6)
  {
    std::cout << "ERROR IN LINE " << __LINE__ << "\nA:\n" << a << "\nD:\nfd" << fd << std::endl;
  }
  ASSERT_LE((fd - a).array().abs().sum(), 1e-6);
}
TEST(Quat, active_rotation_derivative){quat_active_derivative();}


void exp_approx()
{
  for (int j = 0; j < NUM_ITERS; j++)
  {
    Quat q = Quat::Random();
    if (j == 0)
      q = Quat::Identity();
    Vector3d delta;
    Matrix3d I = Matrix3d::Identity();
    delta.setRandom();
    delta *= 0.1;
    
    Quat qp = q + delta;
    
    Matrix3d actual = qp.R();
    Matrix3d approx = (I - skew(delta))*q.R();
    ASSERT_LE((actual - approx).array().abs().sum(), 0.1);
  }
}
TEST(Quat, exp_approx){exp_approx();}


void math_helper_T_zeta()
{
  Vector3d v2;
  for (int i = 0; i < NUM_ITERS; i++)
  {
    v2.setRandom();
    v2 /= v2.norm();
    Quat q2 = Quat::from_two_unit_vectors(e_z, v2);
    Vector2d T_z_v2 = T_zeta(q2).transpose() * v2;
    ASSERT_LE(T_z_v2.norm(), 1e-8);
  }
}
TEST(math_helper, T_zeta){math_helper_T_zeta();}

void math_helper_d_dTdq()
{
  double epsilon = 1e-6;
  for (int j = 0; j < NUM_ITERS; j++)
  {
    Vector3d v;
    v.setRandom();
    Quat q = Quat::Random();
    q.setZ(0);
    q.normalize();
    auto T_z = T_zeta(q);
    Vector2d x0 = T_z.transpose() * v;
    Matrix2d I = Matrix2d::Identity();
    Matrix2d a_dTdq = -T_z.transpose() * skew(v) * T_z;
    Matrix2d d_dTdq;
    d_dTdq.setZero();
    for (int i = 0; i < 2; i++)
    {
      Quat qplus = q_feat_boxplus(q, I.col(i) * epsilon);
      Vector2d xprime = T_zeta(qplus).transpose() * v;
      d_dTdq.col(i) = (xprime - x0) / epsilon;
    }
    MATRIX_EQUAL(d_dTdq, a_dTdq, 1e-6);
  }
}
TEST(math_helper, d_dTdq){math_helper_d_dTdq();}

void math_helper_dzeta_dqzeta()
{
  double epsilon = 1e-6;
  for(int j = 0; j < NUM_ITERS; j++)
  {
    Matrix<double, 3, 2> d_dzeta_dqzeta;
    Quat q = Quat::Random();
    q.setZ(0);
    q.normalize();
    Matrix2d I = Matrix2d::Identity() * epsilon;
    for (int i = 0; i < 2; i++)
    {
      quat::Quat q_prime = q_feat_boxplus(q, I.col(i));
      Vector3d dzeta  = zeta(q_prime) - zeta(q);
      d_dzeta_dqzeta.col(i) = dzeta / epsilon;
    }
    Matrix<double, 3, 2> a_dzeta_dqzeta = skew(zeta(q)) * T_zeta(q);
    MATRIX_EQUAL(a_dzeta_dqzeta, d_dzeta_dqzeta, 1e-6);
  }
}
TEST(math_helper, dzeta_dqzeta){math_helper_dzeta_dqzeta();}

void math_helper_dqzeta_dqzeta()
{
  for(int j = 0; j < NUM_ITERS; j++)
  {
    Matrix2d d_dqdq;
    quat::Quat q = quat::Quat::Random();
    if (j == 0)
      q = quat::Quat::Identity();
    double epsilon = 1e-6;
    Matrix2d I = Matrix2d::Identity() * epsilon;
    for (int i = 0; i < 2; i++)
    {
      quat::Quat q_prime = q_feat_boxplus(q, I.col(i));
      Vector2d dq  = q_feat_boxminus(q_prime, q);
      d_dqdq.row(i) = dq /epsilon;
    }
    Matrix2d a_dqdq = T_zeta(q).transpose() * T_zeta(q);
    MATRIX_EQUAL(a_dqdq, d_dqdq, 1e-2);
  }
}
TEST(math_helper, dqzeta_dqzeta){math_helper_dqzeta_dqzeta();}

void math_helper_manifold_operations()
{
  Vector2d delta;
  Vector2d zeros;
  zeros.setZero();
  for (int i = 0; i < NUM_ITERS; i++)
  {
    delta.setRandom();
    Quat q1 = Quat::Random();
    q1.setZ(0);
    q1.normalize();
    Quat q2 = q_feat_boxplus(q1, delta);
    
    // (q1 [+] 0) == q1
    QUATERNION_EQUALS( q_feat_boxplus(q1, zeros), q1);
    // ((q1 [+] delta) [-] q1) == delta
    VECTOR2_EQUALS( q_feat_boxminus(q2, q1), delta);
    // (q1 [+] (q2 [-] q1)) = q2
    VECTOR3_EQUALS( zeta(q_feat_boxplus( q1, q_feat_boxminus(q2, q1))), zeta(q2));
  }
}
TEST(math_helper, manifold_operations){math_helper_manifold_operations();}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


