#include <iostream>

#include "gtest/gtest.h"
#include "xform.h"

using namespace quat;
using namespace xform;
using namespace Eigen;
using namespace std;

#define NUM_ITERS 10

#define MATRIX_CLOSE(m1, m2, tol) {\
  for (int row = 0; row < (m1).rows(); row++ ) \
{ \
  for (int col = 0; col < (m1).cols(); col++) \
{ \
  EXPECT_NEAR((m1)(row, col), (m2)(row, col), tol); \
  } \
  } \
  }

#define QUATERNION_EQUALS(q1, q2) \
  MATRIX_CLOSE(q1.arr_,  (sign(q2.w()) * sign(q1.w())) * q2.arr_, 1e-8)

#define MATRIX_EQUALS(v1, v2) \
  MATRIX_CLOSE(v1, v2, 1e-8)

#define TRANSFORM_EQUALS(t1, t2) \
  MATRIX_EQUALS((t1).t(), (t2).t()); \
  QUATERNION_EQUALS((t1).q(), (t2).q())

#define TRANSFORM_CLOSE(t1, t2, tol) \
  MATRIX_CLOSE(t1.q_.arr_, (sign(t2.q_.w()) * sign(t1.q_.w())) * t2.q_.arr_, tol); \
  MATRIX_CLOSE(t1.t_, t1.t_, tol)


#define CALL_MEMBER_FN(objectptr,ptrToMember) ((objectptr).*(ptrToMember))
#define HEADER "\033[95m"
#define OKBLUE "\033[94m"
#define OKGREEN "\033[92m"
#define WARNING "\033[93m"
#define FONT_FAIL "\033[91m"
#define ENDC "\033[0m"
#define BOLD "\033[1m"
#define UNDERLINE "\033[4m"

void known_transform()
{
  Xform T_known((Vector3d() << 1, 1, 0).finished(),
                Quat::from_axis_angle((Vector3d() << 0, 0, 1).finished(), M_PI/4.0));
  Xform T_known_inv((Vector3d() << -sqrt(2), 0, 0).finished(),
                    Quat::from_axis_angle((Vector3d() << 0, 0, 1).finished(), -M_PI/4.0));
  TRANSFORM_EQUALS(T_known.inverse(), T_known_inv);
}
TEST(xform, known_transform){known_transform();}

void known_vector_passive_transform()
{
  Vector3d p1; p1 << 1, 0, 0;
  Xform T_known((Vector3d() << 1, 1, 0).finished(),
                Quat::from_axis_angle((Vector3d() << 0, 0, 1).finished(), M_PI/4.0));
  Vector3d p2; p2 << -sqrt(0.5), -sqrt(0.5), 0;
  MATRIX_EQUALS(p2, T_known.transformp(p1));
}
TEST(xform, known_vector_passive_transform){known_vector_passive_transform();}

void known_vector_active_transform()
{
  Vector3d p1; p1 << 1, 0, 0;
  Xform T_known((Vector3d() << 1, 1, 0).finished(),
                Quat::from_axis_angle((Vector3d() << 0, 0, 1).finished(), M_PI/4.0));
  Vector3d p2; p2 << 1+sqrt(0.5), 1+sqrt(0.5), 0;
  MATRIX_EQUALS(p2, T_known.transforma(p1));
}
TEST(xform, known_vector_active_transform){known_vector_active_transform();}

void inverse()
{
  for (int i = 0; i < NUM_ITERS; i++)
  {
    Xform T1 = Xform::Random();
    Xform T2 = T1.inverse();
    Xform T3 = T1 * T2;
    QUATERNION_EQUALS(T3.q(), Quat::Identity());
    EXPECT_NEAR(T3.t().norm(), 0, 1e-8);
  }
}
TEST(xform, inverse){inverse();}

void transform_vector()
{
  for (int i = 0; i < NUM_ITERS; i++)
  {
    Xform T1 = Xform::Random();
    Vector3d p;
    p.setRandom();
    MATRIX_EQUALS(T1.transformp(T1.inverse().transformp(p)), p);
    MATRIX_EQUALS(T1.inverse().transformp(T1.transformp(p)), p);
    MATRIX_EQUALS(T1.transforma(T1.inverse().transforma(p)), p);
    MATRIX_EQUALS(T1.inverse().transforma(T1.transforma(p)), p);
  }
}
TEST(xform, transform_vector){transform_vector();}

void exp()
{
  Vector3d v;
  Vector3d omega;
  for (int i = 0; i < NUM_ITERS; i++)
  {
    v.setRandom();
    omega.setRandom();

    Xform x = Xform::Identity();
    double dt = 0.0001;
    for (double t = 0; t < 1.0; t += dt)
    {
      x.t_ += x.q_.rotp(v * dt);
      x.q_ += omega*dt;
    }

    Vector6d delta; delta << v, omega;
    Xform xexp = Xform::exp(delta);
    TRANSFORM_CLOSE(x, xexp, 1e-4);
  }
}
TEST(xform, exp){exp();}

void log_exp()
{
  for (int i = 0; i < NUM_ITERS; i++)
  {
    Vector6d xi;
    xi.setRandom();
    MATRIX_EQUALS(Xform::log(Xform::exp(xi)), xi);
  }
}
TEST(xform, log_exp){log_exp();}

void boxplus_boxminus()
{
  for (int i = 0; i < NUM_ITERS; i++)
  {
    Xform T = Xform::Random();
    Xform T2 = Xform::Random();
    Vector6d zeros, dT;
    zeros.setZero();
    dT.setRandom();
    TRANSFORM_EQUALS(T + zeros, T);
    TRANSFORM_EQUALS(T + (T2 - T), T2);
    MATRIX_EQUALS((T + dT) - T, dT);
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


