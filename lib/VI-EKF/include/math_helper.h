#pragma once

#include "math.h"
#include "geometry/quat.h"
#include "geometry/support.h"

#include <random>

#include <Eigen/Core>

void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove);
void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove);

inline Eigen::Vector3d zeta(quat::Quatd q)
{
  return q.rota(e_z);
}

inline Eigen::Matrix<double, 3, 2> T_zeta(quat::Quatd q)
{
  return q.doublerota(I_2x3.transpose());
}

// q1 - q0
inline Eigen::Vector2d q_feat_boxminus(quat::Quatd q1, quat::Quatd q0)
{
  Eigen::Vector3d zeta0 = zeta(q0);
  Eigen::Vector3d zeta1 = zeta(q1);

  Eigen::Vector2d dq;
  if ((zeta0 - zeta1).norm() > 1e-8)
  {
    Eigen::Vector3d v = zeta1.cross(zeta0);
    v /= v.norm();
    double theta = std::acos(zeta1.dot(zeta0));
    dq = T_zeta(q0).transpose() * (theta * v);
  }
  else
  {
    dq.setZero();
  }
  return dq;
}

inline quat::Quatd q_feat_boxplus(quat::Quatd q, Eigen::Vector2d dq)
{
  return quat::Quatd::exp(-T_zeta(q) * dq) * q;
}

void concatenate_SE2(Eigen::Vector3d& T1, Eigen::Vector3d& T2, Eigen::Vector3d& Tout);
void concatenate_edges(const Eigen::Matrix<double,7,1>& T1, const Eigen::Matrix<double,7,1>& T2, Eigen::Matrix<double,7,1>& Tout);
const Eigen::Matrix<double,7,1> invert_edge(const Eigen::Matrix<double,7,1>& T1);
void invert_SE2(Eigen::Vector3d& T, Eigen::Vector3d& Tout);

inline double random(double max, double min)
{
  double f = (double)rand() / RAND_MAX;
  return min + f * (max - min);
}

// Gamma is the derivative of the exponential map, its inverse is the logarithmic map's derivative
template<typename T>
Matrix<T,3,3> Gamma(const Matrix<T,3,1> &delta)
{
  T delta_mag = delta.norm();
  Matrix<T,3,3> skew_delta = skew(delta);
  if (delta_mag > 1e-6)
    return I_3x3 - (1.0 - cos(delta_mag)) / (delta_mag * delta_mag) * skew_delta +
           (delta_mag - sin(delta_mag)) / (delta_mag * delta_mag *delta_mag) * skew_delta * skew_delta;
  else
    return I_3x3 - 0.5 * skew_delta;
}
