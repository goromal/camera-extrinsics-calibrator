#include "vi_ekf.h"

namespace vi_ekf
{

void VIEKF::keyframe_reset(const xVector &xm, xVector &xp, dxMatrix &N)
{
  x_[i_] = xm;
  keyframe_reset();
  xp = x_[i_];
  N = A_;    
}

Xformd VIEKF::get_global_pose() const
{
  // Log Global Position Estimate
  Xformd global_pose;
  Xformd rel_pose(x_[i_].block<3,1>((int)xPOS, 0), Quatd(x_[i_].block<4,1>((int)xATT, 0)));
  global_pose = current_node_global_pose_ * rel_pose;
  return global_pose;
}

Matrix6d VIEKF::get_global_cov() const
{
  Matrix6d cov;
  edge_t rel_pose;
  rel_pose.transform.t() = x_[i_].block<3,1>((int)xPOS, 0);
  rel_pose.transform.q() = Quatd(x_[i_].block<4,1>((int)xATT, 0));
  rel_pose.cov.block<3,3>(0, 0) = P_[i_].block<3,3>(xPOS, xPOS);
  rel_pose.cov.block<3,3>(3, 0) = P_[i_].block<3,3>(xATT, xPOS);
  rel_pose.cov.block<3,3>(0, 3) = P_[i_].block<3,3>(xPOS, xATT);
  rel_pose.cov.block<3,3>(3, 3) = P_[i_].block<3,3>(xATT, xATT);
  propagate_global_covariance(global_pose_cov_, rel_pose, cov);
  return cov;
}

Matrix3d brackets(const Matrix3d& A)
{
  return -A.trace()*I_3x3 + A;
}

Matrix3d brackets(const Matrix3d& A, const Matrix3d& B)
{
  return brackets(A)*brackets(B) + brackets(A*B);
}

void VIEKF::propagate_global_covariance(const Matrix6d &P_prev, const edge_t &edge, Matrix6d &P_new) const
{
  Matrix6d Adj = current_node_global_pose_.Adj();
  P_new = P_prev + Adj.transpose() * edge.cov * Adj;

  /// TODO - look at Barfoot's way of propagating uncertainty
}


void VIEKF::keyframe_reset()
{
  // Save off current position into the new edge
  edge_t edge;
  edge.cov.setZero();
  edge.transform.t() = x_[i_].segment<3>(xPOS);
  edge.cov.block<3,3>(0,0) = P_[i_].block<3,3>(xPOS,xPOS);
  
  // Reset global position
  x_[i_].segment<3>(xPOS).setZero();

  // Get quaternion from state
  Quatd q_i2b(x_[i_].segment<4>(xATT));
  
  /// James' way to reset z-axis rotation

//  // precalculate some things
//  Vector3d v = q_i2b.rota(khat);
//  Vector3d s = khat.cross(v); s /= s.norm(); // Axis of rotation (without rotation about khat)
//  double theta = acos(khat.transpose() * v); // Angle of rotation
//  Matrix3d sk_tv = skew(theta*s);
//  Matrix3d sk_u = skew(khat);
//  Matrix3d R_i2b = q_i2b.R();
//  Quat qp = Quat::exp(theta * s); // q+

//  // Save off quaternion and covariance
//  edge.transform.q_ = q_i2b * qp.inverse();
//  edge.cov(5,5) = P_(xATT+2, xATT+2);
  
//  // Reset rotation about z
//  x_[i_].segment<4>(xATT) = qp.elements();
  
//  // Adjust covariance  (use A for N, because it is the right size and there is no need to allocate another one)
//  A_ = I_big_;
//  A_.block<3,3>(dxPOS,dxPOS).setZero();
//  A_.block<3,3>(dxATT,dxATT) = (I_3x3 + ((1.-cos(theta))*sk_tv)/(theta*theta) + ((theta - sin(theta))*sk_tv*sk_tv)/(theta*theta*theta)).transpose()
//          * (-s * (khat.transpose() * R_i2b.transpose() * sk_u) - theta * sk_u * (R_i2b.transpose() * sk_u));

  /// Jerel's way to reset z-axis rotation

//  // Pre-calculations
//  Eigen::Matrix3d Ixy = I_2x3.transpose() * I_2x3;
//  Eigen::Matrix3d Iz = I_3x3 - Ixy;
//  Eigen::Vector3d logq = Quat::log(q_i2b);
//  Eigen::Vector3d Ixylogq = Ixy * logq;
//  Eigen::Vector3d Izlogq = Iz * logq;
//  Eigen::Matrix3d dqz_dq = Gamma(Izlogq) * Iz * Gamma(logq).inverse();

//  // Save off quaternion and covariance
//  // Covariance is a first order approximation
//  edge.transform.q_ = Quat::exp(Izlogq); // Quaternion associated with z-axis rotation
//  edge.cov.block<3,3>(3,3) = dqz_dq * P_.block<3,3>(dxATT,dxATT) * dqz_dq.transpose();

//  // Reset rotation about z
//  x_[i_].segment<4>(xATT) = Quat::exp(Ixylogq).elements();

//  // Adjust covariance  (use A for N, because it is the right size and there is no need to allocate another one)
//  A_ = I_big_;
//  A_.block<3,3>(dxPOS,dxPOS).setZero(); // No altimeter for now, so z also gets reset
//  A_.block<3,3>(dxATT,dxATT) = Gamma(Ixylogq) * Ixy * Gamma(logq).inverse();
  
  
  /// Dan's way to reset z-axis rotation

  double yaw = q_i2b.yaw();
  double roll = q_i2b.roll();
  double pitch = q_i2b.pitch();
  
  // Save off quaternion and covariance
  edge.transform.q_ = Quatd::from_euler(0, 0, yaw);
  edge.cov(5,5) = P_[i_](xATT+2, xATT+2); /// TODO - this is wrong, need to compute covariance of yaw rotation
  
  // Reset attitude
  x_[i_].segment<4>(xATT) = Quatd::from_euler(roll, pitch, 0.0).elements();
  
  // Adjust covariance  (use A for N, because it is the right size and there is no need to allocate another one)
  // RMEKF paper after Eq. 81
  
  double cp = std::cos(roll);
  double sp = std::sin(roll);
  double tt = std::tan(pitch);
  A_ = I_big_;
  A_.block<3,3>(dxPOS,dxPOS).setZero();
  A_.block<3,3>(dxATT, dxATT) << 1, sp*tt, cp*tt,
      0, cp*cp, -cp*sp,
      0, -cp*sp, sp*sp;
  
  NAN_CHECK;
  
  // Remove uncertainty associated with unobservable states
  P_[i_] = A_ * P_[i_] * A_.transpose();
  
  // Propagate global covariance and update global node frame
  propagate_global_covariance(global_pose_cov_, edge, global_pose_cov_);
  current_node_global_pose_ = current_node_global_pose_ * edge.transform;
  
  NAN_CHECK;
  
  // call callback
  if (keyframe_reset_callback_ != nullptr)
    keyframe_reset_callback_();
}

}
