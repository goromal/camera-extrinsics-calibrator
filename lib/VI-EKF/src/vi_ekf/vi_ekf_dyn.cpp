#include "vi_ekf.h"

namespace vi_ekf
{
void VIEKF::dynamics(const xVector &x, const uVector &u, dxVector &xdot, dxMatrix &dfdx, dxuMatrix &dfdu)
{
  dynamics(x, u);
  xdot = dx_;
  dfdx = A_;
  dfdu = G_;
}


void VIEKF::dynamics(const xVector& x, const uVector &u, bool state, bool jac)
{
  if (state)
  {
    dx_.setZero();
  }
  
  if (jac)
  {
    A_.setZero();
    G_.setZero();
  }
  
  Vector3d vel = x.block<3, 1>((int)xVEL, 0);
  Quatd q_I_b(x.block<4,1>((int)xATT,0));
  
  Vector3d acc = u.block<3,1>((int)uA, 0) - x.block<3,1>((int)xB_A, 0);
  Vector3d omega = u.block<3,1>((int)uG, 0) - x.block<3,1>((int)xB_G, 0);
  Vector3d acc_z;
  acc_z << 0, 0, acc(2,0);
  double mu = x((int)xMU);
  
  Matrix3d R_I_b = q_I_b.R();
  Vector3d gravity_B = q_I_b.rotp(gravity); // R_I^b * vel
  Vector3d vel_xy;  
  vel_xy << vel(0), vel(1), 0.0;
  
  // Calculate State Dynamics
  if (state)
  {
    dx_.block<3,1>((int)dxPOS,0) = q_I_b.rota(vel); // R_I^b.T * vel
    if (use_drag_term_)
      dx_.block<3,1>((int)dxVEL,0) = acc_z + gravity_B - omega.cross(vel) - mu*vel_xy;
    else
      dx_.block<3,1>((int)dxVEL,0) = acc + gravity_B - omega.cross(vel);
    dx_.block<3,1>((int)dxATT, 0) = omega;
  }
  
  // State Jacobian
  if (jac)
  {
    A_.block<3,3>((int)dxPOS, (int)dxVEL) = R_I_b.transpose();
    A_.block<3,3>((int)dxPOS, (int)dxATT) = -R_I_b.transpose()*skew(vel);
    if (use_drag_term_)
    {
      A_.block<3,3>((int)dxVEL, (int)dxVEL) = -mu * I_2x3.transpose() * I_2x3 - skew(omega);
      A_.block<3,3>((int)dxVEL, (int)dxB_A) << 0, 0, 0, 0, 0, 0, 0, 0, -1;
      A_.block<3,1>((int)dxVEL, (int)dxMU) = -vel_xy;
    }
    else
    {
      A_.block<3,3>((int)dxVEL, (int)dxB_A) = -I_3x3;
      A_.block<3,3>((int)dxVEL, (int)dxVEL) = -skew(omega);
    }
    A_.block<3,3>((int)dxVEL, (int)dxATT) = skew(gravity_B);
    A_.block<3,3>((int)dxVEL, (int)dxB_G) = -skew(vel);
    A_.block<3,3>((int)dxATT, (int)dxB_G) = -I_3x3;
    
    // Input Jacobian
    if (use_drag_term_)
      G_.block<3,3>((int)dxVEL, (int)uA) << 0, 0, 0, 0, 0, 0, 0, 0, -1;
    else
      G_.block<3,3>((int)dxVEL, (int)uA) = -I_3x3;
    G_.block<3,3>((int)dxVEL, (int)uG) = -skew(vel);
    G_.block<3,3>((int)dxATT, (int)uG) = -I_3x3;
  }
  
  // Camera Dynamics
  Vector3d vel_c_i = q_b_c_.rotp(vel + omega.cross(p_b_c_));
  Vector3d omega_c_i = q_b_c_.rotp(omega);
  
  
  Quatd q_zeta;
  double rho;
  Vector3d zeta;
  Matrix<double, 3, 2> T_z;
  Matrix3d skew_zeta;
  Matrix3d skew_vel_c = skew(vel_c_i);
  Matrix3d skew_p_b_c = skew(p_b_c_);
  Matrix3d R_b_c = q_b_c_.R();
  int xZETA_i, xRHO_i, dxZETA_i, dxRHO_i;
  for (int i = 0; i < len_features_; i++)
  {
    xZETA_i = (int)xZ+i*5;
    xRHO_i = (int)xZ+5*i+4;
    dxZETA_i = (int)dxZ + i*3;
    dxRHO_i = (int)dxZ + i*3+2;
    
    q_zeta = (x.block<4,1>(xZETA_i, 0));
    rho = x(xRHO_i);
    zeta = q_zeta.rota(e_z);
    T_z = T_zeta(q_zeta);
    skew_zeta = skew(zeta);
    
    double rho2 = rho*rho;

    // Feature Dynamics
    if (state)
    {
      dx_.block<2,1>(dxZETA_i,0) = T_z.transpose() * (omega_c_i + rho * zeta.cross(vel_c_i));
      dx_(dxRHO_i) = rho2 * zeta.dot(vel_c_i);
    }
    
    // Feature Jacobian
    if (jac)
    {
      A_.block<2, 3>(dxZETA_i, (int)dxVEL) = rho * T_z.transpose() * skew_zeta * R_b_c;
      A_.block<2, 3>(dxZETA_i, (int)dxB_G) = T_z.transpose() * (rho * skew_zeta * R_b_c * skew_p_b_c - R_b_c);
      A_.block<2, 2>(dxZETA_i, dxZETA_i) = -T_z.transpose() * (skew(omega_c_i + rho * zeta.cross(vel_c_i)) + (rho * skew_vel_c * skew_zeta)) * T_z;
      A_.block<2, 1>(dxZETA_i, dxRHO_i) = T_z.transpose() * zeta.cross(vel_c_i);
      A_.block<1, 3>(dxRHO_i, (int)dxVEL) = rho2 * zeta.transpose() * R_b_c;
      A_.block<1, 3>(dxRHO_i, (int)dxB_G) = rho2 * zeta.transpose() * R_b_c * skew_p_b_c;
      A_.block<1, 2>(dxRHO_i, dxZETA_i) = rho2 * vel_c_i.transpose() * skew_zeta * T_z;
      A_(dxRHO_i, dxRHO_i) = 2 * rho * zeta.transpose() * vel_c_i;
      
      // Feature Input Jacobian
      G_.block<2, 3>(dxZETA_i, (int)uG) = T_z.transpose() * (rho*skew_zeta * R_b_c*skew_p_b_c - R_b_c);
      G_.block<1, 3>(dxRHO_i, (int)uG) = rho2*zeta.transpose() * R_b_c * skew_p_b_c;
    }
  }
}
}
