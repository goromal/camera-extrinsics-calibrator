#include "vi_ekf.h"

namespace vi_ekf
{

bool VIEKF::NaNsInTheHouse() const
{
  int x_max = xZ + len_features_ *5;
  int dx_max = dxZ + len_features_*3;
  if( ( (x_[i_].topRows(x_max)).array() != (x_[i_].topRows(x_max)).array()).any()
      || ((P_[i_].topLeftCorner(dx_max,dx_max)).array() != (P_[i_].topLeftCorner(dx_max,dx_max)).array()).any() )
  {
    std::cout << "x:\n" << x_[i_] << "\n";
    std::cout << "P:\n" << P_[i_] << "\n";
    return true;
  }
  else
    return false;
}

bool VIEKF::BlowingUp() const
{
  if ( ((x_[i_]).array() > 1e6).any() || ((P_[i_]).array() > 1e6).any())
    return true;
  else
    return false;
}

bool VIEKF::NegativeDepth() const
{
  for (int i = 0; i < len_features_; i++)
  {
    int xRHO_i = (int)xZ+5*i+4;
    if (x_[i_](xRHO_i,0) < 0)
      return true;
  }
  return false;
}

}
