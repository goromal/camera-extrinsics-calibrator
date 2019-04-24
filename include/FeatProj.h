#pragma once

#include <Eigen/Core>
#include <ceres/ceres.h>
#include "geometry/cam.h"
#include "geometry/xform.h"

using namespace Eigen;
using namespace xform;

class FeatProj
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FeatProj(const Vector2d &pix, const Xformd &x_I2b)
  {
    pix_ = pix; // measured feature pixels
    x_I2b_ = x_I2b; // UAV pose at time of measurement
  }

  // Calculate residual for given parameters
  template <typename T>
  bool operator()(const T *_ptI, const T *_x_b2c, const T *_f, const T *_c, const T *_s, const T *_d, const T *dt,
                  T *_res) const
  {
    (void) dt;
    typedef Matrix<T, 3, 1> Vec3;
    typedef Matrix<T, 2, 1> Vec2;

    Camera<T> cam(_f, _c, _d, _s);
    Map<const Vec3> pt_I(_ptI); // point in the inertial frame
    Xform<T> x_b2c(_x_b2c); // transform from body to camera
    Map<Vec2> res(_res); // residuals

    Xform<T> x_I2c = x_I2b_.otimes<T>(x_b2c);

    Vec3 pt_c = x_I2c.transformp(pt_I); // point in the camera frame
    Vec2 pixhat; // estimated pixel location
    cam.proj(pt_c, pixhat);

    res = pix_ - pixhat;
    return true;
  }

private:
  Vector2d pix_;
  Xformd x_I2b_;
};
typedef ceres::AutoDiffCostFunction<FeatProj, 2, 3, 7, 2, 2, 1, 5, 1> FeatProjFactor;

