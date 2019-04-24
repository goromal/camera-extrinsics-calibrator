#include "math_helper.h"

void removeRow(Eigen::MatrixXd& matrix, unsigned int rowToRemove)
{
    unsigned int numRows = matrix.rows()-1;
    unsigned int numCols = matrix.cols();

    if( rowToRemove < numRows )
        matrix.block(rowToRemove,0,numRows-rowToRemove,numCols) = matrix.bottomRows(numRows-rowToRemove);

    matrix.conservativeResize(numRows,numCols);
}

void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.rightCols(numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}


void concatenate_SE2(Eigen::Vector3d& T1, Eigen::Vector3d& T2, Eigen::Vector3d& Tout)
{
  double cs = std::cos(T1(2,0));
  double ss = std::sin(T1(2,0));
  Tout(0) = T1(0) + T2(0) * cs + T2(1) * ss;
  Tout(1) = T1(1) - T2(0) * ss + T2(1) * cs;
  double psi= T1(2) + T2(2);
  if (psi > M_PI)
      psi -= 2.*M_PI;
  else if (psi < -M_PI)
      psi += 2.*M_PI;
  Tout(2) = psi;
}

void concatenate_edges(const Eigen::Matrix<double,7,1>& T1, const Eigen::Matrix<double,7,1>& T2, Eigen::Matrix<double,7,1>& Tout)
{
  quat::Quatd q1(T1.block<4,1>(3,0));
  quat::Quatd q2(T2.block<4,1>(3,0));

  Tout.block<3,1>(0,0) = T1.block<3,1>(0,0) + q1.rota(T2.block<3,1>(0,0));
  Tout.block<4,1>(3,0) = (q1 * q2).elements();
}

const Eigen::Matrix<double,7,1> invert_edge(const Eigen::Matrix<double,7,1>& T1)
{
  quat::Quatd qinv(T1.block<4,1>(3,0));
  qinv.invert();
  Eigen::Matrix<double, 7,1> Tout;
  Tout.topRows(3) = qinv.rotp(T1.topRows(3));
  Tout.bottomRows(4) = qinv.elements();
}


void invert_SE2(Eigen::Vector3d& T, Eigen::Vector3d& Tout)
{
  double cs = std::cos(T(2));
  double ss = std::sin(T(2));
  Tout(0) = -(T(0) * cs + T(1) * ss);
  Tout(1) = -(- T(0) * ss + T(1) * cs);
  Tout(2) = -T(2);
}
