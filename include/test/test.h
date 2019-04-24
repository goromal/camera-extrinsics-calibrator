#pragma once

#include <iostream>
#include <experimental/filesystem>
#include "gtest/gtest.h"
#include "cal.h"
#include "sim.h"

/* Helper functions for test.cpp */

void convertMeasurements(const FeatVec &simFeatures,
                         const std::vector<Vector3d, aligned_allocator<Vector3d>> &landmarks_vec,
                         MatrixXd &featureMeasurements, MatrixXd &landmarks)
{
  featureMeasurements.resize(4, simFeatures.size());
  landmarks.resize(3, landmarks_vec.size());

  for (int i = 0; i < simFeatures.size(); i++)
  {
    featureMeasurements.col(i) << simFeatures[i].t, simFeatures[i].id, simFeatures[i].z[0],
                            simFeatures[i].z[1];
  }

  for (int i = 0; i < landmarks_vec.size(); i++)
  {
    landmarks.col(i) << landmarks_vec[i](0), landmarks_vec[i](1), landmarks_vec[i](2);
  }
}

int readLogOfDoubles(const std::string &filename, std::vector<double> &data)
{
  std::ifstream f(filename);
  double read;
  int size = 0;
  while (!f.fail())
  {
    f.read((char *)&read, sizeof(double));
    if (!f.fail())
    {
      data.push_back(read);
      size++;
    }
  }
  f.close();
  return size;
}

void logToMatrix(const std::string &filename, MatrixXd &matrix, int rowSize)
{
  std::vector<double> data;
  int size = readLogOfDoubles(filename, data);
  int num_cols = size / rowSize;
  matrix.resize(rowSize, num_cols);

  for (int i = 0; i < num_cols; i++)
  {
    int idx = i * rowSize;
    for (int j = 0; j < rowSize; j++)
    {
      matrix(j, i) = data[idx+j];
    }
  }
}

void readOffsetLog(const std::string &filename, Xformd &XformB2C)
{
  MatrixXd offsetMatrix;
  logToMatrix(filename, offsetMatrix, 7);
  XformB2C = offsetMatrix.col(0);
}

void readStatesLog(const std::string &filename, MatrixXd &stateMeas)
{
  logToMatrix(filename, stateMeas, 8);
}

void readFeaturesLog(const std::string &filename, MatrixXd &featureMeasurements)
{
  logToMatrix(filename, featureMeasurements, 4);
}

void readLandmarksLog(const std::string &filename, MatrixXd &landmarks)
{
  logToMatrix(filename, landmarks, 3);
}

double calibrate(const std::string &paramfile, Xformd &XformB2C, MatrixXd &featureMeas,
                 MatrixXd &stateMeas, MatrixXd &landmarks,
                 const std::string &xhatvologname, const std::string &ipixreslogname,
                 const std::string &fpixreslogname)
{
  cout << "\n\n\n";
  cout << " ****************************************** " << endl;
  cout << "   SOLVING BODY TO CAMERA (TC) ESTIMATION   " << endl;
  cout << " ****************************************** " << endl;

  CamExtCalibrator cal(paramfile);

  cal.setup(stateMeas, featureMeas, landmarks);
  cal.logxhatVOTC(xhatvologname);
  cal.logPixelResiduals(ipixreslogname);
  cal.solve();
  cal.logPixelResiduals(fpixreslogname);

  double b2c_error = (XformB2C - cal.getEstXformB2C()).norm();
  cout << "x_b2c:    " << XformB2C << endl;
  cout << "xhat_b2c: " << cal.getEstXformB2C() << endl;
  cout << "e:        " << b2c_error << endl << endl;

  return b2c_error;
}
