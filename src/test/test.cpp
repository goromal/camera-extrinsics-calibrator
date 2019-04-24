#include "test/test.h"

// Test calibrator on simulated data (can choose to use UAV truth states or EKF state estimates)
TEST(CalibrationTest, SimulatedData_TightlyCoupled)
{
  if (!experimental::filesystem::exists("../logs"))
    if (experimental::filesystem::create_directory("../logs"))
      cout << "*** Created directory: ../logs/ ***\n";

  // RUN SIMULATION
  MotionCaptureSimulator sim;
  sim.run();
  sim.logTruth("../logs/Simulated.Truth.log");
  sim.logXTruth("../logs/Simulated.x_truth.log");
  sim.logXEst("../logs/Simulated.x_est.log");
  sim.logFeat("../logs/Simulated.Features.Truth.log");
  sim.logLandmarks("../logs/Simulated.Features.Landmarks.log");

  // COLLECT SIMULATION DATA
  MatrixXd featureMeasurements, landmarks;
  Xformd XformB2C = sim.getXformB2C();
  MatrixXd stateMeas = sim.getStateTruth(); // Use truth UAV state for calibration
//  MatrixXd stateMeas = sim.getStateEst();   // Use EKF estimated UAV state for calibration
  convertMeasurements((FeatVec) sim.getFeat(), sim.getLandmarks(), featureMeasurements, landmarks);

  // BUILD AND RUN CALIBRATOR
  double error = calibrate("../params/cal_params_sim.yaml", XformB2C, featureMeasurements,
                           stateMeas, landmarks, "../logs/Simulation.XhatVOTC.log",
                           "../logs/Simulation.PixRes0.log", "../logs/Simulation.PixResf.log");

  EXPECT_LE(error, 0.1);
}

// Test calibrator on data given by Planck 2-26-2019 (logs created by by data/processPlanckData.m)
TEST(CalibrationTest, HardwareData_TightlyCoupled)
{
  // COLLECT HARDWARE DATA
  MatrixXd featureMeasurements, stateMeas, landmarks;
  Xformd XformB2C;

  // LOAD CALIBRATION DATA FROM LOGS
  readOffsetLog("../data/logs/offsets_Planck.log", XformB2C);
  readFeaturesLog("../data/logs/features_Planck.log", featureMeasurements);
  readStatesLog("../data/logs/states_Planck.log", stateMeas);
  readLandmarksLog("../data/logs/landmarks_Planck.log", landmarks);

  // BUILD AND RUN CALIBRATOR
  double error = calibrate("../params/cal_params_hw.yaml", XformB2C, featureMeasurements,
                           stateMeas, landmarks, "../logs/Hardware.XhatVOTC.log",
                           "../logs/Hardware.PixRes0.log", "../logs/Hardware.PixResf.log");

  EXPECT_LE(error, 0.1);
}


