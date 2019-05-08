# Camera Extrinsis Calibrator

C++ Library for interpreting UAV state and image feature measurement data of known landmarks to find the extrinsic parameters of the UAV camera, which is the homogeneous transform between the UAV body frame and the camera frame.

The calibrator takes the following input data:

- UAV poses over a time interval, expressed in an inertial frame
- Visual landmark positions (assumed to be static), expressed in the same inertial frame
- Camera pixel measurement data for each visual landmark over the time interval

and outputs the following data:

- Estimate of the relative position between the camera and the UAV frame (assumed constant)
- Estimate of the relative orientation between the camera and the UAV frame (assumed constant)

This repository is installed and run using CMake in Ubuntu 18.04.

# Underlying Theory

Please see the document *Theory.pdf* at the top-level of this repository for an explanation of the mathematics behind the calibration routine. It should be helpful for understanding everything else.

# Repository Explanation

This repository contains tools for performing and visualizing the extrinsic camera calibration routine in both simulation and with hardware data. These scenarios are covered in two unit tests, found in *src/test/test.cpp*.

## Simulation (Unit Test 1)

This unit test first simulates an environment with visual landmark features and a UAV flight path, then runs the calibration routine on the output data. The following sequence of steps should be taken when running the unit test:

1. Set important simulation parameters such as simulation time, UAV waypoints, landmark positions, control gains, and sensor noise in *params/sim\_params.yaml*.

2. Ensure that the camera and initial offset guess parameters set in *params/cal\_params\_sim.yaml* are satisfactory.

3. Run the unit test.

4. Run the Matlab script *matlab/visualizeSim.m* to view the landmark locations, flight path, and camera measurements over the simulation time interval.

5. Run the Matlab script *matlab/plotSimulationResiduals.m* to visualize the optimizer output in the form of the residuals.

   By default, the calibrator runs off of simulation truth data for the UAV poses. This can be changed to make the calibrator run using UAV poses from a visual odometry EKF by uncommenting and commenting two lines in *src/test/test.cpp*:

   ```c++
     MatrixXd stateMeas = sim.getStateTruth(); // Use truth UAV state for calibration
   //  MatrixXd stateMeas = sim.getStateEst();   // Use EKF estimated UAV state for calibration
   ```

   To run off of estimated UAV pose data, comment the first line and uncomment the second line. The performance of the EKF versus simulation truth can be visualized with the Matlab script *matlab/plotSimEstimates.m*.

## Hardware Data (Unit Test 2)

This unit test runs the calibration routine on provided hardware data. The following sequence of steps should be taken when running the unit test for the first time:

1. Pre-process the data (found in *data/planck\_ctrl\_2019-02-22-21-53-35.mat*) by running the Matlab script *data/processPlanckData.m*. If desired, set the variable *visualize* to *true* to view the landmark locations, flight path, and camera measurements over the time interval.
2. Ensure that the camera and initial offset guess parameters set in  *params/cal\_params\_hw.yaml* are satisfactory.
3. Run the unit test.
4. Run the Matlab script *matlab/plotHardwareResiduals.m* to visualize the optimizer output in the form of the residuals.

# Suggestions for Future Work

- The data for Unit Test 2 is missing accurate information for the following two things, which should be amended:
  - The true size of the visual landmark tag used in the data collection.
  - The true extrinsic offset values. 
- Currently, it is assumed that the provided visual landmark positions are absolutely true, so they are held as constant parameters by the optimizer. This might be disadvantageous in practice. It is possible that setting the landmark positions as additional decision variables for the optimizer could amend things, though it would probably help to specify additional geometric constraints pertaining to the landmark positions to help the optimizer converge.
- *Ceres*, the optimizer library used in this calibration routine, has loss function tools for outlier rejection that have not currently been explored in this application and may be useful.