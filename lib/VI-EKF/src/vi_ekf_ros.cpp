#include "vi_ekf_ros.h"
#include "eigen_helpers.h"

VIEKF_ROS::VIEKF_ROS() :
  nh_private_("~"),
  it_(nh_)
{
  imu_sub_ = nh_.subscribe("imu", 500, &VIEKF_ROS::imu_callback, this);
  pose_sub_ = nh_.subscribe("truth/pose", 10, &VIEKF_ROS::pose_truth_callback, this);
  transform_sub_ = nh_.subscribe("truth/transform", 10, &VIEKF_ROS::transform_truth_callback, this);
//  odometry_pub_ = nh_.advertise<nav_msgs::Odometry>("odom", 1);
//  bias_pub_ = nh_.advertise<sensor_msgs::Imu>("imu/bias", 1);
  
  image_sub_ = it_.subscribe("color", 10, &VIEKF_ROS::color_image_callback, this);
  depth_sub_ = it_.subscribe("depth", 10, &VIEKF_ROS::depth_image_callback, this);
//  output_pub_ = it_.advertise("tracked", 1);
//  cov_img_pub_ = it_.advertise("covariance", 1);
  
  std::string log_directory, feature_mask;
  std::string default_log_folder = ros::package::getPath("vi_ekf") + "/logs/" + to_string(ros::Time::now().sec);
  nh_private_.param<std::string>("log_directory", log_directory, default_log_folder );
  nh_private_.param<std::string>("feature_mask", feature_mask, "");
  nh_private_.param<bool>("record_video", record_video_, true);
  
  
  Eigen::Matrix<double, vi_ekf::VIEKF::xZ, 1> x0;
  Eigen::Matrix<double, vi_ekf::VIEKF::dxZ, 1> P0diag, Qxdiag, lambda;
  uVector Qudiag;
  Vector3d P0feat, Qxfeat, lambdafeat;
  Vector2d cam_center, focal_len;
  Vector4d q_b_c, q_b_IMU, q_I_truth;
  Vector3d p_b_c;
  Vector2d feat_r_diag, acc_r_drag_diag;
  Vector3d att_r_diag, pos_r_diag, vel_r_diag, acc_r_grav_diag;
  Vector2i image_size;
  importMatrixFromParamServer(nh_private_, x0, "x0");
  importMatrixFromParamServer(nh_private_, P0diag, "P0");
  importMatrixFromParamServer(nh_private_, Qxdiag, "Qx");
  importMatrixFromParamServer(nh_private_, lambda, "lambda");
  importMatrixFromParamServer(nh_private_, lambdafeat, "lambda_feat");
  importMatrixFromParamServer(nh_private_, Qudiag, "Qu");
  importMatrixFromParamServer(nh_private_, P0feat, "P0_feat");
  importMatrixFromParamServer(nh_private_, Qxfeat, "Qx_feat");
  importMatrixFromParamServer(nh_private_, cam_center, "cam_center");
  importMatrixFromParamServer(nh_private_, focal_len, "focal_len");
  importMatrixFromParamServer(nh_private_, q_b_c, "q_b_c");
  importMatrixFromParamServer(nh_private_, p_b_c, "p_b_c");
  importMatrixFromParamServer(nh_private_, feat_r_diag, "feat_R");
  importMatrixFromParamServer(nh_private_, acc_r_drag_diag, "acc_R_drag");
  importMatrixFromParamServer(nh_private_, acc_r_grav_diag, "acc_R_grav");
  importMatrixFromParamServer(nh_private_, att_r_diag, "att_R");
  importMatrixFromParamServer(nh_private_, pos_r_diag, "pos_R");
  importMatrixFromParamServer(nh_private_, vel_r_diag, "vel_R");
  importMatrixFromParamServer(nh_private_, q_b_IMU, "q_b_IMU");
  importMatrixFromParamServer(nh_private_, q_I_truth, "q_I_truth");
  importMatrixFromParamServer(nh_private_, image_size, "image_size");
  q_b_IMU_.arr_ = q_b_IMU;
  q_I_truth_.arr_ = q_I_truth;
  double depth_r, alt_r, keyframe_overlap;
  bool partial_update, keyframe_reset;
  int feature_radius, cov_prop_skips;
  ROS_FATAL_COND(!nh_private_.getParam("depth_R", depth_r), "you need to specify the 'depth_R' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("alt_R", alt_r), "you need to specify the 'alt_R' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("min_depth", min_depth_), "you need to specify the 'min_depth' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("imu_LPF", IMU_LPF_), "you need to specify the 'imu_LPF' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("truth_LPF", truth_LPF_), "you need to specify the 'truth_LPF' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("num_features", num_features_), "you need to specify the 'num_features' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("invert_image", invert_image_), "you need to specify the 'invert_image' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("partial_update", partial_update), "you need to specify the 'partial_update' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("drag_term", use_drag_term_), "you need to specify the 'drag_term' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("keyframe_reset", keyframe_reset), "you need to specify the 'keyframe_reset' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("keyframe_overlap", keyframe_overlap), "you need to specify the 'keyframe_overlap' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("feature_radius", feature_radius), "you need to specify the 'feature_radius' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("cov_prop_skips", cov_prop_skips), "you need to specify the 'cov_prop_skips' parameter");
  
  num_features_ = (num_features_ > NUM_FEATURES) ? NUM_FEATURES : num_features_;
  
  P0feat(2,0) = 1.0/(16.0 * min_depth_ * min_depth_);
  
  ekf_.init(x0, P0diag, Qxdiag, lambda, Qudiag, P0feat, Qxfeat, lambdafeat,
            cam_center, focal_len, q_b_c, p_b_c, min_depth_, log_directory, 
            use_drag_term_, partial_update, keyframe_reset, keyframe_overlap, cov_prop_skips);
  ekf_.register_keyframe_reset_callback(std::bind(&VIEKF_ROS::keyframe_reset_callback, this));
  
  is_flying_ = false; // Start out not flying
  ekf_.set_drag_term(false); // Start out not using the drag term
  
  klt_tracker_.init(num_features_, false, feature_radius, cv::Size(image_size(0,0), image_size(1,0)));
  if (!feature_mask.empty())
  {
    klt_tracker_.set_feature_mask(feature_mask);
  }
  
  // Initialize keyframe variables
  kf_att_ = Quatd::Identity();
  kf_pos_.setZero();
  
  // Initialize the depth image to all NaNs
  depth_image_ = cv::Mat(image_size(0,0), image_size(1,0), CV_32FC1, cv::Scalar(NAN));
  got_depth_ = false;
  
  // Initialize the measurement noise covariance matrices
  depth_R_ << depth_r;
  feat_R_ = feat_r_diag.asDiagonal();
  acc_R_drag_ = acc_r_drag_diag.asDiagonal();
  acc_R_grav_ = acc_r_grav_diag.asDiagonal();
  att_R_ = att_r_diag.asDiagonal();
  pos_R_ = pos_r_diag.asDiagonal();
  vel_R_ = vel_r_diag.asDiagonal();
  alt_R_ << alt_r;
  
  // Turn on the specified measurements
  ROS_FATAL_COND(!nh_private_.getParam("use_truth", use_truth_), "you need to specify the 'use_truth' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("use_depth", use_depth_), "you need to specify the 'use_depth' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("use_features", use_features_), "you need to specify the 'use_features' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("use_acc", use_acc_), "you need to specify the 'use_acc' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("use_imu_att", use_imu_att_), "you need to specify the 'use_imu_att' parameter");
  ROS_FATAL_COND(!nh_private_.getParam("use_alt", use_alt_), "you need to specify the 'use_alt' parameter");
  
  cout << "\nlog file: " << log_directory << "\n";
  cout << "\nMEASUREMENTS\tFEATURES\n==============================\n";
  cout << "truth: " << use_truth_ << "\t";
  cout << "partial update: " << partial_update << "\n";
  cout << "depth: " << use_depth_ << "\t";
  cout << "drag_term: " << use_drag_term_ << "\n";
  cout << "features: " << use_features_ << "\t";
  cout << "keyframe_reset: " << keyframe_reset << "\n";
  cout << "acc: " << use_acc_ << "\n";
  cout << "imu_att: " << use_imu_att_ << "\n";
  cout << "imu_alt: " << use_alt_ << "\n";

  
  // Wait for truth to initialize pose
  imu_init_ = false;
  truth_init_ = false;
  
  odom_msg_.header.frame_id = "body";
  
  if (record_video_)
    video_.open(log_directory + "video.avi", cv::VideoWriter::fourcc('M','J','P','G'), 30, cv::Size(image_size(0,0),image_size(1,0)), true);
}

VIEKF_ROS::~VIEKF_ROS()
{}

void VIEKF_ROS::imu_callback(const sensor_msgs::ImuConstPtr &msg)
{
  
  u_ << msg->linear_acceleration.x,
      msg->linear_acceleration.y,
      msg->linear_acceleration.z,
      msg->angular_velocity.x,
      msg->angular_velocity.y,
      msg->angular_velocity.z;
  u_.block<3,1>(0,0) = q_b_IMU_.rotp(u_.block<3,1>(0,0));
  u_.block<3,1>(3,0) = q_b_IMU_.rotp(u_.block<3,1>(3,0));
  
  imu_ = (1. - IMU_LPF_) * u_ + IMU_LPF_ * imu_;
  
  if (!truth_init_)
    return;
  else if (!imu_init_)
  {
    imu_ = u_;
    imu_init_ = true;
    start_time_ = msg->header.stamp;
    return;
  }
  double t = (msg->header.stamp - start_time_).toSec();
  
  // Propagate filter
  ekf_mtx_.lock();
  ekf_.propagate_state(imu_, t);
  ekf_mtx_.unlock();

  
  // update accelerometer measurement
  ekf_mtx_.lock();
  if (ekf_.get_drag_term() == true)
  {
    z_acc_drag_ = imu_.block<2,1>(0, 0);
    ekf_.add_measurement(t, z_acc_drag_, vi_ekf::VIEKF::ACC, acc_R_drag_, use_acc_ && is_flying_);
  }
  else
  {
    z_acc_grav_ = imu_.block<3,1>(0, 0);
    double norm = z_acc_grav_.norm();
    if (norm < 9.80665 * 1.15 && norm > 9.80665 * 0.85)
      ekf_.add_measurement(t, z_acc_grav_, vi_ekf::VIEKF::ACC, acc_R_grav_, use_acc_);
  }
    ekf_mtx_.unlock();
  
  // update attitude measurement
  z_att_ << msg->orientation.w, msg->orientation.x, msg->orientation.y, msg->orientation.z;
  ekf_mtx_.lock();
  if (use_imu_att_)
    ekf_.add_measurement(t, z_att_, vi_ekf::VIEKF::ATT, att_R_, (use_truth_) ? true : use_imu_att_);
  ekf_mtx_.unlock();
  
//  odom_msg_.header.stamp = msg->header.stamp;
//  odom_msg_.pose.pose.position.x = ekf_.get_state()(vi_ekf::VIEKF::xPOS,0);
//  odom_msg_.pose.pose.position.y = ekf_.get_state()(vi_ekf::VIEKF::xPOS+1,0);
//  odom_msg_.pose.pose.position.z = ekf_.get_state()(vi_ekf::VIEKF::xPOS+2,0);
//  odom_msg_.pose.pose.orientation.w = ekf_.get_state()(vi_ekf::VIEKF::xATT,0);
//  odom_msg_.pose.pose.orientation.x = ekf_.get_state()(vi_ekf::VIEKF::xATT+1,0);
//  odom_msg_.pose.pose.orientation.y = ekf_.get_state()(vi_ekf::VIEKF::xATT+2,0);
//  odom_msg_.pose.pose.orientation.z = ekf_.get_state()(vi_ekf::VIEKF::xATT+3,0);
//  odom_msg_.twist.twist.linear.x = ekf_.get_state()(vi_ekf::VIEKF::xVEL,0);
//  odom_msg_.twist.twist.linear.y = ekf_.get_state()(vi_ekf::VIEKF::xVEL+1,0);
//  odom_msg_.twist.twist.linear.z = ekf_.get_state()(vi_ekf::VIEKF::xVEL+2,0);
//  odom_msg_.twist.twist.angular.x = imu_(3);
//  odom_msg_.twist.twist.angular.y = imu_(4);
//  odom_msg_.twist.twist.angular.z = imu_(5);
//  odometry_pub_.publish(odom_msg_);
  
//  bias_msg_.header = msg->header;
//  bias_msg_.linear_acceleration.x = ekf_.get_state()(vi_ekf::VIEKF::xB_A, 0);
//  bias_msg_.linear_acceleration.y = ekf_.get_state()(vi_ekf::VIEKF::xB_A+1, 0);
//  bias_msg_.linear_acceleration.z = ekf_.get_state()(vi_ekf::VIEKF::xB_A+2, 0);
//  bias_msg_.angular_velocity.x = ekf_.get_state()(vi_ekf::VIEKF::xB_G, 0);
//  bias_msg_.angular_velocity.y = ekf_.get_state()(vi_ekf::VIEKF::xB_G+1, 0);
//  bias_msg_.angular_velocity.z = ekf_.get_state()(vi_ekf::VIEKF::xB_G+2, 0);
//  bias_pub_.publish(bias_msg_);
}

void VIEKF_ROS::keyframe_reset_callback()
{
  
  kf_pos_ = truth_pos_;
  kf_pos_(2) = 0.0; // always at the ground
  
  /// OLD WAY
  kf_att_ = Quatd::from_euler(0, 0, truth_att_.yaw());
  
// ///COOL WAY
//  Vector3d u_rot = truth_att_.rota(vi_ekf::khat);
//  Vector3d v = vi_ekf::khat.cross(u_rot); // Axis of rotation (without rotation about khat)
//  double theta = vi_ekf::khat.transpose() * u_rot; // Angle of rotation
//  Quat qp = Quat::exp(theta * v); // This is the same quaternion, but without rotation about z
//  // Extract z-rotation only
//  kf_att_ = (truth_att_ * qp.inverse());  
  
  
}

void VIEKF_ROS::color_image_callback(const sensor_msgs::ImageConstPtr &msg)
{
  if (!imu_init_)
    return;
  
  try
  {
    cv_ptr_ = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_FATAL("cv_bridge exception: %s", e.what());
    return;
  }
  
  if (invert_image_)
    cv::flip(cv_ptr_->image, img_, -1);
  else
    cv_ptr_->image.copyTo(img_);
  
  
  // Track Features in Image
  klt_tracker_.load_image(img_, msg->header.stamp.toSec(), features_, ids_);
  
  ekf_mtx_.lock();
  // Propagate the covariance
//  ekf_.propagate_Image();
  // Set which features we are keeping
  ekf_.keep_only_features(ids_);
  ekf_mtx_.unlock();

  double t = (msg->header.stamp - start_time_).toSec();
  
  for (int i = 0; i < features_.size(); i++)
  {
    int x = round(features_[i].x);
    int y = round(features_[i].y);
    // The depth image encodes depth in mm
    float depth = depth_image_.at<float>(y, x) * 1e-3;
    if (depth > 1e3)
      depth = NAN;
    else if (depth < min_depth_)
      depth = NAN;
    
    z_feat_ << features_[i].x, features_[i].y;
    z_depth_ << depth;
    ekf_mtx_.lock();
    int result = ekf_.add_measurement(t, z_feat_, vi_ekf::VIEKF::FEAT, feat_R_, use_features_, ids_[i], (use_depth_) ? depth : NAN);
    if (result == vi_ekf::VIEKF::MEAS_SUCCESS && got_depth_ && !(depth != depth))
        ekf_.add_measurement(t, z_depth_, vi_ekf::VIEKF::DEPTH, depth_R_, use_depth_, ids_[i]);    
    ekf_mtx_.unlock();   
  }

  ekf_mtx_.lock();
  std::vector<int> gated_ids;
  ekf_.handle_measurements(&gated_ids);
  for (auto it = gated_ids.begin(); it != gated_ids.end(); it++)
  {
    klt_tracker_.drop_feature(*it);
  }
  ekf_mtx_.unlock();
    
//    // Draw depth and position of tracked features
//    z_feat_ = ekf_.get_feat(ids_[i]);
//    circle(img_, features_[i], 5, Scalar(0,255,0));
//    circle(img_, Point(z_feat_.x(), z_feat_.y()), 5, Scalar(255, 0, 255));
//    double h_true = 50.0 /depth;
//    double h_est = 50.0 /ekf_.get_depth(ids_[i]);
//    rectangle(img_, Point(x-h_true, y-h_true), Point(x+h_true, y+h_true), Scalar(0, 255, 0));
//    rectangle(img_, Point(z_feat_.x()-h_est, z_feat_.y()-h_est), Point(z_feat_.x()+h_est, z_feat_.y()+h_est), Scalar(255, 0, 255));
//  }
//  if (record_video_)
//    video_ << img_;
//  sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img_).toImageMsg();
//  output_pub_.publish(img_msg);
}

void VIEKF_ROS::depth_image_callback(const sensor_msgs::ImageConstPtr &msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_FATAL("cv_bridge exception: %s", e.what());
    return;
  }
  if (invert_image_)
    cv::flip(cv_ptr->image, depth_image_, ROTATE_180);
  else
    cv_ptr->image.copyTo(depth_image_);
  got_depth_ = true;
}

void VIEKF_ROS::pose_truth_callback(const geometry_msgs::PoseStampedConstPtr &msg)
{
  z_pos_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  z_att_ << msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z;
  truth_callback(z_pos_, z_att_, msg->header.stamp);
}

void VIEKF_ROS::transform_truth_callback(const geometry_msgs::TransformStampedConstPtr &msg)
{
  z_pos_ << msg->transform.translation.x, msg->transform.translation.y, msg->transform.translation.z;
  z_att_ << msg->transform.rotation.w, msg->transform.rotation.x, msg->transform.rotation.y, msg->transform.rotation.z;
  truth_callback(z_pos_, z_att_, msg->header.stamp);
}

void VIEKF_ROS::truth_callback(Vector3d& z_pos, Vector4d& z_att, ros::Time time)
{
  static int counter = 0;
  if (counter++ < 2)
  {
    return;
  }
  else counter = 0;
  // Rotate measurements into the proper frame
  z_pos = q_I_truth_.rotp(z_pos);
  z_att.block<3,1>(1,0) = q_I_truth_.rotp(z_att.block<3,1>(1,0));
  
  // Make sure that the truth quaternion is the right sign (for plotting)
  if (sign(z_att(0,0)) != sign(ekf_.get_state()(vi_ekf::VIEKF::xATT, 0)))
  {
    z_att *= -1.0;
  }
  
  truth_pos_ = z_pos;
  
  // Initialize Truth
  if (!truth_init_)
  {
    truth_att_ = Quatd(z_att);
    
    // Initialize the EKF to the origin in the Vicon frame, but then immediately keyframe reset to start at origin
    ekf_mtx_.lock();
    Matrix<double, vi_ekf::VIEKF::xZ, 1> x0 = ekf_.get_state().topRows(vi_ekf::VIEKF::xZ);
    x0.block<3,1>((int)vi_ekf::VIEKF::xPOS,0) = z_pos;
    x0.block<4,1>((int)vi_ekf::VIEKF::xATT,0) = z_att;
    ekf_.set_x0(x0);
    ekf_.keyframe_reset();
    ekf_mtx_.unlock();
    
    truth_init_ = true;
    return;
  }
  
  // Decide whether we are flying or not
  if (!is_flying_)
  {
    Vector3d error = truth_pos_ - kf_pos_;
    // If we have moved a centimeter, then assume we are flying
    if (error.norm() > 1e-2)
    {
      is_flying_ = true;
      time_took_off_ = time;
      // The drag term is now valid, activate it if we are supposed to use it.
    }
  }
  
  if (time > time_took_off_ + ros::Duration(10.0))
  {
    // After 1 second of flying, turn on the drag term if we are supposed to
    if (use_drag_term_ == true && ekf_.get_drag_term() ==  false)
      ekf_.set_drag_term(true);
  }
  
  // Low-pass filter Attitude (use manifold)
  Quatd y_t(z_att);
  truth_att_ = y_t + (truth_LPF_ * (truth_att_ - y_t));
  
  z_alt_ << -z_pos(2,0);
  
  global_transform_.t() = z_pos;
  global_transform_.q() = truth_att_;
  ekf_mtx_.lock();
  ekf_.log_global_position(global_transform_);
  ekf_mtx_.unlock();
  
  // Convert truth measurement into current node frame
  z_pos = kf_att_.rotp(z_pos - kf_pos_); // position offset, rotated into keyframe
  z_att = (kf_att_.inverse() * truth_att_).elements();  
  
  bool truth_active = (use_truth_ || !is_flying_);

  double t = (time - start_time_).toSec();
  
  ekf_mtx_.lock();
  ekf_.add_measurement(t, z_pos, vi_ekf::VIEKF::POS, pos_R_, truth_active);
  ekf_.add_measurement(t, z_att, vi_ekf::VIEKF::ATT, att_R_, truth_active);
  ekf_mtx_.unlock();
  
  // Perform Altitude Measurement
  ekf_mtx_.lock();
  ekf_.add_measurement(t, z_alt_, vi_ekf::VIEKF::ALT, alt_R_, !truth_active);
  ekf_mtx_.unlock();
}



