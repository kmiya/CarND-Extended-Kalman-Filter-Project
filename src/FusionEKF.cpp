#include "FusionEKF.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

/**
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225,      0,
                   0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09,      0,    0,
                 0, 0.0009,    0,
                 0,      0, 0.09;

  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  Hj_ << 0, 0, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0;

  // set the acceleration noise components
  noise_ax_ = 9;
  noise_ay_ = 9;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() = default;

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /**
   * Initialization
   */
  if (!is_initialized_) {
    // the initial transition matrix F_
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ << 1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1;

    // state covariance matrix P
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 1, 0,
               0, 0, 0, 1;

    // set the process covariance matrix Q
    ekf_.Q_ = MatrixXd(4, 4);

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates
      // and initialize state.
      const double ro = measurement_pack.raw_measurements_[0];
      const double theta = measurement_pack.raw_measurements_[1];
      ekf_.x_ << ro * cos(theta),
                 ro * sin(theta),
                 0,
                 0;
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state.
      ekf_.x_ << measurement_pack.raw_measurements_[0],
                 measurement_pack.raw_measurements_[1],
                 0,
                 0;
    }
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  // compute the time elapsed between the current and previous measurements
  // dt - expressed in seconds
  const double dt = static_cast<double>((measurement_pack.timestamp_ - previous_timestamp_)) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  /**
   * Prediction
   */

  // Update the state transition matrix F according to the new elapsed time.
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // Update the process noise covariance matrix.
  const double dt_2 = dt * dt;
  const double dt_3 = dt_2 * dt;
  const double dt_4 = dt_3 * dt;
  ekf_.Q_ << dt_4 / 4 * noise_ax_, 0, dt_3 / 2 * noise_ax_, 0,
             0, dt_4 / 4 * noise_ay_, 0, dt_3 / 2 * noise_ay_,
             dt_3 / 2 * noise_ax_, 0, dt_2 * noise_ax_, 0,
             0, dt_3 / 2 * noise_ay_, 0, dt_2 * noise_ay_;

  ekf_.Predict();

  /**
   * Update
   */
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // update Hj_
    CalculateJacobian(ekf_.x_);
    ekf_.H_ = Hj_;
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
void FusionEKF::CalculateJacobian(const VectorXd &x_state) {
  // recover state parameters
  const double px = x_state(0);
  const double py = x_state(1);
  const double vx = x_state(2);
  const double vy = x_state(3);

  // pre-compute a set of terms to avoid repeated calculation
  const double c1 = px * px + py * py;
  const double c2 = sqrt(c1);
  const double c3 = (c1 * c2);

  // check division by zero
  if (fabs(c1) < 0.0001) {
    cout << "CalculateJacobian () - Error - Division by Zero" << endl;
    return;
  }

  // compute the Jacobian matrix
  Hj_ << (px / c2), (py / c2), 0, 0,
         -(py / c1), (px / c1), 0, 0,
         py * (vx * py - vy * px) / c3, px * (px * vy - py * vx) / c3, px / c2, py / c2;
}
