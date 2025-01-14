#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "Eigen/Dense"
#include "kalman_filter.h"
#include "measurement_package.h"
#include <fstream>
#include <string>
#include <vector>

class FusionEKF {
 public:
  /**
   * Constructor.
   */
  FusionEKF();

  /**
   * Destructor.
   */
  virtual ~FusionEKF();

  /**
   * Run the whole flow of the Kalman Filter from here.
   */
  void ProcessMeasurement(const MeasurementPackage &measurement_pack);

  /**
   * Kalman Filter update and prediction math lives in here.
   */
  KalmanFilter ekf_;

 private:
  // check whether the tracking toolbox was initialized or not (first measurement)
  bool is_initialized_;

  // previous timestamp
  long long previous_timestamp_;

  Eigen::MatrixXd R_laser_;
  Eigen::MatrixXd R_radar_;
  Eigen::MatrixXd H_laser_;
  Eigen::MatrixXd Hj_;

  // acceleration noise components
  double noise_ax_;
  double noise_ay_;

  // A helper method to calculate Jacobians.
  void CalculateJacobian(const Eigen::VectorXd &x_state);
};

#endif// FusionEKF_H_
