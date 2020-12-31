#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() = default;

KalmanFilter::~KalmanFilter() = default;

void KalmanFilter::Predict() {
  /**
   * predict the state
   */
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * update the state by using Kalman Filter equations
   */
  const VectorXd y = z - H_ * x_;
  const MatrixXd Ht = H_.transpose();
  const MatrixXd S = H_ * P_ * Ht + R_;
  const MatrixXd K = P_ * Ht * S.inverse();

  // new state
  x_ += K * y;
  const MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * update the state by using Extended Kalman Filter equations
   */
  const double px = x_[0];
  const double py = x_[1];
  const double vx = x_[2];
  const double vy = x_[3];
  const double sq_sum = sqrt(pow(px, 2) + pow(py, 2));
  VectorXd hx = VectorXd(3);
  hx << sq_sum,
        atan2(py, px),
        (px * vx + py * vy) / sq_sum;
  VectorXd y = z - hx;
  // normalize angle y(1) between [-pi, pi]
  if (y(1) > M_PI)
    y(1) -= 2 * M_PI;
  else if (y(1) < -M_PI)
    y(1) += 2 * M_PI;
  const MatrixXd Ht = H_.transpose();
  const MatrixXd S = H_ * P_ * Ht + R_;
  const MatrixXd K = P_ * Ht * S.inverse();

  // new state
  x_ += K * y;
  const MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
}
