#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
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
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  // Uncertainity covariance
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1000, 0, 0, 0,
        0, 1000, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;

  // Process noise
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << 0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;

  // State transition matrix
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
        0, 1, 0, 1,
        0, 0, 1, 0,
        0, 0, 0, 1;

  // Measurement update functions
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  Hj_ << 0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {

     float Px;
     float Py;

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float rho = measurement_pack.raw_measurements_[0];
      float theta = measurement_pack.raw_measurements_[1];

      if (fabs(theta) > M_PI){
          theta -= round(theta /(2.0 * M_PI)) * (2.0 * M_PI);
      }
      float rho_dot = measurement_pack.raw_measurements_[2];
      Px = rho * cos(theta);
      Py = rho * sin(theta);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      Px = measurement_pack.raw_measurements_[0];
      Py = measurement_pack.raw_measurements_[1];
    }

    // Avoid division by 0
    if(fabs(Px) < 0.00001)
        Px = 0.00001;

    if(fabs(Py) < 0.00001)
        Py = 0.00001;

    ekf_.x_ << Px, Py, 0, 0;

    this->previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

   // Calculate delta t

   float dt = (measurement_pack.timestamp_ - this->previous_timestamp_) / 1000000.0;
   this->previous_timestamp_ = measurement_pack.timestamp_;

   float dt_2 = dt * dt;
   float dt_3 = dt_2 * dt;
   float dt_4 = dt_3 * dt;

   // Update the state transition matrix for new dt
   ekf_.F_ << 1, 0, dt, 0,
              0, 1, 0, dt,
              0, 0, 1, 0,
              0, 0, 0, 1;

  // Given noise
  float noise_ax = 9;
  float noise_ay = 9;

  // Update the Process covariance matrix for new dt
  ekf_.Q_ << (dt_4/4) * noise_ax, 0, (dt_3/2) * noise_ax, 0,
             0, (dt_4/4) * noise_ay, 0, (dt_3/2) * noise_ay,
             (dt_3/2) * noise_ax, 0, dt_2 * noise_ax, 0,
             0, (dt_3/2) * noise_ay, 0, dt_2 * noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.R_ = this->R_radar_;
    this->Hj_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.H_ = this->Hj_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.R_ = this->R_laser_;
    ekf_.H_ = this->H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
