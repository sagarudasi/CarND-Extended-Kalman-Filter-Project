#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rms_error(4);
  rms_error << 0, 0, 0, 0;

  // Validate estimations
  if(estimations.size() == 0){
      cout << "Estimations size is ZERO";
      return rms_error;
  }

  if(estimations.size() != ground_truth.size()){
      cout << "Estimations not equals ground truth";
      return rms_error;
  }

  // Calculate the squared error
  for(unsigned int i = 0; i < estimations.size(); i++){
      VectorXd r = estimations[i] - ground_truth[i];

      r = r.array() * r.array();
      rms_error = rms_error + r;
  }

  // Mean
  rms_error = rms_error / estimations.size();

  // Squared root
  rms_error = rms_error.array().sqrt();

  return rms_error;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    MatrixXd temp_Hj(3, 4);
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    float c1 = px * px + py * py;
    float c2 = sqrt(c1);
    float c3 = (c1 * c1);

    if(fabs(c1) < 0.00001){
        cout << "Division by zero";
        return temp_Hj;
        // c1 = 0.00001;
    }

    temp_Hj << (px/c2), (py/c2), 0, 0,
               -(py/c1), (px/c1), 0, 0,
               py * (vx * py - vy * px)/c3, px * (px * vy - py * vx)/ c3, px/c2, py/c2;

    return temp_Hj;
}
