/**TODO:  Add copyright*/

#include <SmartPeak/ml/EuclideanDistance.h>

#include <vector>
#include <cmath>

namespace SmartPeak
{
  EuclideanDistance::EuclideanDistance()
  {        
  }

  EuclideanDistance::~EuclideanDistance()
  {
  }

  double EuclideanDistance::E(const std::vector<double>& y_pred, const std::vector<double>& y_true) const
  {
    double y_O = 0.0;
    for (int i=0; i<y_pred.size(); ++i)
    {
      y_O += std::pow(y_true[i] - y_pred[i], 2);
    }
    y_O = std::sqrt(y_O);
    return y_O;
  }

  double EuclideanDistance::dE(const double& y_pred, const double& y_true) const
  {
    double y_O = (y_true - y_pred) / std::sqrt(std::pow(y_true - y_pred, 2)); 
    return y_O;
  }
}