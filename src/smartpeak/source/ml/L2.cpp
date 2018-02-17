/**TODO:  Add copyright*/

#include <SmartPeak/ml/L2.h>

#include <vector>
#include <cmath>

namespace SmartPeak
{
  L2::L2()
  {        
  }

  L2::~L2()
  {
  }

  double L2::E(const std::vector<double>& y_pred, const std::vector<double>& y_true) const
  {
    double y_O = 0.0;
    for (int i=0; i<y_pred.size(); ++i)
    {
      y_O += std::pow(y_true[i] - y_pred[i], 2);
    }
    y_O = y_O * 0.5; // modified to simplify the derivative
    return y_O;
  }

  double L2::dE(const double& y_pred, const double& y_true) const
  {
    double y_O = y_true - y_pred; // modified to exclude the 0.5
    return y_O;
  }
}