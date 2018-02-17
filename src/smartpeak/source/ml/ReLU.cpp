/**TODO:  Add copyright*/

#include <SmartPeak/ml/ReLU.h>

#include <cmath>

namespace SmartPeak
{
  ReLU::ReLU()
  {        
  }

  ReLU::~ReLU()
  {
  }

  double ReLU::fx(const double& x_I) const
  {
    double y_O = (x_I > 0.0) ? x_I: 0.0;
    return y_O;
  }

  double ReLU::dfx(const double& x_I) const
  {
    double y_O = (x_I > 0.0) ? 1.0: 0.0;
    return y_O;
  }
}