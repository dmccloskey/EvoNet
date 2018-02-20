/**TODO:  Add copyright*/

#include <SmartPeak/ml/ELU.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#include <cmath>

namespace SmartPeak
{
  ELU::ELU()
  {        
  }

  ELU::ELU(const double& alpha)
  {
    alpha_ = alpha;
  }

  ELU::~ELU()
  {
  }

  void ELU::setAlpha(const double& alpha)
  {
    alpha_ = alpha;
  }
  double ELU::getAlpha() const
  {
    return alpha_;
  }

  double ELU::fx(const double& x_I) const
  {
    double y_O = (x_I > 0.0) ? x_I : alpha_ * (std::exp(x_I) - 1);
    return y_O;
  }

  double ELU::dfx(const double& x_I) const
  {
    double y_O = (x_I > 0.0) ? 1.0: fx(x_I) + alpha_;
    return y_O;
  }
}