/**TODO:  Add copyright*/

#include <SmartPeak/algorithm/EMGModel.h>

#include <cmath>

namespace SmartPeak
{
  EMGModel::EMGModel()
  {        
  }

  EMGModel::~EMGModel()
  {
  }

  void setH(const double& h) const
  {
    emg_h_ = h;
  }
  double getH() const
  {
    return emg_h_;
  }

  void setTau(const double& tau) const
  {
    emg_tau_ = tau;
  }
  double getTau() const
  {
    return emg_tau_;
  }

  void setMu(const double& mu) const
  {
    emg_mu_ = mu;
  }
  double getMu() const
  {
    return emg_mu_;
  }

  void setSigma(const double& sigma) const
  {
    emg_sigma_ = sigma;
  }
  double getSigma() const
  {
    return emg_sigma_;
  }

  double z_(const double& x_I) const
  {
    double z = 1/std::sqrt(2)*(emg_sigma_/emg_tau_ - (x_I-emg_mu)/emg_sigma_);
    return z;
  }
}