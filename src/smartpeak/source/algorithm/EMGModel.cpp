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

  void EMGModel::setH(const double& h)
  {
    emg_h_ = h;
  }
  double EMGModel::getH() const
  {
    return emg_h_;
  }

  void EMGModel::setTau(const double& tau)
  {
    emg_tau_ = tau;
  }
  double EMGModel::getTau() const
  {
    return emg_tau_;
  }

  void EMGModel::setMu(const double& mu)
  {
    emg_mu_ = mu;
  }
  double EMGModel::getMu() const
  {
    return emg_mu_;
  }

  void EMGModel::setSigma(const double& sigma)
  {
    emg_sigma_ = sigma;
  }
  double EMGModel::getSigma() const
  {
    return emg_sigma_;
  }

  double EMGModel::z_(const double& x_I) const
  {
    double z = 1/std::sqrt(2)*(emg_sigma_/emg_tau_ - (x_I-emg_mu_)/emg_sigma_);
    return z;
  }
}