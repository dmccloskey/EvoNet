/**TODO:  Add copyright*/

#include <SmartPeak/algorithm/PeakSimulator.h>
#include <SmartPeak/algorithm/EMGModel.h>

#include <vector>
#include <random>

namespace SmartPeak
{
  PeakSimulator::PeakSimulator()
  {        
  }

  PeakSimulator::~PeakSimulator()
  {
  }

  void PeakSimulator::setStepSizeMu(const double& step_size_mu)
  {
    step_size_mu_ = step_size_mu;
  }
  double PeakSimulator::getStepSizeMu() const
  {
    return step_size_mu_;
  }

  void PeakSimulator::setStepSizeSigma(const double& step_size_sigma)
  {
    step_size_sigma_ = step_size_sigma;
  }
  double PeakSimulator::getStepSizeSigma() const
  {
    return step_size_sigma_;
  }

  void PeakSimulator::setWindowStart(const double& window_start)
  {
    window_start_ = window_start;
  }
  double PeakSimulator::getWindowStart() const
  {
    return window_start_;
  }

  void PeakSimulator::setWindowEnd(const double& window_end)
  {
    window_end_ = window_end;
  }
  double PeakSimulator::getWindowEnd() const
  {
    return window_end_;
  }

  void PeakSimulator::setNoiseMu(const double& noise_mu)
  {
    noise_mu_ = noise_mu;
  }
  double PeakSimulator::getNoiseMu() const
  {
    return noise_mu_;
  }

  void PeakSimulator::setNoiseSimga(const double& noise_sigma)
  {
    noise_sigma_ = noise_sigma;
  }
  double PeakSimulator::getNoiseSigma() const
  {
    return noise_sigma_;
  }

  void PeakSimulator::setBaselineLeft(const double& baseline_left)
  {
    baseline_left_ = baseline_left;
  }
  double PeakSimulator::getBaselineLeft() const
  {
    return baseline_left_;
  }

  void PeakSimulator::setBaselineRight(const double& baseline_right)
  {
    baseline_right_ = baseline_right;
  }
  double PeakSimulator::getBaselineRight() const
  {
    return baseline_right_;
  }

  void PeakSimulator::setSaturationLimit(const double& saturation_limit)
  {
    saturation_limit_ = saturation_limit;
  }
  double PeakSimulator::getSaturationLimit() const
  {
    return saturation_limit_;
  }

  std::vector<double> PeakSimulator::generateRangeWithNoise(
    const double& start, const double& step_mu, const double& step_sigma, const double& end)
  {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{step_mu, step_sigma};
    std::vector<double> array;
    double value = start;
    while(value <= end)
    {
      array.push_back(value);
      value += d(gen); // could recode to better handle rounding errors
    }
    return array;
  }

  std::vector<double> PeakSimulator::addNoise(
    const std::vector<double>& array_I,
      const double& mean, const double& std_dev)
  {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{mean, std_dev};
    std::vector<double> array;
    // add noise to a new array
    for (auto value: array_I)
    {
      array.push_back(value + d(gen));
    }
    return array;
  }

  std::vector<double> PeakSimulator::addBaseline(
      const std::vector<double>& x_I,
      const std::vector<double>& y_I,
      const double& baseline_left, const double& baseline_right,
      const double& peak_apex)
  {
    std::vector<double> array;
    for (int i = 0; i < x_I.size(); ++i)
    {
      if (x_I[i] <= peak_apex)
      {
        const double value = std::max(baseline_left, y_I[i]);
        array.push_back(value);
      }
      else
      {
        const double value = std::max(baseline_right, y_I[i]);
        array.push_back(value);
      }
    }
    return array;
  }

  std::vector<double> flattenPeak(
      const std::vector<double>& array_I,
      const double& saturation_limit)
  {
    std::vector<double> array;
    for (auto value: array_I)
    {
      double val = (value > saturation_limit) ? saturation_limit: value;
      array.push_back(val);
      
    }
    return array;
  }

  void PeakSimulator::simulatePeak(
    std::vector<double>& x_O, std::vector<double>& y_O, 
    const EMGModel& emg) const
  {
    x_O.clear();
    y_O.clear();

    // make the time array
    x_O = generateRangeWithNoise(window_start_, step_size_mu_, step_size_sigma_, window_end_);
    // make the intensity array
    for (double x: x_O)
    {
      y_O.push_back(emg.PDF(x));
    }
    // add saturation limit
    y_O = flattenPeak(y_O, saturation_limit_);
    // add a baseline to the intensity array
    y_O = addBaseline(x_O, y_O, baseline_left_, baseline_right_, emg.getMu());
    // add noise to the intensity array
    y_O = addNoise(y_O, noise_mu_, noise_sigma_);
  }
}