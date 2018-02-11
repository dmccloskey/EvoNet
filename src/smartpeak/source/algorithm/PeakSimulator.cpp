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

  std::vector<double> PeakSimulator::generateRange(const double& start, const double& step, const double& end) const
  {
    std::vector<double> array;
    double value = start;
    while(value <= end)
    {
      array.push_back(value);
      value += step; // could recode to better handle rounding errors
    }
    return array;
  }

  std::vector<double> PeakSimulator::generateRangeWithNoise(
    const double& start, const double& step_mu, const double& step_sigma, const double& end) const
  {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{mean, std_dev};
    std::vector<double> array;
    double value = start;
    while(value <= end)
    {
      array.push_back(value);
      value = value + step + d(gen); // could recode to better handle rounding errors
    }
    return array;
  }

  std::vector<double> PeakSimulator::linspan(const double& start, const double& stop, const int& n) const
  {
    const double step = (stop-start) / (n-1);
    std::vector<double> array = generateRange(start, step, stop);
    return array;
  }

  std::vector<double> PeakSimulator::addNoise(
    const std::vector<double>& array_I,
      const double& mean, const double& std_dev) const
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
      const std::vector<double>& array_I,
      const double& baseline_left, const double& baseline_right) const
  {
    std::vector<double> array;
    // add noise to a new array
    for (auto value: array_I)
    {
      array.push_back(value + baseline_left);
    }
    return array;

  }

  void  PeakSimulator::simulatePeak(
    std::vector<double>& x_O, std::vector<double>& y_O) const
  {
    x_O.clear();
    y_O.clear();

    // make the time array
    x_O = generateRangeWithNoise(window_start_, step_size_mu_, step_size_sigma_, window_end_);
    // make the intensity array
    for (double x: x_O)
    {
      y_O.push_back(x);
    }
    // add noise to the intensity array
    y_O.addNoise(y_O, noise_mu_, noise_sigma_);
    // add a baseline to the intensity array
    y_O.addBaseline(y_O, baseline_left_, baseline_right_);
  }
}