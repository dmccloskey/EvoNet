/**TODO:  Add copyright*/

#include <SmartPeak/algorithm/PeakSimulator.h>

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

  void PeakSimulator::setStepSize(const double& step_size)
  {
    step_size_ = step_size;
  }
  double PeakSimulator::getStepSize() const
  {
    return step_size_;
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
}