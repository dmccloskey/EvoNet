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

  std::vector<double> PeakSimulator::makeNoise(
    const double& start, const double& stop, const int& n,
    const double& mean, const double& std_dev) const
  {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{mean, std_dev};

    // not the most efficient (2 loops)
    // but should increase maintainability
    std::vector<double> array = linspan(start, stop, n);
    // override the existing values
    for (double& value: array)
    {
      value = d(gen);
    }
    return array;
  }
}