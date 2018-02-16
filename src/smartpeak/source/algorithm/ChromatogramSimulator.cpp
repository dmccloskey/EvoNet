/**TODO:  Add copyright*/

#include <SmartPeak/algorithm/ChromatogramSimulator.h>
#include <SmartPeak/algorithm/PeakSimulator.h>
#include <SmartPeak/algorithm/EMGModel.h>

#include <vector>
#include <random>
#include <iostream>

namespace SmartPeak
{
  ChromatogramSimulator::ChromatogramSimulator()
  {        
  }

  ChromatogramSimulator::~ChromatogramSimulator()
  {
  }

  void ChromatogramSimulator::simulateChromatogram(std::vector<double>& x_O, std::vector<double>& y_O,
    const std::vector<PeakSimulator>& peak, const std::vector<EMGModel>& emg) const
  {
    // Order the list of peaks from lowest to highest emg_mu

    // Add the peaks in order

    // Extend the baseline
    // Remove overlapping peaks
    
  }
}