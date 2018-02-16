/**TODO:  Add copyright*/

#include <SmartPeak/algorithm/ChromatogramSimulator.h>
#include <SmartPeak/algorithm/PeakSimulator.h>
#include <SmartPeak/algorithm/EMGModel.h>

#include <vector>
#include <algorithm>
#include <iostream>

namespace SmartPeak
{
  ChromatogramSimulator::ChromatogramSimulator()
  {        
  }

  ChromatogramSimulator::~ChromatogramSimulator()
  {
  }
  
  void ChromatogramSimulator::joinPeakWindows(
    PeakSimulator& peak_left, PeakSimulator& peak_right) const
  {
    // Check order of left and right peaks
    if (peak_left.getWindowStart() > peak_right.getWindowStart() &&
      peak_left.getWindowEnd() > peak_right.getWindowEnd())
    {  // peaks are swapped
      std::cout << "Left and right peaks are swapped!" << std::endl;
      std::swap(peak_left, peak_right);
    }
    
    const double x_delta = peak_right.getWindowStart() - peak_left.getWindowEnd();
    const double y_delta = peak_right.getBaselineLeft() - peak_left.getBaselineRight();
    if (x_delta >= 0.0 && y_delta <= 0.0)
    {  
      // Non overlapping windows; Left baseline is higher
      // increase the right peak baseline to match the left peak baseline
      peak_right.setBaselineLeft(peak_left.getBaselineRight()); 
      // extend left baseline to right baseline using the left peak sample rate
      peak_left.setWindowEnd(peak_right.getWindowStart());
    }
    else if (x_delta >= 0.0 && y_delta > 0.0)
    {
      // Non overlapping windows; Left baseline is lower
      // increase the left peak baseline to match the right peak baseline
      peak_left.setBaselineRight(peak_right.getBaselineLeft());
      // extend the left baseline using the left peak sample rate
      peak_left.setWindowEnd(peak_right.getWindowStart());
    }
    else if (x_delta < 0.0 && y_delta <= 0.0)
    {
      // Overlapping windows; Left baseline is higher
      // increase the right peak baseline to match the left peak baseline
      peak_right.setBaselineLeft(peak_left.getBaselineRight()); 
      // shorten the right baseline
      peak_right.setWindowStart(peak_left.getWindowEnd());
    }
    else if (x_delta < 0.0 && y_delta > 0.0)
    {
      // Overlapping windows; Right baseline is higher
      // increase the left peak baseline to match the right peak baseline
      peak_left.setBaselineRight(peak_right.getBaselineLeft());
      // shorten the right baseline
      peak_right.setWindowStart(peak_left.getWindowEnd());
    }    
  }

  void ChromatogramSimulator::simulateChromatogram(std::vector<double>& x_O, std::vector<double>& y_O,
    const std::vector<PeakSimulator>& peaks, const std::vector<EMGModel>& emgs) const
  {
    // check vector lengths
    if (peaks.size() != emgs.size())
    {
      std::cout << "Length of peaks vectors is not equal to length of EMGs vector!" << std::endl;
      std::cout << "There are " << peaks.size() << " peaks and " << emgs.size() << " EMGs." << std::endl;
      return;
    }

    // clear any potential input in x and y vectors
    x_O.clear();
    y_O.clear();

    // Order the list of peaks from lowest to highest emg_mu
    std::vector<std::pair<PeakSimulator, EMGModel>> peak_emg_pairs;
    for (int i=0; i<emgs.size(); ++i)
    {
      const std::pair<PeakSimulator, EMGModel> peak_emg(peaks[i], emgs[i]);
      peak_emg_pairs.push_back(peak_emg);
    }
    std::sort(peak_emg_pairs.begin(), peak_emg_pairs.end(),
      [](std::pair<PeakSimulator, EMGModel> lhs, std::pair<PeakSimulator, EMGModel> rhs)
      {
        return lhs.second.getMu() < rhs.second.getMu(); //ascending order
      }
    );
    // std::vector<PeakSimulator> peaks_sorted;
    // std::vector<EMGModel> emgs_sorted;
    // for (int i=0; i<peak_emg_pairs.size(); ++i)
    // {
    //   peaks_sorted.push_back(peak_emg_pairs.first);
    //   emgs_sorted.push_back(peak_emg_pairs.second);
    // }

    // Add the peaks in order
    for (int i=1; i<peak_emg_pairs.size(); ++i)
    {      
      // join peaks

      // // make the peak
      // std::vector<double> x, y;
      // peak_emg_pairs[i].first.simulatePeak(x, y, peak_emg_pairs[i].second);

      // // extend the chromatogram
      // v.reserve(v.size() + distance(v_prime.begin(),v_prime.end()));
      // v.insert(v.end(),v_prime.begin(),v_prime.end());

    }
    
  }
}