/**TODO:  Add copyright*/

#ifndef SMARTPEAK_CHROMATOGRAMSIMULATOR_H
#define SMARTPEAK_CHROMATOGRAMSIMULATOR_H

#include <SmartPeak/algorithm/EMGModel.h>
#include <SmartPeak/algorithm/PeakSimulator.h>

namespace SmartPeak
{
  /**
    @brief A class to generate points that represent an LC-MS,
      GC-MS, or HPLC chromatogram
  */
  class ChromatogramSimulator
  {
public:
    ChromatogramSimulator(); ///< Default constructor
    ~ChromatogramSimulator(); ///< Default destructor

    /**
      @brief Simulates a chromatogram.  
      
      The left baseline of the first peak window
        will define the left baseline of the chromatogram, while the right baseline of the
        last peak window will define the right baseline of the chromatogram.  Peaks in the middle
        can overlap, but only the highest intensity of the overlapped peak will be kept similar
        to the behavior captured by the total ion chromatogram (TIC) or extract ion chromatogram
        (XIC).  Peak windows can also be disconnected.  Gaps in peak windows will be filled by
        extending the right baseline of the left most peak to the beginning of the left baseline
        of the right most peak.

      @example
        peak 1: noisy baseline that will extend to the actual first peak
        peak 2: the actual first peak.
        peak 3: next peak that may or may not be baseline seperated from
          the first peak
        ...
        peak n: noise baseline that will extend from the last peak to the
          end of the chromatogram window

      @param[out] x_IO A vector of x values representing time or m/z
      @param[out] y_IO A vector of y values representing the intensity at time t or m/z m
      @param[in] peak list of PeakSimulator classes that will compose the chromatogram
      @param[in] emg list of corresponding EMGModel classes that define each peak
    */ 
    void simulateChromatogram(std::vector<double>& x_O, std::vector<double>& y_O,
      const std::vector<PeakSimulator>& peak, const std::vector<EMGModel>& emg) const;

    /**
      @brief Adds a peak to the chromatogram

      @param[in,out] x_IO A vector of x values representing time or m/z
      @param[in,out] y_IO A vector of y values representing the intensity at time t or m/z m
      @param[in] peak PeakSimulator class
      @param[in] emg EMGModel class
    */ 
    double addPeak(std::vector<double>& x_O, std::vector<double>& y_O,
      const PeakSimulator& peak, const EMGModel& emg) const;

    /**
      @brief Removes points from overlapping peak windows.  The smallest inensity values from overlapping
        peaks will be discarded.

      @param[in,out] x_IO A vector of x values representing time or m/z
      @param[in,out] y_IO A vector of y values representing the intensity at time t or m/z m
    */ 
    double removeOverlappingPoints(std::vector<double>& x_O, std::vector<double>& y_O) const;

    /**
      @brief Joins disconnected peaks.  Diconnnected peak windows

      @param[in,out] x_IO A vector of x values representing time or m/z
      @param[in,out] y_IO A vector of y values representing the intensity at time t or m/z m
    */ 
    double joinDisconnectedPoints(std::vector<double>& x_O, std::vector<double>& y_O) const;

  };
}

#endif //SMARTPEAK_CHROMATOGRAMSIMULATOR_H