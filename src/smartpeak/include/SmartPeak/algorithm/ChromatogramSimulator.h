/**TODO:  Add copyright*/

#ifndef SMARTPEAK_CHROMATOGRAMSIMULATOR_H
#define SMARTPEAK_CHROMATOGRAMSIMULATOR_H

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
      @brief Adds a peak to the chromatogram

      @param[in] x_I X value of the EMG PDF

      @returns Y value of the EMG PDF.
    */ 
    double PDF(const double& x_I) const;

  };
}

#endif //SMARTPEAK_CHROMATOGRAMSIMULATOR_H