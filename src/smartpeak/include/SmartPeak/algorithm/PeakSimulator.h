/**TODO:  Add copyright*/

#ifndef SMARTPEAK_PEAKSIMULATOR_H
#define SMARTPEAK_PEAKSIMULATOR_H

#include <SmartPeak/algorithm/EMGModel.h>

#include <vector>
#include <random>

namespace SmartPeak
{
  /**
    @brief Peak simulator.

    This class generates a chromatogram or spectrum peak.  The peak is modeled
      after an exponentially modified gaussian (EMG).

    References:
    Kalambet, Y.; Kozmin, Y.; Mikhailova, K.; Nagaev, I.; Tikhonov, P. (2011).
      "Reconstruction of chromatographic peaks using the exponentially modified Gaussian function". 
      Journal of Chemometrics. 25 (7): 352. doi:10.1002/cem.1343
  */
  class PeakSimulator: public EMGModel
  {
    /**
    Notes on potential optimizations:
    1. make a virtual class called DataSimulator
    2. make a virtual class called simulate
    3. make a virtual class called addNoise
    4. setters/getters would be unique to each derived class
    */
public:
    PeakSimulator(); ///< Default constructor
    ~PeakSimulator(); ///< Default destructor
 
    /**
      @brief simulates two vector of points that correspond to x and y values that
        represent a peak

      @param[out] x_IO A vector of x values representing time or m/z
      @param[out] y_IO A vector of y values representing the intensity at time t or m/z m
    */ 
    void simulatePeak(std::vector<double>& x_O, std::vector<double>& y_O) const;
 
    /**
      @brief Generates a range of values

      @param[in] start Range start
      @param[in] step Range step
      @param[in] end Range end

      @returns A vector of values from range start to end.
    */ 
    std::vector<double> generateRange(const double& start, const double& step, const double& end) const;
 
    /**
      @brief Generates a linear span of values

      @param[in] start Range start
      @param[in] step Range step
      @param[in] n Number of values in the range

      @returns A vector of values from range start to end.
    */ 
    std::vector<double> linspan(const double& start, const double& stop, const int& n) const;
 
    /**
      @brief Add random noise from a normal distribution to a vector of values

      @param[in] array_I Vector of values to add random noise
      @param[in] mean Mean of the normal distribution
      @param[in] std_dev Standard Deviation of the normal distribution

      @returns A vector of values with added random noise.
    */ 
    std::vector<double> addNoise(
      const std::vector<double>& array_I,
      const double& mean, const double& std_dev) const;
 
    /**
      @brief Add a y offset (i.e., baseline) to a vector of values

      @param[in] array_I Vector of values to add random noise
      @param[in] baseline_left Left baseline offset
      @param[in] baseline_right Right baseline offset

      @returns A vector of values with added random noise.
    */ 
    std::vector<double> addBaseline(
      const std::vector<double>& array_I,
      const double& baseline_left, const double& baseline_right) const;

    void setStepSize(const double& step_size); ///< step_size setter
    double getStepSize() const; ///< step_size getter

    void setWindowStart(const double& window_start); ///< window_start setter
    double getWindowStart() const; ///< window_start getter

    void setWindowEnd(const double& window_end); ///< window_end setter
    double getWindowEnd() const; ///< window_end getter

    void setNoiseMu(const double& noise_mu); ///< noise_mu setter
    double getNoiseMu() const; ///< noise_mu getter

    void setNoiseSimga(const double& noise_sigma); ///< noise_sigma setter
    double getNoiseSigma() const; ///< noise_sigma getter

private:
    int step_size_; ///< The spacing between points
    int window_start_; ///< Peak window start
    int window_end_; ///< Peak window end
    double noise_mu_; ///< Mean of random noise generated from a normal distribution
    double noise_sigma_; ///< Standard deviation of random noise generated from a normal distribution
    double baseline_left_; ///< Height of the left baseline
    double baseline_right_; ///< Height of the right baseline

  };
}

#endif //SMARTPEAK_PEAKSIMULATOR_H