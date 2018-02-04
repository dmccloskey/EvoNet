/**TODO:  Add copyright*/

#ifndef SMARTPEAK_EMGMODEL_H
#define SMARTPEAK_EMGMODEL_H

namespace SmartPeak
{
  /**
    @brief A class to generate points following an EMG distribution.

    References:
    Kalambet, Y.; Kozmin, Y.; Mikhailova, K.; Nagaev, I.; Tikhonov, P. (2011).
      "Reconstruction of chromatographic peaks using the exponentially modified Gaussian function". 
      Journal of Chemometrics. 25 (7): 352. doi:10.1002/cem.1343
    Delley, R (1985). 
      "Series for the Exponentially Modified Gaussian Peak Shape". 
      Anal. Chem. 57: 388. doi:10.1021/ac00279a094.
    Dyson, N. A. (1998). 
      Chromatographic Integration Methods. 
      Royal Society of Chemistry, Information Services. p. 27. ISBN 9780854045105. Retrieved 2015-05-15.
  */
  class EMGModel
  {
public:
    EMGModel(); ///< Default constructor
    ~EMGModel(); ///< Default destructor

    void setH(const double& h); ///< EMG h setter
    double getH() const; ///< EMG h getter

    void setTau(const double& tau); ///< EMG tau setter
    double getTau() const; ///< EMG tau getter

    void setMu(const double& mu); ///< EMG mu setter
    double getMu() const; ///< EMG mu getter

    void setSigma(const double& sigma); ///< EMG sigma setter
    double getSigma() const; ///< EMG sigma getter

protected: 
    /**
      @brief Calculates points from an EMG PDF using method 1

      @param[out] x_I X value of the EMG PDF

      @returns Y value of the EMG PDF.
    */ 
    double EMGPDF1_(const double& x_I) const;
 
    /**
      @brief Calculates points from an EMG PDF using method 2

      @param[out] x_I X value of the EMG PDF

      @returns Y value of the EMG PDF.
    */ 
    double EMGPDF2_(const double& x_I) const;
 
    /**
      @brief Calculates points from an EMG PDF using method 3

      @param[out] x_I X value of the EMG PDF

      @returns Y value of the EMG PDF.
    */ 
    double EMGPDF3_(const double& x_I) const;
 
    /**
      @brief Calculates the parameter z, which is used to decide
        which formulation of the EMG PDF to use for calculation.

      @param[out] x_I X value of the EMG PDF

      @returns z parameter.
    */ 
    double z_(const double& x_I) const;


private:
    double emg_h_; ///< Amplitude of the Gaussian peak
    double emg_tau_; ///< Exponential relaxation time 
    double emg_mu_; ///< Mean of the EMG
    double emg_sigma_; ///< Standard deviation of the EGM

  };
}

#endif //SMARTPEAK_EMGMODEL_H