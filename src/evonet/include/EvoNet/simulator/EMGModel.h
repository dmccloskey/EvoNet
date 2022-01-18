/**TODO:  Add copyright*/

#ifndef EVONET_EMGMODEL_H
#define EVONET_EMGMODEL_H

//.cpp
#include <cmath>

namespace EvoNet
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
	template <typename TensorT>
	class EMGModel
	{
		/**
		Notes on potential optimizations:
		1. make a virtual class called StatisticalModel
		2. make a virtual class called PDF
		3. make a virtual class called CDF
		4. setters/getters would be unique to each derived class
		*/
	public:
		EMGModel() = default; ///< Default constructor
		EMGModel(const TensorT& h,
			const TensorT& tau,
			const TensorT& mu,
			const TensorT& sigma); ///< Explicit constructor
		~EMGModel() = default; ///< Default destructor

		void setH(const TensorT& h); ///< EMG h setter
		TensorT getH() const; ///< EMG h getter

		void setTau(const TensorT& tau); ///< EMG tau setter
		TensorT getTau() const; ///< EMG tau getter

		void setMu(const TensorT& mu); ///< EMG mu setter
		TensorT getMu() const; ///< EMG mu getter

		void setSigma(const TensorT& sigma); ///< EMG sigma setter
		TensorT getSigma() const; ///< EMG sigma getter

		/**
			@brief Calculates points from an EMG PDF

			@param[in] x_I X value of the EMG PDF

			@returns Y value of the EMG PDF.
		*/
		TensorT PDF(const TensorT& x_I) const;

	protected:
		/**
			@brief Calculates points from an EMG PDF using method 1

			@param[in] x_I X value of the EMG PDF

			@returns Y value of the EMG PDF.
		*/
		TensorT EMGPDF1_(const TensorT& x_I) const;

		/**
			@brief Calculates points from an EMG PDF using method 2

			@param[in] x_I X value of the EMG PDF

			@returns Y value of the EMG PDF.
		*/
		TensorT EMGPDF2_(const TensorT& x_I) const;

		/**
			@brief Calculates points from an EMG PDF using method 3

			@param[in] x_I X value of the EMG PDF

			@returns Y value of the EMG PDF.
		*/
		TensorT EMGPDF3_(const TensorT& x_I) const;

		/**
			@brief Calculates the parameter z, which is used to decide
				which formulation of the EMG PDF to use for calculation.

			@param[in] x_I X value of the EMG PDF

			@returns z parameter.
		*/
		TensorT z_(const TensorT& x_I) const;

	private:
		TensorT emg_h_ = (TensorT)1.0; ///< Amplitude of the Gaussian peak
		TensorT emg_tau_ = (TensorT)0.1; ///< Exponential relaxation time 
		TensorT emg_mu_ = (TensorT)0.0; ///< Mean of the EMG
		TensorT emg_sigma_ = (TensorT)1.0; ///< Standard deviation of the EGM

	};


	template <typename TensorT>
	EMGModel<TensorT>::EMGModel(const TensorT& h,
		const TensorT& tau,
		const TensorT& mu,
		const TensorT& sigma)
	{
		emg_h_ = h;
		emg_tau_ = tau;
		emg_mu_ = mu;
		emg_sigma_ = sigma;
	}

	template <typename TensorT>
	void EMGModel<TensorT>::setH(const TensorT& h)
	{
		emg_h_ = h;
	}
	template <typename TensorT>
	TensorT EMGModel<TensorT>::getH() const
	{
		return emg_h_;
	}

	template <typename TensorT>
	void EMGModel<TensorT>::setTau(const TensorT& tau)
	{
		emg_tau_ = tau;
	}
	template <typename TensorT>
	TensorT EMGModel<TensorT>::getTau() const
	{
		return emg_tau_;
	}

	template <typename TensorT>
	void EMGModel<TensorT>::setMu(const TensorT& mu)
	{
		emg_mu_ = mu;
	}
	template <typename TensorT>
	TensorT EMGModel<TensorT>::getMu() const
	{
		return emg_mu_;
	}

	template <typename TensorT>
	void EMGModel<TensorT>::setSigma(const TensorT& sigma)
	{
		emg_sigma_ = sigma;
	}
	template <typename TensorT>
	TensorT EMGModel<TensorT>::getSigma() const
	{
		return emg_sigma_;
	}

	template <typename TensorT>
	TensorT EMGModel<TensorT>::z_(const TensorT& x_I) const
	{
		TensorT z = TensorT(1 / std::sqrt(2)*(emg_sigma_ / emg_tau_ - (x_I - emg_mu_) / emg_sigma_));
		return z;
	}

	template <typename TensorT>
	TensorT EMGModel<TensorT>::EMGPDF1_(const TensorT& x_I) const
	{
		const TensorT PI = TensorT(3.141592653589793);
		const TensorT term1a = TensorT(emg_h_ * emg_sigma_ / emg_tau_ * std::sqrt(PI / 2));
		const TensorT term2a = TensorT(0.5*std::pow(emg_sigma_ / emg_tau_, 2) - (x_I - emg_mu_) / emg_tau_);
		const TensorT term3a = TensorT(1 / sqrt(2)*(emg_sigma_ / emg_tau_ - (x_I - emg_mu_) / emg_sigma_));
		const TensorT y = TensorT(term1a * std::exp(term2a)*std::erfc(term3a));
		return y;
	}

  template <typename TensorT>
  TensorT EMGModel<TensorT>::EMGPDF2_(const TensorT& x_I) const
  {
    const TensorT PI = TensorT(3.141592653589793);
    const TensorT term1a = TensorT(emg_h_ * emg_sigma_ / emg_tau_ * std::sqrt(PI / 2));
    const TensorT term2b = TensorT(-0.5*std::pow((x_I - emg_mu_) / emg_sigma_, 2));
    const TensorT term3a = TensorT(1 / sqrt(2)*(emg_sigma_ / emg_tau_ - (x_I - emg_mu_) / emg_sigma_));
    const TensorT y = TensorT(term1a * std::exp(term2b)*std::exp(std::pow(term3a, 2))*std::erfc(term3a));
    return y;
  }

	template <typename TensorT>
	TensorT EMGModel<TensorT>::EMGPDF3_(const TensorT& x_I) const
	{
		const TensorT term1b = TensorT(emg_h_);
		const TensorT term2b = TensorT(-0.5*std::pow((x_I - emg_mu_) / emg_sigma_, 2));
		const TensorT term3b = TensorT(1 - (x_I - emg_mu_)*emg_tau_ / std::pow(emg_sigma_, 2));
		const TensorT y = TensorT(term1b * std::exp(term2b) / term3b);
		return y;
	}

	template <typename TensorT>
	TensorT EMGModel<TensorT>::PDF(const TensorT& x_I) const
	{
		const TensorT z = z_(x_I);
		TensorT y = (TensorT)0;
		if (z < 0)
		{
			y = EMGPDF1_(x_I);
		}
		else if (z >= 0 && z <= 6.71e7)
		{
			y = EMGPDF2_(x_I);
		}
		else if (z > 6.71e7)
		{
			y = EMGPDF3_(x_I);
		}
		return y;
	}
}

#endif //EVONET_EMGMODEL_H