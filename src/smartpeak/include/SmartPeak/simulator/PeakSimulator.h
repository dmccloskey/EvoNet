/**TODO:  Add copyright*/

#ifndef SMARTPEAK_PEAKSIMULATOR_H
#define SMARTPEAK_PEAKSIMULATOR_H

// .h
#include <SmartPeak/simulator/EMGModel.h>
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
	template <typename TensorT>
	class PeakSimulator
	{
		/**
		Notes on potential optimizations:
		1. make a virtual class called DataSimulator
		2. make a virtual class called simulate
		3. make a virtual class called addNoise
		4. setters/getters would be unique to each derived class
		*/
	public:
		PeakSimulator() = default; ///< Default constructor
		PeakSimulator(const TensorT& step_size_mu,
			const TensorT& step_size_sigma,
			const TensorT& window_start,
			const TensorT& window_end,
			const TensorT& noise_mu,
			const TensorT& noise_sigma,
			const TensorT& baseline_left,
			const TensorT& baseline_right,
			const TensorT& saturation_limit); ///< Explicit constructor

		~PeakSimulator() = default; ///< Default destructor

		/**
			@brief calculate the points that define the left and right of the "actual" peak
				based on the fitted emg model points and set baselines.

			@param[in] x_IO A vector of x values representing time or m/z
			@param[in] y_IO A vector of y values representing the intensity at time t or m/z m

			@returns std::pair<TensorT, TensorT> of best left and right points for the peak
		*/
		std::pair<TensorT, TensorT> getBestLeftAndRight(std::vector<TensorT>& x_O, std::vector<TensorT>& y_O, const TensorT& rt) const;

		/**
			@brief simulates two vector of points that correspond to x and y values that
				represent a peak

			@param[out] x_IO A vector of x values representing time or m/z
			@param[out] y_IO A vector of y values representing the intensity at time t or m/z m
			@param[in] emg An emg model class
		*/
		void simulatePeak(std::vector<TensorT>& x_O, std::vector<TensorT>& y_O,
			const EMGModel<TensorT>& emg) const;

		/**
			@brief Generates a range of values with noise sampled from a normal distribution

			@param[in] start Range start
			@param[in] step_mu Range mean step
			@param[in] step_sigma Range step standard deviation
			@param[in] end Range end

			@returns A vector of values from range start to end.
		*/
		static std::vector<TensorT> generateRangeWithNoise(
			const TensorT& start, const TensorT& step_mu,
			const TensorT& step_sigma, const TensorT& end);

		/**
			@brief Add random noise from a normal distribution to a vector of values
				to simulate detector noise.

			@param[in,out] array_IO Vector of values to add random noise
			@param[in] mean Mean of the normal distribution
			@param[in] std_dev Standard Deviation of the normal distribution

			@returns A vector of values with added random noise.
		*/
		static void addNoise(
			std::vector<TensorT>& array_IO,
			const TensorT& mean, const TensorT& std_dev);

		/**
			@brief Add a y offset (i.e., baseline) to a vector of values
				to simulate a rise in the baseline.

			@param[in] x_I Vector of time values
			@param[in,out] y_IO Vector of intensity values
			@param[in] baseline_left Left baseline offset
			@param[in] baseline_right Right baseline offse
			@param[in] peak_apex Time to divide left and right peak sides

			@returns A vector of values with added baselines.
		*/
		static void addBaseline(
			const std::vector<TensorT>& x_I,
			std::vector<TensorT>& y_IO,
			const TensorT& baseline_left, const TensorT& baseline_right,
			const TensorT& peak_apex);

		/**
			@brief Flatten the top of a peak to simulate a saturated peak.

			@param[in,out] array_IO Vector of values to add a saturation point to
			@param[in] saturation_limit Saturation limit of the simulated detector

			@returns A vector of values with a simulated saturation point.
		*/
		static void flattenPeak(
			std::vector<TensorT>& array_IO,
			const TensorT& saturation_limit);

		void setStepSizeMu(const TensorT& step_size_mu); ///< step_size_mu setter
		TensorT getStepSizeMu() const; ///< step_size_mu getter

		void setStepSizeSigma(const TensorT& step_size_mu); ///< step_size_sigma setter
		TensorT getStepSizeSigma() const; ///< step_size_sigma getter

		void setWindowStart(const TensorT& window_start); ///< window_start setter
		TensorT getWindowStart() const; ///< window_start getter

		void setWindowEnd(const TensorT& window_end); ///< window_end setter
		TensorT getWindowEnd() const; ///< window_end getter

		void setNoiseMu(const TensorT& noise_mu); ///< noise_mu setter
		TensorT getNoiseMu() const; ///< noise_mu getter

		void setNoiseSimga(const TensorT& noise_sigma); ///< noise_sigma setter
		TensorT getNoiseSigma() const; ///< noise_sigma getter

		void setBaselineLeft(const TensorT& baseline_left); ///< baseline_left setter
		TensorT getBaselineLeft() const; ///< baseline_left getter

		void setBaselineRight(const TensorT& baseline_right); ///< baseline_right setter
		TensorT getBaselineRight() const; ///< baseline_right getter

		void setSaturationLimit(const TensorT& saturation_limit); ///< saturation_limit setter
		TensorT getSaturationLimit() const; ///< saturation_limit getter

	private:
		TensorT step_size_mu_ = (TensorT)1.0; ///< The mean spacing between points
		TensorT step_size_sigma_ = (TensorT)0.001; ///< The standard deviation of spacing between points
		TensorT window_start_ = (TensorT)0.0; ///< Peak window start
		TensorT window_end_ = (TensorT)100.0; ///< Peak window end
		TensorT noise_mu_ = (TensorT)0.0; ///< Mean of random noise generated from a normal distribution
		TensorT noise_sigma_ = (TensorT)1.0; ///< Standard deviation of random noise generated from a normal distribution
		TensorT baseline_left_ = (TensorT)0.0; ///< Height of the left baseline
		TensorT baseline_right_ = (TensorT)0.0; ///< Height of the right baseline
		TensorT saturation_limit_ = (TensorT)1e6; ///< Maximum point height before peak saturation

	};

	template <typename TensorT>
	PeakSimulator<TensorT>::PeakSimulator(const TensorT& step_size_mu,
		const TensorT& step_size_sigma,
		const TensorT& window_start,
		const TensorT& window_end,
		const TensorT& noise_mu,
		const TensorT& noise_sigma,
		const TensorT& baseline_left,
		const TensorT& baseline_right,
		const TensorT& saturation_limit)
	{
		step_size_mu_ = step_size_mu;
		step_size_sigma_ = step_size_sigma;
		window_start_ = window_start;
		window_end_ = window_end;
		noise_mu_ = noise_mu;
		noise_sigma_ = noise_sigma;
		baseline_left_ = baseline_left;
		baseline_right_ = baseline_right;
		saturation_limit_ = saturation_limit;
	}

	template <typename TensorT>
	void PeakSimulator<TensorT>::setStepSizeMu(const TensorT& step_size_mu)
	{
		step_size_mu_ = step_size_mu;
	}
	template <typename TensorT>
	TensorT PeakSimulator<TensorT>::getStepSizeMu() const
	{
		return step_size_mu_;
	}

	template <typename TensorT>
	void PeakSimulator<TensorT>::setStepSizeSigma(const TensorT& step_size_sigma)
	{
		step_size_sigma_ = step_size_sigma;
	}
	template <typename TensorT>
	TensorT PeakSimulator<TensorT>::getStepSizeSigma() const
	{
		return step_size_sigma_;
	}

	template <typename TensorT>
	void PeakSimulator<TensorT>::setWindowStart(const TensorT& window_start)
	{
		window_start_ = window_start;
	}
	template <typename TensorT>
	TensorT PeakSimulator<TensorT>::getWindowStart() const
	{
		return window_start_;
	}

	template <typename TensorT>
	void PeakSimulator<TensorT>::setWindowEnd(const TensorT& window_end)
	{
		window_end_ = window_end;
	}
	template <typename TensorT>
	TensorT PeakSimulator<TensorT>::getWindowEnd() const
	{
		return window_end_;
	}

	template <typename TensorT>
	void PeakSimulator<TensorT>::setNoiseMu(const TensorT& noise_mu)
	{
		noise_mu_ = noise_mu;
	}
	template <typename TensorT>
	TensorT PeakSimulator<TensorT>::getNoiseMu() const
	{
		return noise_mu_;
	}

	template <typename TensorT>
	void PeakSimulator<TensorT>::setNoiseSimga(const TensorT& noise_sigma)
	{
		noise_sigma_ = noise_sigma;
	}
	template <typename TensorT>
	TensorT PeakSimulator<TensorT>::getNoiseSigma() const
	{
		return noise_sigma_;
	}

	template <typename TensorT>
	void PeakSimulator<TensorT>::setBaselineLeft(const TensorT& baseline_left)
	{
		baseline_left_ = baseline_left;
	}
	template <typename TensorT>
	TensorT PeakSimulator<TensorT>::getBaselineLeft() const
	{
		return baseline_left_;
	}

	template <typename TensorT>
	void PeakSimulator<TensorT>::setBaselineRight(const TensorT& baseline_right)
	{
		baseline_right_ = baseline_right;
	}
	template <typename TensorT>
	TensorT PeakSimulator<TensorT>::getBaselineRight() const
	{
		return baseline_right_;
	}

	template <typename TensorT>
	void PeakSimulator<TensorT>::setSaturationLimit(const TensorT& saturation_limit)
	{
		saturation_limit_ = saturation_limit;
	}
	template <typename TensorT>
	TensorT PeakSimulator<TensorT>::getSaturationLimit() const
	{
		return saturation_limit_;
	}

	template <typename TensorT>
	std::vector<TensorT> PeakSimulator<TensorT>::generateRangeWithNoise(
		const TensorT& start, const TensorT& step_mu, const TensorT& step_sigma, const TensorT& end)
	{
		std::random_device rd{};
		std::mt19937 gen{ rd() };

		TensorT step_mu_used = step_mu;
		TensorT step_sigma_used = step_sigma;
		// TODO: improve defaults
		if (step_mu <= (TensorT)0)
		{
			std::cout << "Warning: mean of step size will generate negative values.  A default mean of 1.0 and std_dev of 1e-9 will be used instead." << std::endl;
			step_mu_used = (TensorT)1.0;
			step_sigma_used = (TensorT)0;
		}
		else if (step_mu - (TensorT)5 * step_sigma <= (TensorT)0)
		{
			std::cout << "Warning: mean and std_dev of range step size may generate negative values.  Reduce std_dev to at least 1/5 the mean of the step size.  A default std_dev of 1e-9 will be used instead." << std::endl;
			step_sigma_used = (TensorT)0;
		}

		std::vector<TensorT> array;
		TensorT value = start;
		int cnt = 0;  // checks to ensure that an infinite loop is not run
		if (step_sigma_used > (TensorT)0) {
			std::normal_distribution<> d{ step_mu_used, step_sigma_used };
			while (value <= end || cnt > 1e6)
			{
				array.push_back(value);
				value += d(gen); // could recode to better handle rounding errors
				cnt += 1;
			}
		}
		else {
			while (value <= end || cnt > 1e6)
			{
				array.push_back(value);
				value += step_mu_used; // could recode to better handle rounding errors
				cnt += 1;
			}
		}
		return array;
	}

	template <typename TensorT>
	void PeakSimulator<TensorT>::addNoise(
		std::vector<TensorT>& array_IO,
		const TensorT& mean, const TensorT& std_dev)
	{
		std::random_device rd{};
		std::mt19937 gen{ rd() };
		if (std_dev > 0) {
			std::normal_distribution<> d{ mean, std_dev };
			// add noise to a new array
			for (TensorT& value : array_IO)
			{
				value = value + d(gen);
			}
		}
		else {
			for (TensorT& value : array_IO)
			{
				value = value + mean;
			}
		}
	}

	template <typename TensorT>
	void PeakSimulator<TensorT>::addBaseline(
		const std::vector<TensorT>& x_I,
		std::vector<TensorT>& y_IO,
		const TensorT& baseline_left, const TensorT& baseline_right,
		const TensorT& peak_apex)
	{
		for (int i = 0; i < x_I.size(); ++i)
		{
			if (x_I[i] <= peak_apex)
			{
				y_IO[i] = std::max<TensorT>(baseline_left, y_IO[i]);
			}
			else
			{
				y_IO[i] = std::max<TensorT>(baseline_right, y_IO[i]);
			}
		}
	}

	template <typename TensorT>
	void PeakSimulator<TensorT>::flattenPeak(
		std::vector<TensorT>& array_IO,
		const TensorT& saturation_limit)
	{
		for (TensorT& value : array_IO)
		{
			value = (value > saturation_limit) ? saturation_limit : value;
		}
	}

	template<typename TensorT>
	inline std::pair<TensorT, TensorT> PeakSimulator<TensorT>::getBestLeftAndRight(std::vector<TensorT>& x_O, std::vector<TensorT>& y_O, const TensorT& rt) const
	{
		TensorT best_left = (TensorT)0;
		TensorT best_right = (TensorT)0;

		// iterate from the left
		for (int i = 1; i < x_O.size() - 1; ++i) {
			if (y_O[i] > baseline_left_ + noise_sigma_) {
				best_left = x_O[i - 1];
				break;
			}
			if (x_O[i] > rt)
				break;
		}

		// iterate from the right
		for (int i = x_O.size() - 2; i >= 0; --i) {
			if (y_O[i] > baseline_right_ + noise_sigma_) {
				best_right = x_O[i + 1];
				break;
			}
			if (x_O[i] < rt)
				break;
		}		

		return std::pair<TensorT, TensorT>(best_left, best_right);
	}

	template <typename TensorT>
	void PeakSimulator<TensorT>::simulatePeak(
		std::vector<TensorT>& x_O, std::vector<TensorT>& y_O,
		const EMGModel<TensorT>& emg) const
	{
		x_O.clear();
		y_O.clear();

		// make the time array
		x_O = generateRangeWithNoise(window_start_, step_size_mu_, step_size_sigma_, window_end_);
		// make the intensity array
		for (TensorT x : x_O)
		{
			y_O.push_back(emg.PDF(x));
		}
		// add a baseline to the intensity array
		addBaseline(x_O, y_O, baseline_left_, baseline_right_, emg.getMu());
		// add noise to the intensity array
		addNoise(y_O, noise_mu_, noise_sigma_);
		// add saturation limit
		flattenPeak(y_O, saturation_limit_);
	}
}

#endif //SMARTPEAK_PEAKSIMULATOR_H