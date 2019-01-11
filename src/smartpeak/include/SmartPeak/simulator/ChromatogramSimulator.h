/**TODO:  Add copyright*/

#ifndef SMARTPEAK_CHROMATOGRAMSIMULATOR_H
#define SMARTPEAK_CHROMATOGRAMSIMULATOR_H

// .h
#include <SmartPeak/simulator/EMGModel.h>
#include <SmartPeak/simulator/PeakSimulator.h>
#include <SmartPeak/simulator/DataSimulator.h>

namespace SmartPeak
{
	/**
		@brief A class to generate points that represent an LC-MS,
			GC-MS, or HPLC chromatogram
	*/
	template <typename TensorT>
	class ChromatogramSimulator: public DataSimulator<TensorT>
	{
	public:
		ChromatogramSimulator() = default; ///< Default constructor
		~ChromatogramSimulator() = default; ///< Default destructor

		/**
			@brief Simulates a chromatogram.

			A random number of peaks with random properties are generated and combined into 
				a chromatogram.  Based on the parameters chosen fewer peaks may actually be made.
				This could be caused by neighboring baselines that are higher than the peak.
				The actual best left and right pairs that define the peaks will be returned.

			@param[out] x_noise_IO A vector of x values representing time or m/z
			@param[out] y_noise_IO A vector of y values representing the intensity at time t or m/z m
			@param[out] x_IO A vector of x values representing time or m/z
			@param[out] y_IO A vector of y values representing the intensity at time t or m/z m
			@param[out] peaks_LR A vector of best left and best right pairs
			@param[in] step_size_mu
			@param[in] step_size_sigma
			@param[in] chrom_window_size The lower and upper bounds for the maximum size of the chromatogram
			@param[in] noise_mu
			@param[in] noise_sigma
			@param[in] baseline_height The lower and upper bounds of the baseline heights
			@param[in] n_peaks The number of peaks in the chromatogram
			@param[in] emg_h
			@param[in] emg_tau
			@param[in] emg_mu_offset The lower and upper bounds for the Distance +/- from the peak window center
			@param[in] emg_sigma
		*/
		void simulateChromatogram(std::vector<TensorT>& x_O, std::vector<TensorT>& y_O,
			std::vector<TensorT>& x_noise_O, std::vector<TensorT>& y_noise_O,
			std::vector<std::pair<TensorT, TensorT>>& peaks_LR,
			const std::pair<TensorT, TensorT>& step_size_mu,
			const std::pair<TensorT, TensorT>& step_size_sigma,
			const std::pair<TensorT, TensorT>& chrom_window_size,
			const std::pair<TensorT, TensorT>& noise_mu,
			const std::pair<TensorT, TensorT>& noise_sigma,
			const std::pair<TensorT, TensorT>& baseline_height,
			const std::pair<TensorT, TensorT>& n_peaks,
			const std::pair<TensorT, TensorT>& emg_h,
			const std::pair<TensorT, TensorT>& emg_tau,
			const std::pair<TensorT, TensorT>& emg_mu_offset,
			const std::pair<TensorT, TensorT>& emg_sigma,
			TensorT saturation_limit = 100) const;

		/**
			@brief Makes a chromatogram.

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
			@param[in] peaks list of PeakSimulator classes that will compose the chromatogram
			@param[in] emgs list of corresponding EMGModel classes that define each peak
		*/
		void makeChromatogram(std::vector<TensorT>& x_O, std::vector<TensorT>& y_O, std::vector<std::pair<TensorT, TensorT>>& peaks_LR,
			const std::vector<PeakSimulator<TensorT>>& peaks, const std::vector<EMGModel<TensorT>>& emgs) const;

		/**
			@brief Joins peak windows.

			Overlapping or disconnected peak windows will be joined by extending the highest
				connecting baseline.

			@param[in,out] peak_left Left peak
			@param[in,out] emg_left Left peak EMGModel
			@param[in,out] peak_right Right peak
			@param[in,out] emg_right Right peak EMGModel
		*/
		void joinPeakWindows(
			PeakSimulator<TensorT>& peak_left, EMGModel<TensorT>& emg_left,
			PeakSimulator<TensorT>& peak_right, EMGModel<TensorT>& emg_right) const;

		/**
			@brief Find the overlap between two peak windows.

			The point of overlap between two peaks will be returned.

			@param[in,out] peak_left Left peak
			@param[in,out] emg_left Left peak EMGModel
			@param[in,out] peak_right Right peak
			@param[in,out] emg_right Right peak EMGModel

			@returns overlap The point at which both peaks overlap
		*/
		TensorT findPeakOverlap(
			const PeakSimulator<TensorT>& peak_left, const EMGModel<TensorT>& emg_left,
			const PeakSimulator<TensorT>& peak_right, const EMGModel<TensorT>& emg_right) const;
	};

	template <typename TensorT>
	TensorT ChromatogramSimulator<TensorT>::findPeakOverlap(
		const PeakSimulator<TensorT>& peak_left, const EMGModel<TensorT>& emg_left,
		const PeakSimulator<TensorT>& peak_right, const EMGModel<TensorT>& emg_right) const
	{
		std::vector<TensorT> x_left, y_left, x_right, y_right;
		PeakSimulator<TensorT> peak_l = peak_left;
		PeakSimulator<TensorT> peak_r = peak_right;

		// move windows just to the overlapping region
		peak_l.setWindowStart(peak_r.getWindowStart());
		peak_r.setWindowEnd(peak_l.getWindowEnd());

		// simulate the peaks for the overlapping regions
		peak_l.simulatePeak(x_left, y_left, emg_left);
		peak_r.simulatePeak(x_right, y_right, emg_right);

		// find the highest point where the peaks cross
		TensorT x_overlap = peak_left.getWindowEnd();
		TensorT y_overlap = 0.0;
		for (int i = x_right.size() - 1; i >= 0; --i)
		{  // iterate in reverse order to extend the left peak
			for (int j = x_left.size() - 1; j >= 0; --j)
			{
				if (x_right[i] <= x_left[j] && y_right[i] <= y_left[j])
				{
					if (y_overlap < y_right[i])
					{
						y_overlap = y_right[i];
						x_overlap = x_right[i];
					}
				}
			}
		}
		return x_overlap;
	}

	template <typename TensorT>
	void ChromatogramSimulator<TensorT>::joinPeakWindows(
		PeakSimulator<TensorT>& peak_left, EMGModel<TensorT>& emg_left,
		PeakSimulator<TensorT>& peak_right, EMGModel<TensorT>& emg_right) const
	{
		// Check order of left and right peaks
		if (peak_left.getWindowStart() > peak_right.getWindowStart() &&
			peak_left.getWindowEnd() > peak_right.getWindowEnd())
		{  // peaks are swapped
			std::cout << "Left and right peaks are swapped!" << std::endl;
			std::swap(peak_left, peak_right);
			std::swap(emg_left, emg_right);
		}

		const TensorT x_delta = peak_right.getWindowStart() - peak_left.getWindowEnd();
		const TensorT y_delta = peak_right.getBaselineLeft() - peak_left.getBaselineRight();
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
			// find the overlap
			const TensorT overlap = findPeakOverlap(
				peak_left, emg_left,
				peak_right, emg_right
			);
			peak_right.setWindowStart(overlap);
			peak_left.setWindowEnd(overlap);
		}
		else if (x_delta < 0.0 && y_delta > 0.0)
		{
			// Overlapping windows; Right baseline is higher
			// increase the left peak baseline to match the right peak baseline
			peak_left.setBaselineRight(peak_right.getBaselineLeft());
			// find the overlap
			const TensorT overlap = findPeakOverlap(
				peak_left, emg_left,
				peak_right, emg_right
			);
			peak_right.setWindowStart(overlap);
			peak_left.setWindowEnd(overlap);
		}
	}

	template<typename TensorT>
	inline void ChromatogramSimulator<TensorT>::simulateChromatogram(std::vector<TensorT>& x_O, std::vector<TensorT>& y_O, 
		std::vector<TensorT>& x_noise_O, std::vector<TensorT>& y_noise_O, std::vector<std::pair<TensorT, TensorT>>& peaks_LR,
		const std::pair<TensorT, TensorT>& step_size_mu, const std::pair<TensorT, TensorT>& step_size_sigma,
		const std::pair<TensorT, TensorT>& chrom_window_size, const std::pair<TensorT, TensorT>& noise_mu, const std::pair<TensorT, TensorT>& noise_sigma,
		const std::pair<TensorT, TensorT>& baseline_height, const std::pair<TensorT, TensorT>& n_peaks,
		const std::pair<TensorT, TensorT>& emg_h, const std::pair<TensorT, TensorT>& emg_tau, const std::pair<TensorT, TensorT>& emg_mu_offset, const std::pair<TensorT, TensorT>& emg_sigma,
		TensorT saturation_limit) const
	{
		// lampda for choosing a random number within l/u bounds
		auto random_bounds = [](const TensorT& lb, const TensorT& ub)->TensorT {
			std::random_device rd; // obtain a random number from hardware
			std::mt19937 eng(rd()); // seed the generator
			std::uniform_int_distribution<> distr(lb, ub); // define the range
			return (TensorT)distr(eng);
		};

		// determine the chrom window size, saturation limits, and number of peaks
		TensorT chrom_window_size_rand = random_bounds(chrom_window_size.first, chrom_window_size.second);
		TensorT n_peaks_rand = random_bounds(n_peaks.first, n_peaks.second);
		TensorT peak_window_length = chrom_window_size_rand / n_peaks_rand;

		// determine the sampling rate
		TensorT step_size_mu_rand = random_bounds(step_size_mu.first, step_size_mu.second);
		TensorT step_size_sigma_rand = random_bounds(step_size_sigma.first, step_size_sigma.second);

		// generate a random set of peaks
		std::vector<PeakSimulator<TensorT>> peaks, peaks_noise;
		std::vector<EMGModel<TensorT>> emgs;
		for (int peak_iter = 0; peak_iter < n_peaks_rand; ++peak_iter) {
			// Define the peak
			TensorT baseline_left = random_bounds(baseline_height.first, baseline_height.second);
			TensorT baseline_right = random_bounds(baseline_height.first, baseline_height.second);
			TensorT noise_mu_rand = random_bounds(noise_mu.first, noise_mu.second);
			TensorT noise_sigma_rand = random_bounds(noise_sigma.first, noise_sigma.second);
			TensorT peak_start = (TensorT)peak_iter * peak_window_length;
			TensorT peak_end = (TensorT)(peak_iter + 1) * peak_window_length;
			peaks.push_back(PeakSimulator<TensorT>(step_size_mu_rand, 0, peak_start, peak_end,
				0, 0, baseline_left, baseline_right, saturation_limit));
			peaks_noise.push_back(PeakSimulator<TensorT>(step_size_mu_rand, step_size_sigma_rand, peak_start, peak_end,
				noise_mu_rand, noise_sigma_rand, baseline_left, baseline_right, saturation_limit));

			// Define the EMG generator
			TensorT h = random_bounds(emg_h.first, emg_h.second);
			TensorT tau = random_bounds(emg_tau.first, emg_tau.second);
			TensorT mu = random_bounds(emg_mu_offset.first, emg_mu_offset.second) + ((peak_end - peak_start) / (TensorT)2 + peak_start);
			TensorT sigma = random_bounds(emg_sigma.first, emg_sigma.second);
			emgs.push_back(EMGModel<TensorT>(h, tau, mu, sigma));
		}

		// make the chromatogram
		makeChromatogram(x_O, y_O, peaks_LR, peaks, emgs);
		makeChromatogram(x_noise_O, y_noise_O, std::vector<std::pair<TensorT, TensorT>>() = {}, peaks_noise, emgs);
	}

	template <typename TensorT>
	void ChromatogramSimulator<TensorT>::makeChromatogram(std::vector<TensorT>& x_O, std::vector<TensorT>& y_O, std::vector<std::pair<TensorT, TensorT>>& peaks_LR,
		const std::vector<PeakSimulator<TensorT>>& peaks, const std::vector<EMGModel<TensorT>>& emgs) const
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
		peaks_LR.clear();

		// Order the list of peaks from lowest to highest emg_mu
		std::vector<std::pair<PeakSimulator<TensorT>, EMGModel<TensorT>>> peak_emg_pairs;
		for (int i = 0; i < emgs.size(); ++i)
		{
			const std::pair<PeakSimulator<TensorT>, EMGModel<TensorT>> peak_emg(peaks[i], emgs[i]);
			peak_emg_pairs.push_back(peak_emg);
		}
		std::sort(peak_emg_pairs.begin(), peak_emg_pairs.end(),
			[](std::pair<PeakSimulator<TensorT>, EMGModel<TensorT>> lhs, std::pair<PeakSimulator<TensorT>, EMGModel<TensorT>> rhs)
		{
			return lhs.second.getMu() < rhs.second.getMu(); //ascending order
		}
		);

		// Join the peaks in order
		if (peak_emg_pairs.size() > 1)
		{
			for (int i = 1; i < peak_emg_pairs.size(); ++i)
			{
				joinPeakWindows(peak_emg_pairs[i - 1].first, peak_emg_pairs[i - 1].second,
					peak_emg_pairs[i].first, peak_emg_pairs[i].second);
			}
		}

		// Add the peaks in order
		for (int i = 0; i < peak_emg_pairs.size(); ++i)
		{
			// make the first peak
			std::vector<TensorT> x, y;
			peak_emg_pairs[i].first.simulatePeak(x, y, peak_emg_pairs[i].second);

			// Determine the best left/right
			std::pair<TensorT, TensorT> best_lr = peak_emg_pairs[i].first.getBestLeftAndRight(x, y, peak_emg_pairs[i].second.getMu());
			if (best_lr.first != best_lr.second != 0)
				peaks_LR.push_back(best_lr);

			// extend the chromatogram
			x_O.reserve(x_O.size() + distance(x.begin(), x.end()));
			x_O.insert(x_O.end(), x.begin(), x.end());
			y_O.reserve(y_O.size() + distance(y.begin(), y.end()));
			y_O.insert(y_O.end(), y.begin(), y.end());
		}
	}
}

#endif //SMARTPEAK_CHROMATOGRAMSIMULATOR_H