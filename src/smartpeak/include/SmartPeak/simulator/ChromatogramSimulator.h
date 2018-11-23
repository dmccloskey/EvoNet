/**TODO:  Add copyright*/

#ifndef SMARTPEAK_CHROMATOGRAMSIMULATOR_H
#define SMARTPEAK_CHROMATOGRAMSIMULATOR_H

// .h
#include <SmartPeak/simulator/EMGModel.h>
#include <SmartPeak/simulator/PeakSimulator.h>

namespace SmartPeak
{
	/**
		@brief A class to generate points that represent an LC-MS,
			GC-MS, or HPLC chromatogram
	*/
	template <typename TensorT>
	class ChromatogramSimulator
	{
	public:
		ChromatogramSimulator() = default; ///< Default constructor
		~ChromatogramSimulator() = default; ///< Default destructor

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
			@param[in] peaks list of PeakSimulator classes that will compose the chromatogram
			@param[in] emgs list of corresponding EMGModel classes that define each peak
		*/
		void simulateChromatogram(std::vector<TensorT>& x_O, std::vector<TensorT>& y_O,
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
		PeakSimulator peak_l = peak_left;
		PeakSimulator peak_r = peak_right;

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

	template <typename TensorT>
	void ChromatogramSimulator<TensorT>::simulateChromatogram(std::vector<TensorT>& x_O, std::vector<TensorT>& y_O,
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

		// Order the list of peaks from lowest to highest emg_mu
		std::vector<std::pair<PeakSimulator, EMGModel>> peak_emg_pairs;
		for (int i = 0; i < emgs.size(); ++i)
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

			// extend the chromatogram
			x_O.reserve(x_O.size() + distance(x.begin(), x.end()));
			x_O.insert(x_O.end(), x.begin(), x.end());
			y_O.reserve(y_O.size() + distance(y.begin(), y.end()));
			y_O.insert(y_O.end(), y.begin(), y.end());
		}
	}
}

#endif //SMARTPEAK_CHROMATOGRAMSIMULATOR_H