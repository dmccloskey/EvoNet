/**TODO:  Add copyright*/

#ifndef SMARTPEAK_ADDPROBSIMULATOR_H
#define SMARTPEAK_ADDPROBSIMULATOR_H

#include <SmartPeak/simulator/DataSimulator.h>
#include <SmartPeak/core/Preprocessing.h>

namespace SmartPeak
{
  /**
		@brief implementation of the add problem that
		has been used to test sequence prediction in
		RNNS

		References:
		[TODO]
  */
	template<typename TensorT>
  class AddProbSimulator: public DataSimulator<TensorT>
  {
public:
		/*
		@brief implementation of the add problem that
		has been used to test sequence prediction in
		RNNS

		@param[in, out] random_sequence
		@param[in, out] mask_sequence
		@param[in] n_masks The number of random additions

		@returns the result of the two random numbers in the sequence
		**/
		static TensorT AddProb(
			Eigen::Tensor<TensorT, 1>& random_sequence,
			Eigen::Tensor<TensorT, 1>& mask_sequence,
			const int& n_masks)
		{
			TensorT result = 0.0;
			const int sequence_length = random_sequence.size();

			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_real_distribution<> zero_to_one(0.0, 1.0); // in the range of abs(min/max(+/-0.5)) + abs(min/max(+/-0.5)) for TanH
			std::uniform_int_distribution<> zero_to_length(0, sequence_length - 1);

			// generate 2 random and unique indexes between 
			// [0, sequence_length) for the mask
			std::vector<int> mask_indices = { zero_to_length(gen) };
			for (int i = 0; i < n_masks - 1; ++i)
			{
				int mask_index = 0;
				do {
					mask_index = zero_to_length(gen);
				} while (std::count(mask_indices.begin(), mask_indices.end(), mask_index) != 0);
				mask_indices.push_back(mask_index);
			}

			// generate the random sequence
			// and the mask sequence
			for (int i = 0; i < sequence_length; ++i)
			{
				// the random sequence
				random_sequence(i) = zero_to_one(gen);
				// the mask
				if (std::count(mask_indices.begin(), mask_indices.end(), i) != 0)
					mask_sequence(i) = 1.0;
				else
					mask_sequence(i) = 0.0;

				// result update
				result += mask_sequence(i) * random_sequence(i);
			}

			//std::cout<<"mask sequence: "<<mask_sequence<<std::endl; [TESTS:convert to a test!]
			//std::cout<<"random sequence: "<<random_sequence<<std::endl; [TESTS:convert to a test!]
			//std::cout<<"result: "<<result<<std::endl; [TESTS:convert to a test!]

			return result;
		}

		int n_mask_ = 5;
		int sequence_length_ = 25;
  };
}

#endif //SMARTPEAK_ADDPROBSIMULATOR_H