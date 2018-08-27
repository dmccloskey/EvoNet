#ifndef SMARTPEAK_PREPROCESSING_H
#define SMARTPEAK_PREPROCESSING_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <map>
#include <random>

namespace SmartPeak
{
	/*
	@brief Methods for data preprocessing and normalization
	*/

	template<typename T>
	T selectRandomElement(std::vector<T> elements)
	{
		try
		{
			// select a random node
			// based on https://www.rosettacode.org/wiki/Pick_random_element
			std::random_device seed;
			std::mt19937 engine(seed());
			std::uniform_int_distribution<int> choose(0, elements.size() - 1);
			return elements[choose(engine)];
		}
		catch (std::exception& e)
		{
			printf("Exception in selectRandomElement: %s", e.what());
		}
	}

	template<typename T>
	class UnitScale
	{
	public:
		UnitScale() {};
		UnitScale(const Eigen::Tensor<T, 2>& data) { setUnitScale(data); };
		~UnitScale() {};
		void setUnitScale(const Eigen::Tensor<T, 2>& data)
		{
			const Eigen::Tensor<T, 0> max_value = data.maximum();
			const Eigen::Tensor<T, 0> min_value = data.minimum();
			unit_scale_ = 1 / sqrt(pow(max_value(0) - min_value(0), 2));
		}
		T getUnitScale() { return unit_scale_; }
		T operator()(const T& x_I) const { return x_I * unit_scale_; };
	private:
		T unit_scale_;
	};

	/*
	@brief One hot encoder

	@param[in] data Tensor of input labels in a single column vector
	@param{in] all_possible_values

	@returns an integer tensor where all rows have been expanded
		across the columns with the one hot encoding 
	*/
	template<typename T>
	Eigen::Tensor<int, 2> OneHotEncoder(Eigen::Tensor<T, 2>& data, const std::vector<T>& all_possible_values)
	{
		// integer encode input data
		std::map<T, int> T_to_int;
		for (int i = 0; i<all_possible_values.size(); ++i)
			T_to_int.emplace(all_possible_values[i], i);

		// convert to 1 hot vector
		Eigen::Tensor<int, 2> onehot_encoded(data.dimension(0), T_to_int.size());
		onehot_encoded.setConstant(0);
		for (int i = 0; i<data.dimension(0); ++i)
			onehot_encoded(i, T_to_int.at(data(i, 0))) = 1;

		return onehot_encoded;
	}
	
	/*
	@brief One hot encoder

	@param[in] data input label
	@param{in] all_possible_values

	@returns an integer tensor with the one hot encoding
	*/
	template<typename T>
	Eigen::Tensor<int, 1> OneHotEncoder(const T& data, const std::vector<T>& all_possible_values)
	{
		// integer encode input data
		std::map<T, int> T_to_int;
		for (int i = 0; i<all_possible_values.size(); ++i)
			T_to_int.emplace(all_possible_values[i], i);

		// convert to 1 hot vector
		Eigen::Tensor<int, 1> onehot_encoded(T_to_int.size());
		onehot_encoded.setConstant(0);
		onehot_encoded(T_to_int.at(data)) = 1;

		return onehot_encoded;
	}

}

#endif //SMARTPEAK_PREPROCESSING_H