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

	/*
	@brief Scale by magnitude of the data
	*/
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
	@brief Project the data onto a specific range
	*/
	template<typename T>
	class LinearScale
	{
	public:
		LinearScale() = default;
		LinearScale(const T& domain_min, const T& domain_max, const T& range_min, const T& range_max):
			domain_min_(domain_min), domain_max_(domain_max), range_min_(range_min), range_max_(range_max){};
		~LinearScale() = default;
		T operator()(const T& x_I) const { 
			T t = (x_I - domain_min_) / (domain_max_ - domain_min_);
			return (range_min_ + (range_max_ - range_min_) * t); 
		};
	private:
		T domain_min_;
		T domain_max_;
		T range_min_;
		T range_max_;
	};

	/*
	@brief "Smooth" binary labels 0 and 1 by a certain offset
	*/
	template<typename T>
	class LabelSmoother
	{
	public:
		LabelSmoother() = default;
		LabelSmoother(const T& zero_offset, const T& one_offset) :
			zero_offset_(zero_offset), one_offset_(one_offset) {};
		~LabelSmoother() = default;
		T operator()(const T& x_I) const {
			const T eps = 1e-3;
			if (x_I < eps) return x_I + zero_offset_;
			else if (x_I > 1 - eps) return x_I - one_offset_;
			else return x_I;
		};
	private:
		T zero_offset_;
		T one_offset_;
	};

	/*
	@brief One hot encoder

	@param[in] data Tensor of input labels in a single column vector
	@param{in] all_possible_values

	@returns an integer tensor where all rows have been expanded
		across the columns with the one hot encoding 
	*/
	template<typename Ta, typename Tb>
	Eigen::Tensor<Tb, 2> OneHotEncoder(const Eigen::Tensor<Ta, 2>& data, const std::vector<Ta>& all_possible_values)
	{
		// integer encode input data
		std::map<Ta, int> T_to_int;
		for (int i = 0; i<all_possible_values.size(); ++i)
			T_to_int.emplace(all_possible_values[i], i);

		// convert to 1 hot vector
		Eigen::Tensor<Tb, 2> onehot_encoded(data.dimension(0), (int)T_to_int.size());
		onehot_encoded.setZero();
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
	template<typename Ta, typename Tb>
	Eigen::Tensor<Tb, 1> OneHotEncoder(const Ta& data, const std::vector<Ta>& all_possible_values)
	{
		// integer encode input data
		std::map<Ta, int> T_to_int;
		for (int i = 0; i<all_possible_values.size(); ++i)
			T_to_int.emplace(all_possible_values[i], i);

		// convert to 1 hot vector
		Eigen::Tensor<Tb, 1> onehot_encoded(T_to_int.size());
		onehot_encoded.setConstant(0);
		onehot_encoded(T_to_int.at(data)) = 1;

		return onehot_encoded;
	}

}

#endif //SMARTPEAK_PREPROCESSING_H