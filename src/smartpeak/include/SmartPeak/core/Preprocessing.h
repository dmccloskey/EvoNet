#ifndef SMARTPEAK_PREPROCESSING_H
#define SMARTPEAK_PREPROCESSING_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <map>

namespace SmartPeak
{
	/*
	@brief Methods for data preprocessing and normalization
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
		T operator()(const T& x_I) const { return x_I / unit_scale_; };
	private:
		T unit_scale_;
	};

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

}

#endif //SMARTPEAK_PREPROCESSING_H