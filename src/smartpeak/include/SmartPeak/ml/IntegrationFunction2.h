/**TODO:  Add copyright*/

#ifndef SMARTPEAK_INTEGRATIONFUNCTION_H
#define SMARTPEAK_INTEGRATIONFUNCTION_H

#include <SmartPeak/ml/SharedFunctions.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace SmartPeak
{
	/**
	@brief Functor for use with calculate activation/derivative.
	*/
	template<typename TensorT, typename DeviceT>
	class WeightMultOp
	{
	public:
		WeightMultOp() = default;
		WeightMultOp(const TensorT& weight) : weight_(weight) {};
		~WeightMultOp() = default;
		TensorT operator()(const TensorT& x_I) const {
			return x_I*weight_;
		}
	private:
		TensorT weight_ = 1;
	};

	template<typename TensorT>
	class TensorMultOp
	{
	public:
		TensorMultOp() = default;
		TensorMultOp(TensorT* weight) : weight_(weight) {};
		~TensorMultOp() = default;
		TensorT operator()(const TensorT& x_I) const {
			return x_I * (*weight_);
		}
	private:
		TensorT* weight_;
	};

  /**
    @brief Base class for all integration functions.
  */
	template<typename TensorT, typename DeviceT>
  class IntegrationOp
  {
public: 
    IntegrationOp() = default;
		IntegrationOp(const TensorT& eps) : eps_(eps) {};
    ~IntegrationOp() = default;
    virtual std::string getName() const = 0;
    virtual void operator()(std::vector<TensorT*> source_outputs, std::vector<TensorT*> weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const std::vector<int>& source_time_steps, const int& sink_time_step, DeviceT& device) = 0;
	protected:
		TensorT eps_ = 1e-9;
  };

  /**
    @brief Sum integration function
  */
	template<typename TensorT, typename DeviceT>
  class SumOp: public IntegrationOp<TensorT, DeviceT>
  {
public: 
		SumOp(){};
    ~SumOp(){};
    void operator()(std::vector<TensorT*> source_outputs, std::vector<TensorT*> weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const std::vector<int>& source_time_steps, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input_tensor(sink_input, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[0], batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> weight(weights[0]);
			sink_input_tensor.chip(sink_time_step, 1).device(device) = source_output.chip(source_time_steps[0], 1).unaryExpr(TensorMultOp<TensorT>(weights[0]));
			for (int i = 1; i < source_outputs.size(); ++i) {
				Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[i], batch_size, memory_size);
				Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> weight(weights[i]); 
				sink_input_tensor.chip(sink_time_step, 1).device(device) += source_output.chip(source_time_steps[i], 1).unaryExpr(TensorMultOp<TensorT>(weights[i]));
			}
		};
		//// Note that the use of broadcast appears not to work!
		//void operator()(std::vector<TensorT*> source_outputs, std::vector<TensorT*> weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const std::vector<int>& source_time_steps, const int& sink_time_step, DeviceT& device) {
		//	Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input_tensor(sink_input, batch_size, memory_size);
		//	sink_input_tensor.chip(sink_time_step, 1).setConstant(0).device(device);
		//	Eigen::array<int, 2> bcast = { batch_size, 1 };
		//	for (int i = 0; i < source_outputs.size(); ++i) {
		//		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[i], batch_size, memory_size);
		//		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight(weights[i], 1, 1);
		//		sink_input_tensor.chip(sink_time_step, 1).device(device) += source_output.chip(source_time_steps[i], 1) * weight.broadcast(bcast);
		//	}
		//};
    std::string getName() const{return "SumOp";};
  };

	/**
	@brief Product integration function
	*/
	template<typename TensorT, typename DeviceT>
	class ProdOp : public IntegrationOp<TensorT, DeviceT>
	{
	public:
		ProdOp() {};
		~ProdOp() {};
		void operator()(std::vector<TensorT*> source_outputs, std::vector<TensorT*> weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const std::vector<int>& source_time_steps, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input_tensor(sink_input, batch_size, memory_size);
			sink_input_tensor.chip(sink_time_step, 1).setConstant(1).device(device);
			for (int i = 0; i < source_outputs.size(); ++i) {
				Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[i], batch_size, memory_size);
				Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> weight(weights[i]);
				sink_input_tensor.chip(sink_time_step, 1).device(device) = sink_input_tensor.chip(sink_time_step, 1) * source_output.chip(source_time_steps[i], 1).unaryExpr(WeightMultOp<TensorT, DeviceT>(weight(0)));
			}
		};
		std::string getName() const { return "ProdOp"; };
	};

	/**
	@brief Max integration function
	*/
	template<typename TensorT, typename DeviceT>
	class MaxOp : public IntegrationOp<TensorT, DeviceT>
	{
	public:
		MaxOp() {};
		~MaxOp() {};
		void operator()(std::vector<TensorT*> source_outputs, std::vector<TensorT*> weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const std::vector<int>& source_time_steps, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input_tensor(sink_input, batch_size, memory_size);
			sink_input_tensor.chip(sink_time_step, 1).setConstant(0).device(device);
			for (int i = 0; i < source_outputs.size(); ++i) {
				Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[i], batch_size, memory_size);
				Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> weight(weights[i]);
				sink_input_tensor.chip(sink_time_step, 1).device(device) = sink_input_tensor.chip(sink_time_step, 1).cwiseMax(
					source_output.chip(source_time_steps[i], 1).unaryExpr(WeightMultOp<TensorT, DeviceT>(weight(0))));
			}
		};
		std::string getName() const { return "MaxOp"; };
	};

	/**
		@brief Mean integration function
	*/
	template<typename TensorT, typename DeviceT>
	class MeanOp : public IntegrationOp<TensorT, DeviceT>
	{
	public:
		MeanOp() {};
		~MeanOp() {};
		void operator()(std::vector<TensorT*> source_outputs, std::vector<TensorT*> weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const std::vector<int>& source_time_steps, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input_tensor(sink_input, batch_size, memory_size);
			sink_input_tensor.chip(sink_time_step, 1).setConstant(0).device(device);
			for (int i = 0; i < source_outputs.size(); ++i) {
				Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[i], batch_size, memory_size);
				Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> weight(weights[i]);
				sink_input_tensor.chip(sink_time_step, 1).device(device) += source_output.chip(source_time_steps[i], 1).unaryExpr(WeightMultOp<TensorT, DeviceT>(weight(0) / source_outputs.size()));
			}
		};
		std::string getName() const { return "MeanOp"; };
	};

	/**
		@brief VarMod integration function

		Modified variance integration function: 1/n Sum[0 to n](Xi)^2
		where Xi = xi - u (u: mean, xi: single sample)
	*/
	template<typename TensorT, typename DeviceT>
	class VarModOp : public IntegrationOp<TensorT, DeviceT>
	{
	public:
		VarModOp() {};
		~VarModOp() {};
		void operator()(std::vector<TensorT*> source_outputs, std::vector<TensorT*> weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const std::vector<int>& source_time_steps, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input_tensor(sink_input, batch_size, memory_size);
			sink_input_tensor.chip(sink_time_step, 1).setConstant(0).device(device);
			for (int i = 0; i < source_outputs.size(); ++i) {
				Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[i], batch_size, memory_size);
				Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> weight(weights[i]);
				auto input = source_output.chip(source_time_steps[i], 1).unaryExpr(WeightMultOp<TensorT, DeviceT>(weight(0)));
				sink_input_tensor.chip(sink_time_step, 1).device(device) += (input * input).unaryExpr(WeightMultOp<TensorT, DeviceT>(1 / source_outputs.size()));
			}
		};
		std::string getName() const { return "VarModOp"; };
	};

	/**
		@brief Count integration function
	*/
	template<typename TensorT, typename DeviceT>
	class CountOp : public IntegrationOp<TensorT, DeviceT>
	{
	public:
		CountOp() {};
		void initNetNodeInput(const int& batch_size) {
			Eigen::Tensor<TensorT, 1> net_node_input(batch_size);
			net_node_input.setConstant(0);
			this->setNetNodeInput(net_node_input);
		}
		~CountOp() {};
		void operator()(std::vector<TensorT*> source_outputs, std::vector<TensorT*> weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const std::vector<int>& source_time_steps, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input_tensor(sink_input, batch_size, memory_size);
			sink_input_tensor.chip(sink_time_step, 1).setConstant(source_outputs.size()).device(device);
		};
		std::string getName() const { return "CountOp"; };
	};

	///**
	//@brief Base class for all integration error functions.
	//*/
	//template<typename TensorT, typename DeviceT>
	//class IntegrationErrorOp
	//{
	//public:
	//	IntegrationErrorOp() = default;
	//	IntegrationErrorOp(const TensorT& eps) : eps_(eps) {};
	//	~IntegrationErrorOp() = default;
	//	virtual std::string getName() const = 0;
	//	/*
	//	@brief Sum integration error void operator

	//	@param[in] x1 The weight tensor
	//	@param[in] x2 The source error tensor
	//	@param[in] x3 The source net input tensor
	//	@param[in] x4 The sink output tensor
	//	@param[in] x5 The number of inputs
	//	*/
	//	virtual Eigen::Tensor<TensorT, 1> operator()(const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_error, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& sink_output, const Eigen::Tensor<TensorT, 1>& n) = 0;
	//protected:
	//	TensorT eps_ = 1e-6;
	//};

	///**
	//@brief Sum integration error function
	//*/
	//template<typename TensorT, typename DeviceT>
	//class SumErrorOp : public IntegrationErrorOp<TensorT, DeviceT>
	//{
	//public:
	//	SumErrorOp() {};
	//	~SumErrorOp() {};
	//	Eigen::Tensor<TensorT, 1> operator()(const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_error, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& sink_output, const Eigen::Tensor<TensorT, 1>& n) {
	//		return weight * source_error;
	//	};
	//	std::string getName() const { return "SumErrorOp"; };
	//};

	///**
	//@brief Product integration error function
	//*/
	//template<typename TensorT, typename DeviceT>
	//class ProdErrorOp : public IntegrationErrorOp<TensorT, DeviceT>
	//{
	//public:
	//	ProdErrorOp() {};
	//	~ProdErrorOp() {};
	//	Eigen::Tensor<TensorT, 1> operator()(const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_error, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& sink_output, const Eigen::Tensor<TensorT, 1>& n) {
	//		Eigen::Tensor<TensorT, 1> eps(weight.dimension(0));
	//		eps.setConstant(this->eps_);
	//		return (source_net_input * source_error / (sink_output + eps)).unaryExpr(ClipOp<TensorT, DeviceT>(1e-6, -1e9, 1e9)); // .unaryExpr(std::ptr_fun(checkNan<TensorT, DeviceT>));
	//		//return (source_net_input * source_error / sink_output).unaryExpr(ClipOp<TensorT, DeviceT>(1e-6, -1e9, 1e9)).unaryExpr(std::ptr_fun(checkNan<TensorT, DeviceT>));
	//	};
	//	std::string getName() const { return "ProdErrorOp"; };
	//};

	///**
	//@brief Max integration error function
	//*/
	//template<typename TensorT, typename DeviceT>
	//class MaxErrorOp : public IntegrationErrorOp<TensorT, DeviceT>
	//{
	//public:
	//	MaxErrorOp() {};
	//	~MaxErrorOp() {};
	//	Eigen::Tensor<TensorT, 1> operator()(const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_error, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& sink_output, const Eigen::Tensor<TensorT, 1>& n)
	//	{
	//		auto perc_max_tensor = (sink_output / source_net_input).unaryExpr([](const TensorT& v) {
	//			if (v < 1 - 1e-6) 
	//				return 0;
	//			else 
	//				return 1;
	//		});
	//		return weight * source_error * perc_max_tensor;
	//	};
	//	std::string getName() const { return "MaxErrorOp"; };
	//};

	///**
	//@brief Mean integration error function
	//*/
	//template<typename TensorT, typename DeviceT>
	//class MeanErrorOp : public IntegrationErrorOp<TensorT, DeviceT>
	//{
	//public:
	//	MeanErrorOp() {};
	//	~MeanErrorOp() {};
	//	Eigen::Tensor<TensorT, 1> operator()(const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_error, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& sink_output, const Eigen::Tensor<TensorT, 1>& n) {
	//		return weight * source_error / n;
	//	};
	//	std::string getName() const { return "MeanErrorOp"; };
	//};

	///**
	//@brief VarMod integration error function
	//*/
	//template<typename TensorT, typename DeviceT>
	//class VarModErrorOp : public IntegrationErrorOp<TensorT, DeviceT>
	//{
	//public:
	//	VarModErrorOp() {};
	//	~VarModErrorOp() {};
	//	Eigen::Tensor<TensorT, 1> operator()(const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_error, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& sink_output, const Eigen::Tensor<TensorT, 1>& n) {
	//		Eigen::Tensor<TensorT, 1> constant(weight.dimension(0));
	//		constant.setConstant(2);
	//		return weight * source_error * constant / n;
	//	};
	//	std::string getName() const { return "VarModErrorOp"; };
	//};

	///**
	//@brief Count integration error function
	//*/
	//template<typename TensorT, typename DeviceT>
	//class CountErrorOp : public IntegrationErrorOp<TensorT, DeviceT>
	//{
	//public:
	//	CountErrorOp() {};
	//	~CountErrorOp() {};
	//	Eigen::Tensor<TensorT, 1> operator()(const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_error, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& sink_output, const Eigen::Tensor<TensorT, 1>& n) {
	//		Eigen::Tensor<TensorT, 1> constant(weight.dimension(0));
	//		constant.setConstant(0);
	//		return constant;
	//	};
	//	std::string getName() const { return "CountErrorOp"; };
	//};

	///**
	//@brief Base class for all integration error functions.
	//*/
	//template<typename TensorT, typename DeviceT>
	//class IntegrationWeightGradOp
	//{
	//public:
	//	IntegrationWeightGradOp() = default;
	//	IntegrationWeightGradOp(const TensorT& eps) : eps_(eps) {};
	//	~IntegrationWeightGradOp() = default;
	//	void initNetWeightError() { net_weight_error_ = 0; }
	//	void setNetWeightError(const TensorT& net_weight_error) { net_weight_error_ = net_weight_error; }
	//	TensorT getNetWeightError() const { return net_weight_error_; }
	//	virtual std::string getName() const = 0;
	//	virtual void operator()(const Eigen::Tensor<TensorT, 1>& sink_error, const Eigen::Tensor<TensorT, 1>& source_output, const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& n) = 0;
	//protected:
	//	TensorT net_weight_error_ = 0; ///< 
	//	TensorT eps_ = 1e-6;
	//	//std::atomic<TensorT, DeviceT> net_weight_error_ = 0; ///< 
	//};

	///**
	//@brief Sum integration error function
	//*/
	//template<typename TensorT, typename DeviceT>
	//class SumWeightGradOp : public IntegrationWeightGradOp<TensorT, DeviceT>
	//{
	//public:
	//	SumWeightGradOp() { this->setNetWeightError(TensorT(0)); };
	//	~SumWeightGradOp() {};
	//	void operator()(const Eigen::Tensor<TensorT, 1>& sink_error, const Eigen::Tensor<TensorT, 1>& source_output, const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& n) {
	//		Eigen::Tensor<TensorT, 0> derivative_mean_tensor = (-sink_error * source_output).mean(); // average derivative
	//		this->net_weight_error_ += derivative_mean_tensor(0);
	//	};
	//	std::string getName() const { return "SumWeightGradOp"; };
	//};

	///**
	//@brief Product integration error function
	//*/
	//template<typename TensorT, typename DeviceT>
	//class ProdWeightGradOp : public IntegrationWeightGradOp<TensorT, DeviceT>
	//{
	//public:
	//	ProdWeightGradOp() { this->setNetWeightError(TensorT(0)); };
	//	~ProdWeightGradOp() {};
	//	void operator()(const Eigen::Tensor<TensorT, 1>& sink_error, const Eigen::Tensor<TensorT, 1>& source_output, const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& n) {
	//		Eigen::Tensor<TensorT, 0> derivative_mean_tensor = ((-sink_error * source_net_input / weight).unaryExpr(ClipOp<TensorT, DeviceT>(1e-6, -1e9, 1e9))).mean(); // average derivative
	//		this->net_weight_error_ += derivative_mean_tensor(0);
	//	};
	//	std::string getName() const { return "ProdWeightGradOp"; };
	//};

	///**
	//@brief Max integration error function
	//*/
	//template<typename TensorT, typename DeviceT>
	//class MaxWeightGradOp : public IntegrationWeightGradOp<TensorT, DeviceT>
	//{
	//public:
	//	MaxWeightGradOp() { this->setNetWeightError(TensorT(0)); };
	//	~MaxWeightGradOp() {};
	//	void operator()(const Eigen::Tensor<TensorT, 1>& sink_error, const Eigen::Tensor<TensorT, 1>& source_output, const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& n) {
	//		Eigen::Tensor<TensorT, 0> derivative_mean_tensor = (-sink_error * source_output).mean(); // average derivative
	//		this->net_weight_error_ += derivative_mean_tensor(0);
	//	};
	//	std::string getName() const { return "MaxWeightGradOp"; };
	//};

	///**
	//@brief Count integration error function
	//*/
	//template<typename TensorT, typename DeviceT>
	//class CountWeightGradOp : public IntegrationWeightGradOp<TensorT, DeviceT>
	//{
	//public:
	//	CountWeightGradOp() { this->setNetWeightError(TensorT(0)); };
	//	~CountWeightGradOp() {};
	//	void operator()(const Eigen::Tensor<TensorT, 1>& sink_error, const Eigen::Tensor<TensorT, 1>& source_output, const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& n) {
	//		this->net_weight_error_ += 0;
	//	};
	//	std::string getName() const { return "CountWeightGradOp"; };
	//};

	///**
	//@brief Mean integration error function
	//*/
	//template<typename TensorT, typename DeviceT>
	//class MeanWeightGradOp : public IntegrationWeightGradOp<TensorT, DeviceT>
	//{
	//public:
	//	MeanWeightGradOp() { this->setNetWeightError(TensorT(0)); };
	//	~MeanWeightGradOp() {};
	//	void operator()(const Eigen::Tensor<TensorT, 1>& sink_error, const Eigen::Tensor<TensorT, 1>& source_output, const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& n) {
	//		Eigen::Tensor<TensorT, 0> derivative_mean_tensor = (-sink_error * source_output / n).mean(); // average derivative
	//		this->net_weight_error_ += derivative_mean_tensor(0);
	//	};
	//	std::string getName() const { return "MeanWeightGradOp"; };
	//};

	///**
	//@brief VarMod integration error function
	//*/
	//template<typename TensorT, typename DeviceT>
	//class VarModWeightGradOp : public IntegrationWeightGradOp<TensorT, DeviceT>
	//{
	//public:
	//	VarModWeightGradOp() { this->setNetWeightError(TensorT(0)); };
	//	~VarModWeightGradOp() {};
	//	void operator()(const Eigen::Tensor<TensorT, 1>& sink_error, const Eigen::Tensor<TensorT, 1>& source_output, const Eigen::Tensor<TensorT, 1>& weight, const Eigen::Tensor<TensorT, 1>& source_net_input, const Eigen::Tensor<TensorT, 1>& n) {
	//		Eigen::Tensor<TensorT, 1> constant(weight.dimension(0));
	//		constant.setConstant(2);
	//		Eigen::Tensor<TensorT, 0> derivative_mean_tensor = (-sink_error * source_output * constant / n).mean(); // average derivative
	//		this->net_weight_error_ += derivative_mean_tensor(0);
	//	};
	//	std::string getName() const { return "VarModWeightGradOp"; };
	//};
}
#endif //SMARTPEAK_INTEGRATIONFUNCTION_H