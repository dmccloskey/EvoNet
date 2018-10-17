/**TODO:  Add copyright*/

#ifndef SMARTPEAK_INTEGRATIONFUNCTION_H
#define SMARTPEAK_INTEGRATIONFUNCTION_H

#include <SmartPeak/core/preprocessing.h>
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
    virtual void operator()(std::vector<TensorT*>& source_outputs, std::vector<TensorT*>& weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const std::vector<int>& source_time_steps, const int& sink_time_step, DeviceT& device) = 0;
	protected:
		TensorT eps_ = 1e-9;
  };

	/**
		@brief Sum integration function
	*/
	template<typename TensorT, typename DeviceT>
	class SumOp : public IntegrationOp<TensorT, DeviceT>
	{
	public:
		SumOp() {};
		~SumOp() {};
		//void operator()(std::vector<TensorT*>& source_outputs, std::vector<TensorT*>& weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const std::vector<int>& source_time_steps, const int& sink_time_step, DeviceT& device) {
		//	Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input_tensor(sink_input, batch_size, memory_size);
		//	Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[0], batch_size, memory_size);
		//	Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> weight(weights[0]);
		//	sink_input_tensor.chip(sink_time_step, 1).device(device) = source_output.chip(source_time_steps[0], 1) * weight.broadcast(bcast);
		//	for (int i = 1; i < source_outputs.size(); ++i) {
		//		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[i], batch_size, memory_size);
		//		Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> weight(weights[i]);
		//		sink_input_tensor.chip(sink_time_step, 1).device(device) += source_output.chip(source_time_steps[i], 1) * weight.broadcast(bcast);
		//	}
		//};
		void operator()(std::vector<TensorT*>& source_outputs, std::vector<TensorT*>& weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const std::vector<int>& source_time_steps, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input_tensor(sink_input, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[0], batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> weight(weights[0], 1);
			Eigen::array<int, 2> bcast = { batch_size, 1 };
			sink_input_tensor.chip(sink_time_step, 1).device(device) = source_output.chip(source_time_steps[0], 1) * weight.broadcast(bcast);
			for (int i = 1; i < source_outputs.size(); ++i) {
				Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[i], batch_size, memory_size);
				Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> weight(weights[i], 1);
				sink_input_tensor.chip(sink_time_step, 1).device(device) += source_output.chip(source_time_steps[i], 1) * weight.broadcast(bcast);
			}
		};
		std::string getName() const { return "SumOp"; };
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
		void operator()(std::vector<TensorT*>& source_outputs, std::vector<TensorT*>& weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const std::vector<int>& source_time_steps, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input_tensor(sink_input, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[0], batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> weight(weights[0], 1);
			Eigen::array<int, 2> bcast = { batch_size, 1 };
			sink_input_tensor.chip(sink_time_step, 1).device(device) = source_output.chip(source_time_steps[0], 1) * weight.broadcast(bcast);
			for (int i = 1; i < source_outputs.size(); ++i) {
				Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[i], batch_size, memory_size);
				Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> weight(weights[i], 1);
				sink_input_tensor.chip(sink_time_step, 1).device(device) = sink_input_tensor.chip(sink_time_step, 1) * source_output.chip(source_time_steps[i], 1) * weight.broadcast(bcast);
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
		void operator()(std::vector<TensorT*>& source_outputs, std::vector<TensorT*>& weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const std::vector<int>& source_time_steps, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input_tensor(sink_input, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[0], batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> weight(weights[0], 1);
			Eigen::array<int, 2> bcast = { batch_size, 1 };
			sink_input_tensor.chip(sink_time_step, 1).device(device) = source_output.chip(source_time_steps[0], 1) * weight.broadcast(bcast);
			for (int i = 1; i < source_outputs.size(); ++i) {
				Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[i], batch_size, memory_size);
				Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> weight(weights[i], 1);
				sink_input_tensor.chip(sink_time_step, 1).device(device) = sink_input_tensor.chip(sink_time_step, 1).cwiseMax(
					source_output.chip(source_time_steps[i], 1) * weight.broadcast(bcast));
			}
		};
		std::string getName() const { return "MaxOp"; };
	};

	/**
		@brief Mean integration function

		[TODO: fix scaling by N]
	*/
	template<typename TensorT, typename DeviceT>
	class MeanOp : public IntegrationOp<TensorT, DeviceT>
	{
	public:
		MeanOp() {};
		~MeanOp() {};
		void operator()(std::vector<TensorT*>& source_outputs, std::vector<TensorT*>& weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const std::vector<int>& source_time_steps, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input_tensor(sink_input, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[0], batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> weight(weights[0], 1);
			Eigen::array<int, 2> bcast = { batch_size, 1 };
			sink_input_tensor.chip(sink_time_step, 1).device(device) = source_output.chip(source_time_steps[0], 1) * weight.broadcast(bcast);
			for (int i = 1; i < source_outputs.size(); ++i) {
				Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[i], batch_size, memory_size);
				Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> weight(weights[i], 1);
				sink_input_tensor.chip(sink_time_step, 1).device(device) += source_output.chip(source_time_steps[i], 1) * weight.broadcast(bcast);
			}
			sink_input_tensor.chip(sink_time_step, 1).device(device) = sink_input_tensor.chip(sink_time_step, 1) * sink_input_tensor.chip(sink_time_step, 1).constant(1 / (TensorT)source_outputs.size());
		};
		std::string getName() const { return "MeanOp"; };
	};

	/**
		@brief VarMod integration function

		[TODO: fix scaling by N]

		Modified variance integration function: 1/n Sum[0 to n](Xi)^2
		where Xi = xi - u (u: mean, xi: single sample)
	*/
	template<typename TensorT, typename DeviceT>
	class VarModOp : public IntegrationOp<TensorT, DeviceT>
	{
	public:
		VarModOp() {};
		~VarModOp() {};
		void operator()(std::vector<TensorT*>& source_outputs, std::vector<TensorT*>& weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const std::vector<int>& source_time_steps, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input_tensor(sink_input, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[0], batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> weight(weights[0], 1);
			Eigen::array<int, 2> bcast = { batch_size, 1 };
			auto input = source_output.chip(source_time_steps[0], 1) * weight.broadcast(bcast);
			sink_input_tensor.chip(sink_time_step, 1).device(device) = (input * input);
			for (int i = 1; i < source_outputs.size(); ++i) {
				Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output(source_outputs[i], batch_size, memory_size);
				Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> weight(weights[i], 1);
				auto input = source_output.chip(source_time_steps[i], 1) * weight.broadcast(bcast);
				sink_input_tensor.chip(sink_time_step, 1).device(device) += (input * input);
			}
			sink_input_tensor.chip(sink_time_step, 1).device(device) = sink_input_tensor.chip(sink_time_step, 1) * sink_input_tensor.chip(sink_time_step, 1).constant(1 / (TensorT)source_outputs.size());
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
		void operator()(std::vector<TensorT*>& source_outputs, std::vector<TensorT*>& weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const std::vector<int>& source_time_steps, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_input_tensor(sink_input, batch_size, memory_size);
			sink_input_tensor.chip(sink_time_step, 1).device(device) = sink_input_tensor.chip(sink_time_step, 1).constant(source_outputs.size());  // will not work with CUDA!
		};
		std::string getName() const { return "CountOp"; };
	};

	/**
	@brief Base class for all integration error functions.
	*/
	template<typename TensorT, typename DeviceT>
	class IntegrationErrorOp
	{
	public:
		IntegrationErrorOp() = default;
		IntegrationErrorOp(const TensorT& eps) : eps_(eps) {};
		~IntegrationErrorOp() = default;
		virtual std::string getName() const = 0;
		/*
		@brief Integration error void operator
		*/
		virtual void operator()(TensorT* source_error, TensorT *source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, const int& batch_size, const int& memory_size, const int& source_time_step, const int& sink_time_step, const int& n, DeviceT& device) = 0;
	protected:
		TensorT eps_ = 1e-6;
	};

	/**
	@brief Sum integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class SumErrorOp : public IntegrationErrorOp<TensorT, DeviceT>
	{
	public:
		SumErrorOp() {};
		~SumErrorOp() {};
		void operator()(TensorT* source_error, TensorT *source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, const int& batch_size, const int& memory_size, const int& source_time_step, const int& sink_time_step, const int& n, DeviceT& device)  {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_error_tensor(sink_error, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_error_tensor(source_error, batch_size, memory_size);
			//sink_error_tensor.chip(sink_time_step, 1).device(device) += source_error_tensor.chip(source_time_step, 1).unaryExpr(TensorMultOp<TensorT>(weight));
			Eigen::TensorMap<Eigen::Tensor<TensorT, 1>> weight_tensor(weight, 1);
			Eigen::array<int, 2> bcast = { batch_size, 1 };
			sink_error_tensor.chip(sink_time_step, 1).device(device) += source_error_tensor.chip(source_time_step, 1) * weight_tensor.broadcast(bcast);
		};
		std::string getName() const { return "SumErrorOp"; };
	};

	///**
	//@brief Product integration error function
	//*/
	//template<typename TensorT, typename DeviceT>
	//class ProdErrorOp : public IntegrationErrorOp<TensorT, DeviceT>
	//{
	//public:
	//	ProdErrorOp() {};
	//	~ProdErrorOp() {};
	//	void operator()(TensorT* source_error, TensorT *source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, const int& batch_size, const int& memory_size, const int& source_time_step, const int& sink_time_step, const int& n, DeviceT& device)  {
	//		Eigen::Tensor<TensorT, 1> eps(weight.dimension(0));
	//		eps.setConstant(this->eps_);
	//		return (source_net_input * source_error / (sink_output + eps)).unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9));
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
	//	void operator()(TensorT* source_error, TensorT *source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, const int& batch_size, const int& memory_size, const int& source_time_step, const int& sink_time_step, const int& n, DeviceT& device) 
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
	//	void operator()(TensorT* source_error, TensorT *source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, const int& batch_size, const int& memory_size, const int& source_time_step, const int& sink_time_step, const int& n, DeviceT& device)  {
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
	//	void operator()(TensorT* source_error, TensorT *source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, const int& batch_size, const int& memory_size, const int& source_time_step, const int& sink_time_step, const int& n, DeviceT& device)  {
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
	//	void operator()(TensorT* source_error, TensorT *source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, const int& batch_size, const int& memory_size, const int& source_time_step, const int& sink_time_step, const int& n, DeviceT& device)  {
	//		Eigen::Tensor<TensorT, 1> constant(weight.dimension(0));
	//		constant.setConstant(0);
	//		return constant;
	//	};
	//	std::string getName() const { return "CountErrorOp"; };
	//};

	/**
	@brief Base class for all integration error functions.
	*/
	template<typename TensorT, typename DeviceT>
	class IntegrationWeightGradOp
	{
	public:
		IntegrationWeightGradOp() = default;
		IntegrationWeightGradOp(const TensorT& eps) : eps_(eps) {};
		~IntegrationWeightGradOp() = default;
		virtual std::string getName() const = 0;
		virtual void operator()(TensorT* sink_error, TensorT* source_output, TensorT* weight, TensorT* source_input, TensorT* weight_error, const int& n, const int& batch_size, const int& memory_size, DeviceT& device) = 0;
	protected:
		TensorT eps_ = 1e-6;
		//std::atomic<TensorT, DeviceT> net_weight_error_ = 0; ///< 
	};

	/**
	@brief Sum integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class SumWeightGradOp : public IntegrationWeightGradOp<TensorT, DeviceT>
	{
	public:
		void operator()(TensorT* sink_error, TensorT* source_output, TensorT* weight, TensorT* source_input, TensorT* weight_error, const int& n, const int& batch_size, const int& memory_size, DeviceT& device){
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> sink_error_tensor(sink_error, batch_size, memory_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_output_tensor(source_output, batch_size, memory_size);
			//Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> source_input_tensor(source_input, batch_size, memory_size);
			//Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> weight_tensor(weight);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> weight_error_tensor(weight_error);
			weight_error_tensor.device(device) += ((-sink_error_tensor * source_output_tensor).sum(Eigen::array<int, 2>({ 1 }))).mean({ 0 }); // sum across time; average across batches			
		};
		std::string getName() const { return "SumWeightGradOp"; };
	};

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
	//		Eigen::Tensor<TensorT, 0> derivative_mean_tensor = ((-sink_error * source_net_input / weight).unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9))).mean(); // average derivative
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