/**TODO:  Add copyright*/

#ifndef SMARTPEAK_TENSORINTEGRATIONFUNCTION_H
#define SMARTPEAK_TENSORINTEGRATIONFUNCTION_H

#include <SmartPeak/core/preprocessing.h>
#include <SmartPeak/ml/ActivationFunctionTensor.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace SmartPeak
{
  /**
    @brief Base class for all integration functions.
  */
	template<typename TensorT, typename DeviceT>
  class IntegrationTensorOp
  {
public: 
    IntegrationTensorOp() = default;
		IntegrationTensorOp(const TensorT& eps) : eps_(eps) {};
    ~IntegrationTensorOp() = default;
    virtual std::string getName() const = 0;
    virtual void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) = 0;
	protected:
		TensorT eps_ = 1e-9;
  };

	/**
		@brief Fully Connected Sum integration function
	*/
	template<typename TensorT, typename DeviceT>
	class FullyConnectedSumOp : public IntegrationTensorOp<TensorT, DeviceT>
	{
	public:
		FullyConnectedSumOp() {};
		~FullyConnectedSumOp() {};
		void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight(weights, source_layer_size, sink_layer_size);
			Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
			sink_input_tensor.chip(sink_time_step, 1).device(device) = (source_output_tensor.chip(source_time_step, 1)).contract(weight, product_dims);
		};
		std::string getName() const { return "FullyConnectedSumOp"; };
	};

	/**
		@brief Fully Connected Sum integration function
	*/
	template<typename TensorT, typename DeviceT>
	class SumTensorOp : public IntegrationTensorOp<TensorT, DeviceT>
	{
	public:
		SumTensorOp() {};
		~SumTensorOp() {};
		void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight(weights, source_layer_size, sink_layer_size);
			Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
			sink_input_tensor.chip(sink_time_step, 1).device(device) += (source_output_tensor.chip(source_time_step, 1)).contract(weight, product_dims);
		};
		std::string getName() const { return "SumTensorOp"; };
	};

	/**
		@brief Singly Connected Sum integration function
	*/
	template<typename TensorT, typename DeviceT>
	class SinglyConnectedSumOp : public IntegrationTensorOp<TensorT, DeviceT>
	{
	public:
		SinglyConnectedSumOp() {};
		~SinglyConnectedSumOp() {};
		void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			assert(source_layer_size == sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight(weights, 1, source_layer_size);
			Eigen::array<int, 2> bcast = { batch_size, 1 };
			sink_input_tensor.chip(sink_time_step, 1).device(device) += source_output_tensor.chip(source_time_step, 1) * weight.broadcast(bcast);
		}
		std::string getName() const { return "SinglyConnectedSumOp"; };
	};

	/**
		@brief Fan In Sum integration function
	*/
	template<typename TensorT, typename DeviceT>
	class FanInSumOp : public IntegrationTensorOp<TensorT, DeviceT>
	{
	public:
		FanInSumOp() {};
		~FanInSumOp() {};
		void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			assert(sink_layer_size == 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight(weights, 1, source_layer_size);
			Eigen::array<int, 2> bcast = { batch_size, 1 };
			Eigen::array<int, 1> dims({ 1 });
			sink_input_tensor.chip(sink_time_step, 1).device(device) += (source_output_tensor.chip(source_time_step, 1) * weight.broadcast(bcast)).sum(dims);
		}
		std::string getName() const { return "FanInSumOp"; };
	};

	/**
		@brief Fan out Sum integration function
	*/
	template<typename TensorT, typename DeviceT>
	class FanOutSumOp : public IntegrationTensorOp<TensorT, DeviceT>
	{
	public:
		FanOutSumOp() {};
		~FanOutSumOp() {};
		void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			assert(source_layer_size == 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight(weights, 1, sink_layer_size);
			Eigen::array<int, 2> bcast_batch = { batch_size, 1 };
			Eigen::array<int, 2> bcast_sink = { sink_layer_size, 1 };
			sink_input_tensor.chip(sink_time_step, 1).device(device) += (source_output_tensor.chip(source_time_step, 1)).broadcast(bcast_sink) * weight.broadcast(bcast_batch).broadcast(cast_sink);
		}
		std::string getName() const { return "FanOutSumOp"; };
	};

	/**
		@brief Singly Connected Prod integration function
	*/
	template<typename TensorT, typename DeviceT>
	class SinglyConnectedProdOp : public IntegrationTensorOp<TensorT, DeviceT>
	{
	public:
		SinglyConnectedProdOp() {};
		~SinglyConnectedProdOp() {};
		void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			assert(source_layer_size == sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight(weights, 1, source_layer_size);
			Eigen::array<int, 2> bcast = { batch_size, 1 };
			sink_input_tensor.chip(sink_time_step, 1).device(device) = sink_input_tensor.chip(sink_time_step, 1) * source_output_tensor.chip(source_time_step, 1) * weight.broadcast(bcast);
		}
		std::string getName() const { return "SinglyConnectedProdOp"; };
	};

	/**
		@brief Fan In Prod integration function
	*/
	template<typename TensorT, typename DeviceT>
	class ProdTensorOp : public IntegrationTensorOp<TensorT, DeviceT>
	{
	public:
		ProdTensorOp() {};
		~ProdTensorOp() {};
		void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> weight_tensor(weights, 1, source_layer_size, sink_layer_size);
			Eigen::array<int, 2> bcast = { batch_size, 0 };
			Eigen::array<int, 1> dims({ 1 });
			sink_input_tensor.chip(sink_time_step, 1).device(device) = sink_input_tensor.chip(sink_time_step, 1) * (source_output_tensor.chip(source_time_step, 1) * weight_tensor.broadcast(bcast)).prod(dims);
		}
		std::string getName() const { return "ProdTensorOp"; };
	};

	/**
		@brief Singly Connected Max integration function
	*/
	template<typename TensorT, typename DeviceT>
	class SinglyConnectedMaxOp : public IntegrationTensorOp<TensorT, DeviceT>
	{
	public:
		SinglyConnectedMaxOp() {};
		~SinglyConnectedMaxOp() {};
		void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			assert(source_layer_size == sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight_tensor(weights, 1, source_layer_size);
			Eigen::array<int, 2> bcast = { batch_size, 1 };
			sink_input_tensor.chip(sink_time_step, 1).device(device) = sink_input_tensor.chip(sink_time_step, 1).cwiseMax(source_output_tensor.chip(source_time_step, 1) * weight_tensor.broadcast(bcast));
		}
		std::string getName() const { return "SinglyConnectedMaxOp"; };
	};

	/**
		@brief Fan In Max integration function
	*/
	template<typename TensorT, typename DeviceT>
	class FanInMaxOp : public IntegrationTensorOp<TensorT, DeviceT>
	{
	public:
		FanInMaxOp() {};
		~FanInMaxOp() {};
		void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			assert(sink_layer_size == 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight(weights, 1, source_layer_size);
			Eigen::array<int, 2> bcast = { batch_size, 1 };
			Eigen::array<int, 1> dims({ 1 });
			sink_input_tensor.chip(sink_time_step, 1).device(device) = sink_input_tensor.chip(sink_time_step, 1).cwiseMax((source_output_tensor.chip(source_time_step, 1) * weight.broadcast(bcast)).maximum(dims));
		}
		std::string getName() const { return "FanInMaxOp"; };
	};

	/**
		@brief Max integration function
	*/
	template<typename TensorT, typename DeviceT>
	class MaxTensorOp : public IntegrationTensorOp<TensorT, DeviceT>
	{
	public:
		MaxTensorOp() {};
		~MaxTensorOp() {};
		void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> weight_tensor(weights, 1, source_layer_size, sink_layer_size);
			Eigen::array<int, 2> bcast = { batch_size, 0 };
			Eigen::array<int, 1> dims({ 1 });
			sink_input_tensor.chip(sink_time_step, 1).device(device) = sink_input_tensor.chip(sink_time_step, 1).cwiseMax((source_output_tensor.chip(source_time_step, 1) * weight_tensor.broadcast(bcast)).maximum(dims));
		}
		std::string getName() const { return "MaxTensorOp"; };
	};

	/**
		@brief Fan In Mean integration function
	*/
	template<typename TensorT, typename DeviceT>
	class FanInMeanOp : public IntegrationTensorOp<TensorT, DeviceT>
	{
	public:
		FanInMeanOp() {};
		~FanInMeanOp() {};
		void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			assert(sink_layer_size == 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight(weights, 1, source_layer_size);
			Eigen::array<int, 2> bcast = { batch_size, 0 };
			Eigen::array<int, 1> dims({ 1 });
			sink_input_tensor.chip(sink_time_step, 1).device(device) = (source_output_tensor.chip(source_time_step, 1) * weight.broadcast(bcast)).mean(dims);
		}
		std::string getName() const { return "FanInMeanOp"; };
	};

	/**
		@brief Mean integration function
	*/
	template<typename TensorT, typename DeviceT>
	class MeanTensorOp : public IntegrationTensorOp<TensorT, DeviceT>
	{
	public:
		MeanTensorOp() {};
		~MeanTensorOp() {};
		void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> weight(weights, 1, source_layer_size, sink_layer_size);
			Eigen::array<int, 2> bcast = { batch_size, 0 };
			Eigen::array<int, 1> dims({ 1 });
			sink_input_tensor.chip(sink_time_step, 1).device(device) = (source_output_tensor.chip(source_time_step, 1) * weight.broadcast(bcast)).mean(dims);
		}
		std::string getName() const { return "MeanTensorOp"; };
	};

	/**
		@brief Fan In Var integration function
	*/
	template<typename TensorT, typename DeviceT>
	class FanInVarOp : public IntegrationTensorOp<TensorT, DeviceT>
	{
	public:
		FanInVarOp() {};
		~FanInVarOp() {};
		void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			assert(sink_layer_size == 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight(weights, 1, source_layer_size);
			Eigen::array<int, 2> bcast = { batch_size, 0 };
			Eigen::array<int, 1> dims({ 1 });
			auto mean = (source_output_tensor.chip(source_time_step, 1) * weight.broadcast(bcast)).mean(dims);
			auto input = (source_output_tensor.chip(source_time_step, 1) * weight.broadcast(bcast)) - mean;
			sink_input_tensor.chip(sink_time_step, 1).device(device) = (input * input)*input.constant(1 / (TensorT)source_layer_size);
		}
		std::string getName() const { return "FanInVarOp"; };
	};

	/**
		@brief Var integration function
	*/
	template<typename TensorT, typename DeviceT>
	class VarTensorOp : public IntegrationTensorOp<TensorT, DeviceT>
	{
	public:
		VarTensorOp() {};
		~VarTensorOp() {};
		void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> weight(weights, 1, source_layer_size, sink_layer_size);
			Eigen::array<int, 2> bcast = { batch_size, 0 };
			Eigen::array<int, 1> dims({ 1 });
			auto mean = (source_output_tensor.chip(source_time_step, 1) * weight.broadcast(bcast)).mean(dims);
			auto input = (source_output_tensor.chip(source_time_step, 1) * weight.broadcast(bcast)) - mean;
			sink_input_tensor.chip(sink_time_step, 1).device(device) = (input * input)*input.constant(1 / (TensorT)source_layer_size);
		}
		std::string getName() const { return "VarTensorOp"; };
	};

	/**
		@brief Fan In Count integration function
	*/
	template<typename TensorT, typename DeviceT>
	class FanInCountOp : public IntegrationTensorOp<TensorT, DeviceT>
	{
	public:
		FanInCountOp() {};
		~FanInCountOp() {};
		void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			assert(sink_layer_size == 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
			sink_input_tensor.chip(sink_time_step, 1).device(device) = sink_input_tensor.chip(sink_time_step, 1).constant(source_layer_size);
		}
		std::string getName() const { return "FanInCountOp"; };
	};

	/**
		@brief Count integration function
	*/
	template<typename TensorT, typename DeviceT>
	class CountTensorOp : public IntegrationTensorOp<TensorT, DeviceT>
	{
	public:
		CountTensorOp() {};
		~CountTensorOp() {};
		void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
			sink_input_tensor.chip(sink_time_step, 1).device(device) += sink_input_tensor.chip(sink_time_step, 1).constant(source_layer_size);
		}
		std::string getName() const { return "CountTensorOp"; };
	};

	/**
	@brief Base class for all integration error functions.
	*/
	template<typename TensorT, typename DeviceT>
	class IntegrationErrorTensorOp
	{
	public:
		IntegrationErrorTensorOp() = default;
		IntegrationErrorTensorOp(const TensorT& eps) : eps_(eps) {};
		~IntegrationErrorTensorOp() = default;
		virtual std::string getName() const = 0;
		/*
		@brief Integration error void operator
		*/
		virtual void operator()(TensorT* source_error, TensorT *source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, TensorT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) = 0;
	protected:
		TensorT eps_ = 1e-6;
	};

	/**
	@brief Fully Connected Sum integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class FullyConnectedSumErrorOp : public IntegrationErrorTensorOp<TensorT, DeviceT>
	{
	public:
		FullyConnectedSumErrorOp() {};
		~FullyConnectedSumErrorOp() {};
		void operator()(TensorT* source_error, TensorT *source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, TensorT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_derivative_tensor(sink_derivative, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_error_tensor(source_error, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight_tensor(weight, sink_layer_size, source_layer_size); // NOTE: source/sink are reversed
			Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) }; // NOTE: we are taking the transpose of the weight matrix
			sink_error_tensor.chip(sink_time_step, 1).device(device) += (source_error_tensor.chip(source_time_step, 1)).contract(weight_tensor.shuffle(Eigen::array<int, 2>({ 1, 0 })), product_dims) * (sink_derivative_tensor.chip(source_time_step, 1));
		};
		std::string getName() const { return "FullyConnectedSumErrorOp"; };
	};

	/**
	@brief Sum integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class SinglyConnectedSumErrorOp : public IntegrationErrorTensorOp<TensorT, DeviceT>
	{
	public:
		SinglyConnectedSumErrorOp() {};
		~SinglyConnectedSumErrorOp() {};
		void operator()(TensorT* source_error, TensorT *source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, TensorT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device)  {
			assert(sink_layer_size == source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_derivative_tensor(sink_derivative, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_error_tensor(source_error, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight_tensor(weight, 1, sink_layer_size); // Note: sink/source are in the backward direction!
			Eigen::array<int, 2> bcast = { batch_size, 1 };
			sink_error_tensor.chip(sink_time_step, 1).device(device) += source_error_tensor.chip(source_time_step, 1) * weight_tensor.broadcast(bcast);
		};
		std::string getName() const { return "SinglyConnectedSumErrorOp"; };
	};

	/**
	@brief Fan In Sum integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class FanInSumErrorOp : public IntegrationErrorTensorOp<TensorT, DeviceT>
	{
	public:
		FanInSumErrorOp() {};
		~FanInSumErrorOp() {};
		void operator()(TensorT* source_error, TensorT *source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, TensorT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			assert(source_layer_size == 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_derivative_tensor(sink_derivative, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_error_tensor(source_error, batch_size, memory_size, 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight_tensor(weight, 1, sink_layer_size); // Note: sink/source are in the backward direction!
			Eigen::array<int, 2> bcast_batch = { batch_size, 1 };
			Eigen::array<int, 2> bcast_source = { sink_layer_size, 1 };
			Eigen::array<int, 1> dims({ 1 });
			sink_error_tensor.chip(sink_time_step, 1).device(device) += ((source_error_tensor.chip(source_time_step, 1)).broadcast(bcast_source) * weight_tensor.broadcast(bcast_batch)).sum(dims);
		};
		std::string getName() const { return "FanInSumErrorOp"; };
	};

	/**
	@brief Fan Out Sum integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class FanOutSumErrorOp : public IntegrationErrorTensorOp<TensorT, DeviceT>
	{
	public:
		FanOutSumErrorOp() {};
		~FanOutSumErrorOp() {};
		void operator()(TensorT* source_error, TensorT *source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, TensorT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			assert(sink_layer_size == 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_derivative_tensor(sink_derivative, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_error_tensor(source_error, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight_tensor(weight, 1, source_layer_size); // Note: sink/source are in the backward direction!
			Eigen::array<int, 2> bcast_batch = { batch_size, 1 };
			Eigen::array<int, 1> dims({ 1 });
			sink_error_tensor.chip(sink_time_step, 1).device(device) += (source_error_tensor.chip(source_time_step, 1) * weight_tensor.broadcast(bcast_batch)).sum(dims);
		};
		std::string getName() const { return "FanOutSumErrorOp"; };
	};

	/**
	@brief Fully Connected Sum integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class SumErrorTensorOp : public IntegrationErrorTensorOp<TensorT, DeviceT>
	{
	public:
		SumErrorTensorOp() {};
		~SumErrorTensorOp() {};
		void operator()(TensorT* source_error, TensorT *source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, TensorT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_derivative_tensor(sink_derivative, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_error_tensor(source_error, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight_tensor(weight, sink_layer_size, source_layer_size); // NOTE: source/sink are reversed
			Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) }; // NOTE: we are taking the transpose of the weight matrix
			sink_error_tensor.chip(sink_time_step, 1).device(device) += (source_error_tensor.chip(source_time_step, 1)).contract(weight_tensor.shuffle(Eigen::array<int, 2>({ 1, 0 })), product_dims) * (sink_derivative_tensor.chip(source_time_step, 1));
		};
		std::string getName() const { return "SumErrorTensorOp"; };
	};

	/**
	@brief Product integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class ProdErrorTensorOp : public IntegrationErrorTensorOp<TensorT, DeviceT>
	{
	public:
		ProdErrorTensorOp() {};
		~ProdErrorTensorOp() {};
		void operator()(TensorT* source_error, TensorT *source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, TensorT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device)  {
			//TODO
			
			//return (source_net_input * source_error / (sink_output + eps)).unaryExpr(ClipOp<TensorT>(1e-6, -1e9, 1e9));

			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_derivative_tensor(sink_derivative, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_error_tensor(source_error, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_input_tensor(source_input, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight_tensor(weight, sink_layer_size, source_layer_size); // NOTE: source/sink are reversed
			Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) }; // NOTE: we are taking the transpose of the weight matrix
			sink_error_tensor.chip(sink_time_step, 1).device(device) = (source_error_tensor.chip(source_time_step, 1)).contract(weight_tensor.shuffle(Eigen::array<int, 2>({ 1, 0 })), product_dims) * (sink_derivative_tensor.chip(source_time_step, 1));

			//Eigen::array<int, 2> bcast = { batch_size, 1 };
			//Eigen::array<int, 1> dims({ 1 });
			//sink_input_tensor.chip(sink_time_step, 1).device(device) = sink_input_tensor.chip(sink_time_step, 1) * (source_output_tensor.chip(source_time_step, 1) * weight.broadcast(bcast)).prod(dims);

		};
		std::string getName() const { return "ProdErrorTensorOp"; };
	};

	/**
	@brief Max integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class MaxErrorTensorOp : public IntegrationErrorTensorOp<TensorT, DeviceT>
	{
	public:
		MaxErrorTensorOp() {};
		~MaxErrorTensorOp() {};
		void operator()(TensorT* source_error, TensorT *source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, TensorT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) 
		{
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_output_tensor(sink_output, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_derivative_tensor(sink_derivative, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> source_input_tensor(source_input, batch_size, memory_size, source_layer_size, 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_error_tensor(source_error, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> weight_tensor(weight, 1, sink_layer_size, source_layer_size); // NOTE: source/sink are reversed
			// step 1: determine the maximum
			auto comp_tensor = source_output_tensor.chip(source_time_step, 1) * weight_tensor.shuffle(Eigen::array<int, 2>({ 1, 0 })).broadcast(Eigen::array<int, 2>({ batch_size, 0 }));
			auto max_tensor = source_input_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 2>({ sink_layer_size, 3 }));
			auto selection_tensor = ((max_tensor - comp_tensor) > (0 - max_tensor.constant(1e-6)).select(max_tensor.constant(1), max_tensor.constant(0));

			// step 2: select out the error to propogate
			auto error = source_error_tensor.chip(source_time_step, 1) * weight_tensor.shuffle(Eigen::array<int, 2>({ 1, 0 })).broadcast(Eigen::array<int, 2>({ batch_size, 0 }));
			auto selected_error = (error * selection_tensor).sum(Eigen::array<int, 1>({ 1 })); // sum along the source layer
			
			sink_error_tensor.chip(sink_time_step, 1).device(device) += selected_error * sink_derivative_tensor.chip(source_time_step, 1);
		};
		std::string getName() const { return "MaxErrorTensorOp"; };
	};

	/**
	@brief Mean integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class MeanErrorTensorOp : public IntegrationErrorTensorOp<TensorT, DeviceT>
	{
	public:
		MeanErrorTensorOp() {};
		~MeanErrorTensorOp() {};
		void operator()(TensorT* source_error, TensorT *source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, TensorT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device)  {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_derivative_tensor(sink_derivative, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_error_tensor(source_error, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight_tensor(weight, sink_layer_size, source_layer_size); // NOTE: source/sink are reversed
			Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) }; // NOTE: we are taking the transpose of the weight matrix
			sink_error_tensor.chip(sink_time_step, 1).device(device) += (source_error_tensor.chip(source_time_step, 1)).contract(weight_tensor.shuffle(Eigen::array<int, 2>({ 1, 0 })), product_dims) * sink_error_tensor.constant(1/n_input_nodes) * (sink_derivative_tensor.chip(source_time_step, 1));
		};
		std::string getName() const { return "MeanErrorTensorOp"; };
	};

	/**
	@brief VarMod integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class VarErrorTensorOp : public IntegrationErrorTensorOp<TensorT, DeviceT>
	{
	public:
		VarErrorTensorOp() {};
		~VarErrorTensorOp() {};
		void operator()(TensorT* source_error, TensorT *source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, TensorT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device)  {
			//TODO
		};
		std::string getName() const { return "VarErrorTensorOp"; };
	};

	/**
	@brief Count integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class CountErrorTensorOp : public IntegrationErrorTensorOp<TensorT, DeviceT>
	{
	public:
		CountErrorTensorOp() {};
		~CountErrorTensorOp() {};
		void operator()(TensorT* source_error, TensorT *source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, TensorT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device)  {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
			sink_error_tensor.chip(sink_time_step, 1).device(device) = sink_error_tensor.chip(sink_time_step, 1).constant(0);
		};
		std::string getName() const { return "CountErrorTensorOp"; };
	};

	/**
	@brief Base class for all integration error functions.
	*/
	template<typename TensorT, typename DeviceT>
	class IntegrationWeightGradTensorOp
	{
	public:
		IntegrationWeightGradTensorOp() = default;
		IntegrationWeightGradTensorOp(const TensorT& eps) : eps_(eps) {};
		~IntegrationWeightGradTensorOp() = default;
		virtual std::string getName() const = 0;
		virtual void operator()(TensorT* sink_error, TensorT* source_output, TensorT* weight, TensorT* source_input, TensorT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) = 0;
	protected:
		TensorT eps_ = 1e-6;
	};

	/**
	@brief Fully Connected Sum integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class FullyConnectedSumWeightGradOp : public IntegrationWeightGradTensorOp<TensorT, DeviceT>
	{
	public:
		void operator()(TensorT* sink_error, TensorT* source_output, TensorT* weight, TensorT* source_input, TensorT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device){
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight_error_tensor(weight_error, source_layer_size, sink_layer_size);

			Eigen::array<Eigen::IndexPair<int>, 2> double_contraction_product_dims = { Eigen::IndexPair<int>(1,1), Eigen::IndexPair<int>(0,0) };
			auto tmp = -source_output_tensor.contract(sink_error_tensor, double_contraction_product_dims);
			// NOTE: Double contraction along the memory and batch (equivalent to a double sum along the products of the batch and memory dimensions)
			weight_error_tensor.device(device) += tmp * weight_error_tensor.constant(1 / (TensorT)batch_size);
			// NOTE: Requires a correction by dividing by the batch size
		};
		std::string getName() const { return "FullyConnectedSumWeightGradOp"; };
	};

	/**
	@brief Singly Connected Sum integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class SinglyConnectedSumWeightGradOp : public IntegrationWeightGradTensorOp<TensorT, DeviceT>
	{
	public:
		void operator()(TensorT* sink_error, TensorT* source_output, TensorT* weight, TensorT* source_input, TensorT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
			assert(sink_layer_size == source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size, 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size, 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight_error_tensor(weight_error, source_layer_size, sink_layer_size);
			auto tmp = -source_output_tensor.broadcast(Eigen::array<int, 4>({ 1, 1, 1, sink_layer_size })) * sink_error_tensor.broadcast(Eigen::array<int, 4>({ 1, 1, 1, source_layer_size })).shuffle(Eigen::array<int, 4>({ 0, 1, 3, 2 }));
			weight_error_tensor.device(device) += tmp.sum(Eigen::array<int, 1>({ 1 })).mean(Eigen::array<int, 1>({ 0 }));
		};
		std::string getName() const { return "SinglyConnectedSumWeightGradOp"; };
	};

	/**
	@brief Fully Connected Sum integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class SumWeightGradTensorOp : public IntegrationWeightGradTensorOp<TensorT, DeviceT>
	{
	public:
		void operator()(TensorT* sink_error, TensorT* source_output, TensorT* weight, TensorT* source_input, TensorT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight_error_tensor(weight_error, source_layer_size, sink_layer_size);

			Eigen::array<Eigen::IndexPair<int>, 2> double_contraction_product_dims = { Eigen::IndexPair<int>(1,1), Eigen::IndexPair<int>(0,0) };
			auto tmp = -source_output_tensor.contract(sink_error_tensor, double_contraction_product_dims);
			// NOTE: Double contraction along the memory and batch (equivalent to a double sum along the products of the batch and memory dimensions)
			weight_error_tensor.device(device) += tmp * weight_error_tensor.constant(1 / (TensorT)batch_size);
			// NOTE: Requires a correction by dividing by the batch size
		};
		std::string getName() const { return "SumWeightGradTensorOp"; };
	};

	/**
	@brief Product integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class ProdWeightGradTensorOp : public IntegrationWeightGradTensorOp<TensorT, DeviceT>
	{
	public:
		ProdWeightGradTensorOp() {};
		~ProdWeightGradTensorOp() {};
		void operator()(TensorT* sink_error, TensorT* source_output, TensorT* weight, TensorT* source_input, TensorT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
			// TODO
		};
		std::string getName() const { return "ProdWeightGradTensorOp"; };
	};

	/**
	@brief Max integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class MaxWeightGradTensorOp : public IntegrationWeightGradTensorOp<TensorT, DeviceT>
	{
	public:
		MaxWeightGradTensorOp() {};
		~MaxWeightGradTensorOp() {};
		void operator()(TensorT* sink_error, TensorT* source_output, TensorT* weight, TensorT* source_input, TensorT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight_error_tensor(weight_error, source_layer_size, sink_layer_size);

			Eigen::array<Eigen::IndexPair<int>, 2> double_contraction_product_dims = { Eigen::IndexPair<int>(1,1), Eigen::IndexPair<int>(0,0) };
			auto tmp = -source_output_tensor.contract(sink_error_tensor, double_contraction_product_dims);
			// NOTE: Double contraction along the memory and batch (equivalent to a double sum along the products of the batch and memory dimensions)
			weight_error_tensor.device(device) += tmp * weight_error_tensor.constant(1 / (TensorT)batch_size);
			// NOTE: Requires a correction by dividing by the batch size
		};
		std::string getName() const { return "MaxWeightGradTensorOp"; };
	};

	/**
	@brief Count integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class CountWeightGradTensorOp : public IntegrationWeightGradTensorOp<TensorT, DeviceT>
	{
	public:
		CountWeightGradTensorOp() {};
		~CountWeightGradTensorOp() {};
		void operator()(TensorT* sink_error, TensorT* source_output, TensorT* weight, TensorT* source_input, TensorT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight_error_tensor(weight_error, source_layer_size, sink_layer_size);
			weight_error_tensor.device(device) = weight_error_tensor.constant(0);
		};
		std::string getName() const { return "CountWeightGradTensorOp"; };
	};

	/**
	@brief Mean integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class MeanWeightGradTensorOp : public IntegrationWeightGradTensorOp<TensorT, DeviceT>
	{
	public:
		MeanWeightGradTensorOp() {};
		~MeanWeightGradTensorOp() {};
		void operator()(TensorT* sink_error, TensorT* source_output, TensorT* weight, TensorT* source_input, TensorT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight_error_tensor(weight_error, source_layer_size, sink_layer_size);

			Eigen::array<Eigen::IndexPair<int>, 2> double_contraction_product_dims = { Eigen::IndexPair<int>(1,1), Eigen::IndexPair<int>(0,0) };
			auto tmp = -source_output_tensor.contract(sink_error_tensor, double_contraction_product_dims);
			// NOTE: Double contraction along the memory and batch (equivalent to a double sum along the products of the batch and memory dimensions)
			weight_error_tensor.device(device) += tmp * weight_error_tensor.constant(1 / (TensorT)batch_size) * weight_error_tensor.constant(1 / (TensorT)n_input_nodes);
			// NOTE: Requires a correction by dividing by the batch size
		};
		std::string getName() const { return "MeanWeightGradTensorOp"; };
	};

	/**
	@brief Var integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class VarWeightGradTensorOp : public IntegrationWeightGradTensorOp<TensorT, DeviceT>
	{
	public:
		VarWeightGradTensorOp() {};
		~VarWeightGradTensorOp() {};
		void operator()(TensorT* sink_error, TensorT* source_output, TensorT* weight, TensorT* source_input, TensorT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
			// TODO
		};
		std::string getName() const { return "VarWeightGradTensorOp"; };
	};
}
#endif //SMARTPEAK_TENSORINTEGRATIONFUNCTION_H