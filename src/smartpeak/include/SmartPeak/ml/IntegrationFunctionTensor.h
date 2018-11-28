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
			Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size, 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> weight_tensor(weights, 1, source_layer_size, sink_layer_size);
			// NOTE this should be *=, but the starting values are all 0...
			//sink_input_tensor.chip(sink_time_step, 1).device(device) = sink_input_tensor.chip(sink_time_step, 1) * (
			//	source_output_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, 1, sink_layer_size })) *
			//	weight_tensor.broadcast(Eigen::array<int, 3>({ batch_size, 1, 1 }))
			//	).prod(Eigen::array<int, 1>({ 1 }));

			sink_input_tensor.chip(sink_time_step, 1).device(device) = (source_output_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, 1, sink_layer_size })) *
					weight_tensor.broadcast(Eigen::array<int, 3>({ batch_size, 1, 1 }))
				).prod(Eigen::array<int, 1>({ 1 }));
		}
		std::string getName() const { return "ProdTensorOp"; };
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
			Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size, 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> weight_tensor(weights, 1, source_layer_size, sink_layer_size);
			sink_input_tensor.chip(sink_time_step, 1).device(device) = sink_input_tensor.chip(sink_time_step, 1).cwiseMax(
				(source_output_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, 1, sink_layer_size })) *
					weight_tensor.broadcast(Eigen::array<int, 3>({ batch_size, 1, 1 }))
					).maximum(Eigen::array<int, 1>({ 1 })));
		}
		std::string getName() const { return "MaxTensorOp"; };
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
			Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size, 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> weight(weights, 1, source_layer_size, sink_layer_size);
			sink_input_tensor.chip(sink_time_step, 1).device(device) = (
				source_output_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, 1, sink_layer_size })) *
					weight.broadcast(Eigen::array<int, 3>({ batch_size, 1, 1 }))
				).mean(Eigen::array<int, 1>({ 1 }));
		}
		std::string getName() const { return "MeanTensorOp"; };
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
			Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size, 1, 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> weight(weights, 1, source_layer_size, 1, sink_layer_size);
			auto mean = (source_output_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 4>({ 1, 1, 1, sink_layer_size })) * weight.broadcast(Eigen::array<int, 4>({ batch_size, 1, 1, 1 }))).mean(Eigen::array<int, 1>({ 1 })).broadcast(Eigen::array<int, 3>({ 1, source_layer_size, 1 })); // dim3
			auto input = (source_output_tensor.chip(source_time_step, 1).chip(source_time_step, 3).broadcast(Eigen::array<int, 3>({ 1, 1, sink_layer_size })) * weight.chip(0, 2).broadcast(Eigen::array<int, 3>({ batch_size, 1, 1 })) - mean); // dim3
			sink_input_tensor.chip(sink_time_step, 1).device(device) = ((input * input)*input.constant(1 / (TensorT)source_layer_size)).sum(Eigen::array<int, 1>({ 1 }));
		}
		std::string getName() const { return "VarTensorOp"; };
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
			sink_error_tensor.chip(sink_time_step, 1).device(device) += (source_error_tensor.chip(source_time_step, 1)).contract(weight_tensor.shuffle(Eigen::array<int, 2>({ 1, 0 })), product_dims) * (sink_derivative_tensor.chip(sink_time_step, 1));
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
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> sink_output_tensor(sink_output, batch_size, memory_size, sink_layer_size, 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_derivative_tensor(sink_derivative, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> source_input_tensor(source_input, batch_size, memory_size, 1, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> source_error_tensor(source_error, batch_size, memory_size, 1, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> weight_tensor(weight, 1, sink_layer_size, source_layer_size); // NOTE: source/sink are reversed
			// step 1: re-compute the intermediate tensor and expand the net input (dims: batch, source, sink)
			auto comp_tensor = sink_output_tensor.chip(sink_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, 1, source_layer_size })) * weight_tensor.broadcast(Eigen::array<int, 3>({ batch_size, 1, 1 }));
			auto source_exp_input_tensor = source_input_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, sink_layer_size, 1 }));
			
			// step 2: divide out the comp_tensor, scale by the source error, and reduce by taking the sum along the source layer
			auto tmp = (source_exp_input_tensor * source_error_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, sink_layer_size, 1 })) / (comp_tensor + comp_tensor.constant(1e-6))).sum(Eigen::array<int, 1>({ 2 }));
			// NOTE this should be *=, but the sink error tensor is initialized to 0...
			sink_error_tensor.chip(sink_time_step, 1).device(device) += tmp.clip(-1e9,1e9) * sink_derivative_tensor.chip(sink_time_step, 1);

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
			Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> sink_output_tensor(sink_output, batch_size, memory_size, sink_layer_size, 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_derivative_tensor(sink_derivative, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> source_input_tensor(source_input, batch_size, memory_size, 1, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> source_error_tensor(source_error, batch_size, memory_size, 1, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> weight_tensor(weight, 1, sink_layer_size, source_layer_size); // NOTE: source/sink are reversed
			// step 1: determine the maximum
			auto comp_tensor = sink_output_tensor.chip(sink_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, 1, source_layer_size })) * weight_tensor.broadcast(Eigen::array<int, 3>({ batch_size, 1, 1 }));
			auto max_tensor = source_input_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, sink_layer_size, 1 }));
			auto selection_tensor = ((comp_tensor - max_tensor) > (max_tensor.constant((TensorT)0) - max_tensor.constant(1e-6))).select(max_tensor.constant(1), max_tensor.constant(0));

			// step 2: select out the error to propogate
			auto error = source_error_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, sink_layer_size, 1 })) * weight_tensor.broadcast(Eigen::array<int, 3>({ batch_size, 1, 1 }));
			auto selected_error = (error * selection_tensor).sum(Eigen::array<int, 1>({ 2 })); // sum along the source layer
			
			sink_error_tensor.chip(sink_time_step, 1).device(device) += selected_error * sink_derivative_tensor.chip(sink_time_step, 1);
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
			sink_error_tensor.chip(sink_time_step, 1).device(device) += (source_error_tensor.chip(source_time_step, 1)).contract(weight_tensor.shuffle(Eigen::array<int, 2>({ 1, 0 })), product_dims) * sink_error_tensor.chip(sink_time_step, 1).constant(1/(TensorT)n_input_nodes) * (sink_derivative_tensor.chip(sink_time_step, 1));
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
			Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> sink_error_tensor(sink_error, batch_size, memory_size, 1, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> source_input_tensor(source_input, batch_size, memory_size, source_layer_size, 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> weight_tensor(weight, 1, 1, source_layer_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight_error_tensor(weight_error, source_layer_size, sink_layer_size);
			// step 1: compute the weight-normalized source net input expanded across batch and memory
			auto input_normalized_tensor = source_input_tensor.broadcast(Eigen::array<int, 4>({ 1, 1, 1, sink_layer_size })) / weight_tensor.broadcast(Eigen::array<int, 4>({ batch_size, memory_size, 1, 1 }));
			// step 2: scale to the sink error
			auto scaled_error = -sink_error_tensor.broadcast(Eigen::array<int, 4>({ 1, 1, source_layer_size, 1 })) * input_normalized_tensor;
			// step 3: sum along the memory and average along the batch dimensions
			weight_error_tensor.device(device) += scaled_error.sum(Eigen::array<int, 2>({ 0, 1 })) * weight_error_tensor.constant(1 / (TensorT)batch_size);
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
			// NOTE : same as SumWeightGrad given the assumption that the errors for non max links have been zero'd out previously
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