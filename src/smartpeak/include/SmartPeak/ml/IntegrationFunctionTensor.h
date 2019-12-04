/**TODO:  Add copyright*/

#ifndef SMARTPEAK_TENSORINTEGRATIONFUNCTION_H
#define SMARTPEAK_TENSORINTEGRATIONFUNCTION_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <SmartPeak/core/Preprocessing.h>
#include <SmartPeak/ml/ActivationFunctionTensor.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <typeinfo>

//#include <cereal/access.hpp>  // serialiation of private members
//#undef min // clashes with std::limit on windows in polymorphic.hpp
//#undef max // clashes with std::limit on windows in polymorphic.hpp
//#include <cereal/types/polymorphic.hpp>

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
		TensorT eps_ = TensorT(1e-24);
    TensorT min_ = TensorT(-1e9);
    TensorT max_ = TensorT(1e9);
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(eps_);
	//	}
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
      operator_(source_output, weights, sink_input, batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);
		};
    template<typename TT=TensorT, std::enable_if_t<std::is_same<TT, double>::value, int> = 0>
    void operator_(TT* source_output, TT* weights, TT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device){
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 4>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> weight_tensor(weights, 1, source_layer_size, sink_layer_size);
      auto weight_tensor_exp = weight_tensor.broadcast(Eigen::array<int, 3>({ batch_size, 1, 1 }));
      auto source_bcast = source_output_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, 1, sink_layer_size }));
      sink_input_tensor.chip(sink_time_step, 1).device(device) += (source_bcast * weight_tensor_exp).sum(Eigen::array<int, 1>({ 1 })).clip(this->min_, this->max_);
    }
    template<typename TT = TensorT, std::enable_if_t<!std::is_same<TT, double>::value, int> = 0>
    void operator_(TT* source_output, TT* weights, TT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device){
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 2>> weight_tensor(weights, source_layer_size, sink_layer_size);
      Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
      sink_input_tensor.chip(sink_time_step, 1).device(device) += (source_output_tensor.chip(source_time_step, 1)).contract(weight_tensor, product_dims).clip(this->min_, this->max_);
    }
		std::string getName() const { return "SumTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
		@brief Prod integration function
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

      //// DEBUG (only on CPU)
      //std::cout << "[ProdTensorOp]Sink (Start): " << sink_input_tensor.chip(sink_time_step, 1) << std::endl;

      // Step 1: expand source across the sink layer dim and weight tensor across the batch dim and multiply
      auto weight_tensor_exp = weight_tensor.broadcast(Eigen::array<int, 3>({ batch_size, 1, 1 }));
      auto source_bcast = source_output_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, 1, sink_layer_size }));

      TensorT* tmp_data;
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        tmp_data = new TensorT[batch_size*source_layer_size*sink_layer_size];
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        size_t bytes = batch_size * source_layer_size*sink_layer_size * sizeof(TensorT);
        assert(cudaMalloc((void**)(&tmp_data), bytes) == cudaSuccess);
      }
#endif
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_weight_exp(tmp_data, batch_size, source_layer_size, sink_layer_size);
      source_weight_exp.device(device) = source_bcast * weight_tensor_exp;

      // Step 2: determine where the 0s in the original input are propogated to in the source_weight_exp tensor
      auto source_1 = (source_output_tensor.chip(source_time_step, 1) == source_output_tensor.chip(source_time_step, 1).constant(TensorT(0))).select(
      //auto source_1 = (source_output_tensor.chip(source_time_step, 1) > source_output_tensor.chip(source_time_step, 1).constant(-this->eps_) &&
      //  source_output_tensor.chip(source_time_step, 1) < source_output_tensor.chip(source_time_step, 1).constant(this->eps_)).select(
          source_output_tensor.chip(source_time_step, 1).constant(TensorT(1)), source_output_tensor.chip(source_time_step, 1).constant(TensorT(0)));
      auto source_weight_exp_1 = source_1.broadcast(Eigen::array<int, 3>({ 1, 1, sink_layer_size }))*weight_tensor_exp;

      // Step 3: Substitute 1 for all 0 entries (assuming 0s are non entries) except for the 0s that were propogated from the source output
      auto source_weight_1 = (
        (source_weight_exp == source_weight_exp.constant(TensorT(0))) && (source_weight_exp_1 != source_weight_exp.constant(TensorT(1)))
      //auto source_weight_1 = (
      //  (source_weight_exp > source_weight_exp.constant(-this->eps_) && source_weight_exp < source_weight_exp.constant(this->eps_)) &&
      //  (source_weight_exp_1 < source_weight_exp.constant(TensorT(1) - this->eps_) || source_weight_exp_1 > source_weight_exp.constant(TensorT(1) + this->eps_))
        ).select(source_weight_exp.constant(TensorT(1)), source_weight_exp);

      // Step 4: multiply along the source dim
			sink_input_tensor.chip(sink_time_step, 1).device(device) = sink_input_tensor.chip(sink_time_step, 1) * (source_weight_1
				).prod(Eigen::array<int, 1>({ 1 })).clip(this->min_, this->max_);

      // Deallocate temporary memory
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        delete[] tmp_data;
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        assert(cudaFree(tmp_data) == cudaSuccess);
      }
#endif
		}
		std::string getName() const { return "ProdTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

  /**
    @brief Prod Singly Connected integration function
  */
  template<typename TensorT, typename DeviceT>
  class ProdSCTensorOp : public IntegrationTensorOp<TensorT, DeviceT>
  {
  public:
    ProdSCTensorOp() {};
    ~ProdSCTensorOp() {};
    void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
      //assert(source_layer_size == sink_layer_size);

      // NOTE: Should work with optimized Weight tensors but this requires specialized methods for the solvers
      //Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
      //Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
      //Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight_tensor(weights, 1, source_layer_size);

      //sink_input_tensor.chip(sink_time_step, 1).device(device) = sink_input_tensor.chip(sink_time_step, 1) *
      //  source_output_tensor.chip(source_time_step, 1) * weight_tensor.broadcast(Eigen::array<int, 2>({ batch_size, 1}));

      // NOTE: Works for diagonal weight tensors between source and sink layers
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> weight_tensor(weights, 1, source_layer_size, sink_layer_size);

      sink_input_tensor.chip(sink_time_step, 1).device(device) = sink_input_tensor.chip(sink_time_step, 1) * (
        source_output_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, 1, sink_layer_size })) *
        weight_tensor.broadcast(Eigen::array<int, 3>({ batch_size, 1, 1 }))
        ).sum(Eigen::array<int, 1>({ 1 }));  // NOTE the use of sum instead of prod here

      //// DEBUG (only on CPU)
      //std::cout << "[ProdSCTensorOp]Source: " << source_output_tensor.chip(source_time_step, 1) << std::endl;
      //std::cout << "[ProdSCTensorOp]Weight: " << weight_tensor << std::endl;
      //std::cout << "[ProdSCTensorOp]Intermediate: " << source_output_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, 1, sink_layer_size })) *
      //  weight_tensor.broadcast(Eigen::array<int, 3>({ batch_size, 1, 1 })) << std::endl;
      //std::cout << "[ProdSCTensorOp]Sink (End): " << sink_input_tensor.chip(sink_time_step, 1) << std::endl;
    }
    std::string getName() const { return "ProdSCTensorOp"; };
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(cereal::base_class<IntegrationTensorOp<TensorT, DeviceT>>(this));
    //	}
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
					).maximum(Eigen::array<int, 1>({ 1 }))).clip(this->min_, this->max_);
		}
		std::string getName() const { return "MaxTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

  /**
    @brief Min integration function
  */
  template<typename TensorT, typename DeviceT>
  class MinTensorOp : public IntegrationTensorOp<TensorT, DeviceT>
  {
  public:
    MinTensorOp() {};
    ~MinTensorOp() {};
    void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> weight_tensor(weights, 1, source_layer_size, sink_layer_size);

      // Step 1: Substitute 1e24 for all 0 entries (assuming 0s are non entries) in the input
      auto sink_input_chip = sink_input_tensor.chip(sink_time_step, 1);
      auto sink_input_large = (
        //sink_input_chip > sink_input_chip.constant(-this->eps_) && sink_input_chip < sink_input_chip.constant(this->eps_)
        sink_input_chip == sink_input_chip.constant(TensorT(0))
        ).select(sink_input_chip.constant(TensorT(1e24)), sink_input_chip).eval();
      
      // Step 2: expand source across the sink layer dim and weight tensor across the batch dim and multiply
      auto weight_tensor_exp = weight_tensor.broadcast(Eigen::array<int, 3>({ batch_size, 1, 1 }));
      auto source_weight_exp = (source_output_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, 1, sink_layer_size })).eval() *  weight_tensor_exp);
      
      // Step 3: Substitute 1e24 for all 0 entries (assuming 0s are non entries)
      //         This unfortunately requires temporary memory to remain under 4096 bytes
      TensorT* tmp_data;
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        tmp_data = new TensorT[batch_size*source_layer_size*sink_layer_size];
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        size_t bytes = batch_size * source_layer_size*sink_layer_size * sizeof(TensorT);
        assert(cudaMalloc((void**)(&tmp_data), bytes) == cudaSuccess);
      }
#endif
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> source_weight_1(tmp_data, batch_size, source_layer_size, sink_layer_size);
      source_weight_1.device(device) = (
        //weight_tensor_exp > weight_tensor_exp.constant(-this->eps_) && weight_tensor_exp < weight_tensor_exp.constant(this->eps_)
        weight_tensor_exp == weight_tensor_exp.constant(TensorT(0))
        ).select(source_weight_exp.constant(TensorT(1e24)), source_weight_exp).eval();
      
      // Step 4: Take the Minimum along the source dim
      auto sink_input_tensor_tmp = sink_input_large.cwiseMin(source_weight_1.minimum(Eigen::array<int, 1>({ 1 })));
     
      // Step 5: Replace all 1e24 with 0
      sink_input_tensor.chip(sink_time_step, 1).device(device) = (sink_input_tensor_tmp == sink_input_tensor_tmp.constant(TensorT(1e24))).select(sink_input_tensor_tmp.constant(TensorT(0)), sink_input_tensor_tmp).clip(this->min_, this->max_);

      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        delete[] tmp_data;
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        assert(cudaFree(tmp_data) == cudaSuccess);
      }
#endif
    }
    std::string getName() const { return "MinTensorOp"; };
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(cereal::base_class<IntegrationTensorOp<TensorT, DeviceT>>(this));
    //	}
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
				).mean(Eigen::array<int, 1>({ 1 })).clip(this->min_, this->max_);
		}
		std::string getName() const { return "MeanTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
		@brief VarMod integration function

		Modified variance integration function: 1/n Sum[0 to n](Xi)^2
		where Xi = xi - u (u: mean, xi: single sample)
	*/
	template<typename TensorT, typename DeviceT>
	class VarModTensorOp : public IntegrationTensorOp<TensorT, DeviceT>
	{
	public:
		VarModTensorOp() {};
		~VarModTensorOp() {};
		void operator()(TensorT* source_output, TensorT* weights, TensorT* sink_input, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_input_tensor(sink_input, batch_size, memory_size, sink_layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size, 1);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> weight(weights, 1, source_layer_size, sink_layer_size);
			auto input = source_output_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, 1, sink_layer_size })) * weight.broadcast(Eigen::array<int, 3>({ batch_size, 1, 1 })); // dim3
			sink_input_tensor.chip(sink_time_step, 1).device(device) = ((input * input)*input.constant(TensorT(1) / (TensorT)source_layer_size)).sum(Eigen::array<int, 1>({ 1 })).clip(this->min_, this->max_);
		}
		std::string getName() const { return "VarModTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationTensorOp<TensorT, DeviceT>>(this));
	//	}
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
			sink_input_tensor.chip(sink_time_step, 1).device(device) = ((input * input)*input.constant(TensorT(1) / (TensorT)source_layer_size)).sum(Eigen::array<int, 1>({ 1 })).clip(this->min_, this->max_);
		}
		std::string getName() const { return "VarTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationTensorOp<TensorT, DeviceT>>(this));
	//	}
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
			sink_input_tensor.chip(sink_time_step, 1).device(device) += sink_input_tensor.chip(sink_time_step, 1).constant((TensorT)source_layer_size).clip(this->min_, this->max_);
		}
		std::string getName() const { return "CountTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationTensorOp<TensorT, DeviceT>>(this));
	//	}
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
		TensorT eps_ = TensorT(1e-24);
    TensorT min_ = TensorT(-1e9);
    TensorT max_ = TensorT(1e9);
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(eps_);
	//	}
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
      operator_(source_error, source_input, weight, sink_output, sink_error, sink_derivative, n_input_nodes, batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);
		};
    template<typename TT = TensorT, std::enable_if_t<std::is_same<TT, double>::value, int> = 0>
    void operator_(TT* source_error, TT* source_input, TT* weight, TT* sink_output, TT* sink_error, TT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> sink_derivative_tensor(sink_derivative, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 4>> source_error_tensor(source_error, batch_size, memory_size, 1, source_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> weight_tensor(weight, 1, sink_layer_size, source_layer_size); // NOTE: source/sink are reversed
      auto source_error_bcast = source_error_tensor.chip(source_time_step, 1).broadcast(Eigen::array<Eigen::Index, 3>({ 1, sink_layer_size, 1 }));
      auto weight_bcast = weight_tensor.broadcast(Eigen::array<Eigen::Index, 3>({ batch_size, 1, 1 }));
      sink_error_tensor.chip(sink_time_step, 1).device(device) += ((source_error_bcast * weight_bcast).sum(Eigen::array<int, 1>({ 2 })) * sink_derivative_tensor.chip(sink_time_step, 1)).clip(this->min_, this->max_);
    }
    template<typename TT = TensorT, std::enable_if_t<!std::is_same<TT, double>::value, int> = 0>
    void operator_(TT* source_error, TT* source_input, TT* weight, TT* sink_output, TT* sink_error, TT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> sink_derivative_tensor(sink_derivative, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> source_error_tensor(source_error, batch_size, memory_size, source_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 2>> weight_tensor(weight, sink_layer_size, source_layer_size); // NOTE: source/sink are reversed
      Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) }; // NOTE: we are taking the transpose of the weight matrix
      sink_error_tensor.chip(sink_time_step, 1).device(device) += ((source_error_tensor.chip(source_time_step, 1)).contract(weight_tensor.shuffle(Eigen::array<int, 2>({ 1, 0 })), product_dims) * (sink_derivative_tensor.chip(sink_time_step, 1))).clip(this->min_, this->max_);
    }
		std::string getName() const { return "SumErrorTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationErrorTensorOp<TensorT, DeviceT>>(this));
	//	}
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
      // NOTE for numerical stability, we multiply by the comp_tensor in order to zero out non contributing elements,
      //      which otherwise would result in an very large error even though their contribution was 0,
      //      and then divide by the square of the comp_tensor plus a small constant to avoid division by 0
      TensorT* tmp_data;
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        tmp_data = new TensorT[batch_size*sink_layer_size*source_layer_size];
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        size_t bytes = batch_size * sink_layer_size * source_layer_size * sizeof(TensorT);
        assert(cudaMalloc((void**)(&tmp_data), bytes) == cudaSuccess);
      }
#endif
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> comp_tensor_clipped_neg(tmp_data, batch_size, sink_layer_size, source_layer_size);
      // Step 2 Option 1
      //auto tmp = (source_exp_input_tensor * source_error_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, sink_layer_size, 1 }))
      //  * comp_tensor / (comp_tensor * comp_tensor + comp_tensor.constant(this->eps_))).sum(Eigen::array<int, 1>({ 2 }));

      // calculate numerator
      auto tmp_numerator = source_exp_input_tensor * source_error_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, sink_layer_size, 1 }));

      // remove small values (both positive and negative) from the intermediate tensor
      comp_tensor_clipped_neg.device(device) = (comp_tensor > comp_tensor.constant(1 / this->min_) &&
        comp_tensor < comp_tensor.constant(TensorT(0))).select(
          comp_tensor.constant(1 / this->min_), comp_tensor);
      auto comp_tensor_clipped_pos = (comp_tensor_clipped_neg <= comp_tensor_clipped_neg.constant(1 / this->max_) &&
        comp_tensor > comp_tensor.constant(TensorT(0))).select(
          comp_tensor_clipped_neg.constant(1 / this->max_), comp_tensor_clipped_neg);

      // remove all 0's from the intermediate tensor and finish the calculation
      auto tmp_non_zero = (comp_tensor_clipped_neg != comp_tensor_clipped_neg.constant(TensorT(0))).select(
        tmp_numerator / comp_tensor_clipped_pos, comp_tensor_clipped_neg.constant(TensorT(0))).sum(Eigen::array<int, 1>({ 2 }));

      sink_error_tensor.chip(sink_time_step, 1).device(device) += (tmp_non_zero * sink_derivative_tensor.chip(sink_time_step, 1)).clip(this->min_, this->max_);

      // Deallocate temporary memory
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        delete[] tmp_data;
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        assert(cudaFree(tmp_data) == cudaSuccess);
      }
#endif

      //// DEBUG (only on CPU)
      //std::cout << "[ProdErrorTensorOp]comp_tensor: " << comp_tensor << std::endl;
      //std::cout << "[ProdErrorTensorOp]tmp: " << tmp << std::endl;
      //std::cout << "[ProdErrorTensorOp]sink_error_tensor (End): " << sink_error_tensor.chip(sink_time_step, 1) << std::endl;
		};
		std::string getName() const { return "ProdErrorTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationErrorTensorOp<TensorT, DeviceT>>(this));
	//	}
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
			auto selection_tensor = ((comp_tensor - max_tensor).abs() > (max_tensor.constant(TensorT(0)) - max_tensor.constant(this->eps_)) &&
        (comp_tensor - max_tensor).abs() < (max_tensor.constant(TensorT(0)) + max_tensor.constant(this->eps_))).select(max_tensor.constant(TensorT(1)), max_tensor.constant(TensorT(0)));

			// step 2: select out the error to propogate
			auto error = source_error_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, sink_layer_size, 1 })) * weight_tensor.broadcast(Eigen::array<int, 3>({ batch_size, 1, 1 }));
			auto selected_error = (error * selection_tensor).sum(Eigen::array<int, 1>({ 2 })); // sum along the source layer
			
			sink_error_tensor.chip(sink_time_step, 1).device(device) += (selected_error * sink_derivative_tensor.chip(sink_time_step, 1)).clip(this->min_, this->max_);
		};
		std::string getName() const { return "MaxErrorTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationErrorTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

  /**
  @brief Min integration error function
  */
  template<typename TensorT, typename DeviceT>
  class MinErrorTensorOp : public IntegrationErrorTensorOp<TensorT, DeviceT>
  {
  public:
    MinErrorTensorOp() {};
    ~MinErrorTensorOp() {};
    void operator()(TensorT* source_error, TensorT *source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, TensorT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device)
    {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> sink_output_tensor(sink_output, batch_size, memory_size, sink_layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> sink_derivative_tensor(sink_derivative, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> source_input_tensor(source_input, batch_size, memory_size, 1, source_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> source_error_tensor(source_error, batch_size, memory_size, 1, source_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> weight_tensor(weight, 1, sink_layer_size, source_layer_size); // NOTE: source/sink are reversed
      // step 1: determine the minimum
      auto comp_tensor = sink_output_tensor.chip(sink_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, 1, source_layer_size })) * weight_tensor.broadcast(Eigen::array<int, 3>({ batch_size, 1, 1 }));
      auto min_tensor = source_input_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, sink_layer_size, 1 }));
      auto selection_tensor = ((comp_tensor - min_tensor).abs() > (min_tensor.constant(TensorT(0)) - min_tensor.constant(this->eps_)) &&
        (comp_tensor - min_tensor).abs() < (min_tensor.constant(TensorT(0)) + min_tensor.constant(this->eps_))).select(min_tensor.constant(TensorT(1)), min_tensor.constant(TensorT(0)));

      // step 2: select out the error to propogate
      auto error = source_error_tensor.chip(source_time_step, 1).broadcast(Eigen::array<int, 3>({ 1, sink_layer_size, 1 })) * weight_tensor.broadcast(Eigen::array<int, 3>({ batch_size, 1, 1 }));
      auto selected_error = (error * selection_tensor).sum(Eigen::array<int, 1>({ 2 })); // sum along the source layer

      sink_error_tensor.chip(sink_time_step, 1).device(device) += (selected_error * sink_derivative_tensor.chip(sink_time_step, 1)).clip(this->min_, this->max_);
    };
    std::string getName() const { return "MinErrorTensorOp"; };
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(cereal::base_class<IntegrationErrorTensorOp<TensorT, DeviceT>>(this));
    //	}
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
    void operator()(TensorT* source_error, TensorT* source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, TensorT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
      operator_(source_error, source_input, weight, sink_output, sink_error, sink_derivative, n_input_nodes, batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);
    };
    template<typename TT = TensorT, std::enable_if_t<std::is_same<TT, double>::value, int> = 0>
    void operator_(TT* source_error, TT* source_input, TT* weight, TT* sink_output, TT* sink_error, TT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> sink_derivative_tensor(sink_derivative, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 4>> source_error_tensor(source_error, batch_size, memory_size, 1, source_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> weight_tensor(weight, 1, sink_layer_size, source_layer_size); // NOTE: source/sink are reversed
      auto source_error_bcast = source_error_tensor.chip(source_time_step, 1).broadcast(Eigen::array<Eigen::Index, 3>({ 1, sink_layer_size, 1 }));
      auto weight_bcast = weight_tensor.broadcast(Eigen::array<Eigen::Index, 3>({ batch_size, 1, 1 }));
      sink_error_tensor.chip(sink_time_step, 1).device(device) += ((source_error_bcast * weight_bcast).sum(Eigen::array<int, 1>({ 2 })) * sink_error_tensor.chip(sink_time_step, 1).constant(TT(1) / (TT)n_input_nodes) * sink_derivative_tensor.chip(sink_time_step, 1)).clip(this->min_, this->max_);
    }
    template<typename TT = TensorT, std::enable_if_t<!std::is_same<TT, double>::value, int> = 0>
    void operator_(TT* source_error, TT* source_input, TT* weight, TT* sink_output, TT* sink_error, TT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> sink_derivative_tensor(sink_derivative, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> source_error_tensor(source_error, batch_size, memory_size, source_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 2>> weight_tensor(weight, sink_layer_size, source_layer_size); // NOTE: source/sink are reversed
      Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) }; // NOTE: we are taking the transpose of the weight matrix
      sink_error_tensor.chip(sink_time_step, 1).device(device) += ((source_error_tensor.chip(source_time_step, 1)).contract(weight_tensor.shuffle(Eigen::array<int, 2>({ 1, 0 })), product_dims) * sink_error_tensor.chip(sink_time_step, 1).constant(TT(1)/(TT)n_input_nodes) * sink_derivative_tensor.chip(sink_time_step, 1)).clip(this->min_, this->max_);
    }
		std::string getName() const { return "MeanErrorTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationErrorTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief VarMod integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class VarModErrorTensorOp : public IntegrationErrorTensorOp<TensorT, DeviceT>
	{
	public:
		VarModErrorTensorOp() {};
		~VarModErrorTensorOp() {};
    void operator()(TensorT* source_error, TensorT* source_input, TensorT* weight, TensorT* sink_output, TensorT* sink_error, TensorT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
      operator_(source_error, source_input, weight, sink_output, sink_error, sink_derivative, n_input_nodes, batch_size, memory_size, source_layer_size, sink_layer_size, source_time_step, sink_time_step, device);
    };
    template<typename TT = TensorT, std::enable_if_t<std::is_same<TT, double>::value, int> = 0>
    void operator_(TT* source_error, TT* source_input, TT* weight, TT* sink_output, TT* sink_error, TT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> sink_derivative_tensor(sink_derivative, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 4>> source_error_tensor(source_error, batch_size, memory_size, 1, source_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> weight_tensor(weight, 1, sink_layer_size, source_layer_size); // NOTE: source/sink are reversed
      auto source_error_bcast = source_error_tensor.chip(source_time_step, 1).broadcast(Eigen::array<Eigen::Index, 3>({ 1, sink_layer_size, 1 }));
      auto weight_bcast = weight_tensor.broadcast(Eigen::array<Eigen::Index, 3>({ batch_size, 1, 1 }));
      sink_error_tensor.chip(sink_time_step, 1).device(device) += ((source_error_bcast * weight_bcast).sum(Eigen::array<int, 1>({ 2 })) * sink_error_tensor.chip(sink_time_step, 1).constant(TT(1) / (TT)n_input_nodes).eval() * sink_error_tensor.chip(sink_time_step, 1).constant((TT)2).eval()
        * sink_derivative_tensor.chip(sink_time_step, 1)).clip(this->min_, this->max_);
    }
    template<typename TT = TensorT, std::enable_if_t<!std::is_same<TT, double>::value, int> = 0>
    void operator_(TT* source_error, TT* source_input, TT* weight, TT* sink_output, TT* sink_error, TT* sink_derivative, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, const int& source_time_step, const int& sink_time_step, DeviceT& device) {
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> sink_derivative_tensor(sink_derivative, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> source_error_tensor(source_error, batch_size, memory_size, source_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 2>> weight_tensor(weight, sink_layer_size, source_layer_size); // NOTE: source/sink are reversed
      Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) }; // NOTE: we are taking the transpose of the weight matrix
      sink_error_tensor.chip(sink_time_step, 1).device(device) += ((source_error_tensor.chip(source_time_step, 1)).contract(weight_tensor.shuffle(Eigen::array<int, 2>({ 1, 0 })), product_dims) 
      	* sink_error_tensor.chip(sink_time_step, 1).constant(TT(1) / (TT)n_input_nodes).eval() * sink_error_tensor.chip(sink_time_step, 1).constant((TT)2).eval()
      	* sink_derivative_tensor.chip(sink_time_step, 1)).clip(this->min_, this->max_);
    }
		std::string getName() const { return "VarModErrorTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationErrorTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief Var integration error function
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
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationErrorTensorOp<TensorT, DeviceT>>(this));
	//	}
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
			sink_error_tensor.chip(sink_time_step, 1).device(device) = sink_error_tensor.chip(sink_time_step, 1).constant(TensorT(0)).clip(this->min_, this->max_);
		};
		std::string getName() const { return "CountErrorTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationErrorTensorOp<TensorT, DeviceT>>(this));
	//	}
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
		TensorT eps_ = TensorT(1e-24);
    TensorT min_ = TensorT(-1e9);
    TensorT max_ = TensorT(1e9);
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(eps_);
	//	}
	};

	/**
	@brief Fully Connected Sum integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class SumWeightGradTensorOp : public IntegrationWeightGradTensorOp<TensorT, DeviceT>
	{
	public:
		void operator()(TensorT* sink_error, TensorT* source_output, TensorT* weight, TensorT* source_input, TensorT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
      operator_(sink_error, source_output, weight, source_input, weight_error, n_input_nodes, batch_size, memory_size, source_layer_size, sink_layer_size, device);
		};
    template<typename TT = TensorT, std::enable_if_t<std::is_same<TT, double>::value, int> = 0>
    void operator_(TT* sink_error, TT* source_output, TT* weight, TT* source_input, TT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
      Eigen::TensorMap<Eigen::Tensor<TT, 4>> sink_error_tensor(sink_error, batch_size, memory_size, 1, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 4>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TT, 2>> weight_error_tensor(weight_error, source_layer_size, sink_layer_size);
      auto sink_error_bcast = sink_error_tensor.broadcast(Eigen::array<Eigen::Index, 4>({ 1, 1, source_layer_size, 1 }));
      auto source_output_bcast = source_output_tensor.broadcast(Eigen::array<Eigen::Index, 4>({ 1, 1, 1, sink_layer_size }));
      auto tmp = -(source_output_bcast * sink_error_bcast).sum(Eigen::array<int, 2>({ 0, 1 }));
      weight_error_tensor.device(device) += (tmp * tmp.constant(TT(1) / (TT)batch_size)).clip(this->min_, this->max_);
    }
    template<typename TT = TensorT, std::enable_if_t<!std::is_same<TT, double>::value, int> = 0>
    void operator_(TT* sink_error, TT* source_output, TT* weight, TT* source_input, TT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 2>> weight_error_tensor(weight_error, source_layer_size, sink_layer_size);
      Eigen::array<Eigen::IndexPair<int>, 2> double_contraction_product_dims = { Eigen::IndexPair<int>(1,1), Eigen::IndexPair<int>(0,0) };
      auto tmp = -source_output_tensor.contract(sink_error_tensor, double_contraction_product_dims);
      // NOTE: Double contraction along the memory and batch (equivalent to a double sum along the products of the batch and memory dimensions)
      weight_error_tensor.device(device) += (tmp * tmp.constant(TT(1) / (TT)batch_size)).clip(this->min_, this->max_);
      // NOTE: Requires a correction by dividing by the batch size
    }
		std::string getName() const { return "SumWeightGradTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationWeightGradTensorOp<TensorT, DeviceT>>(this));
	//	}
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
      // step 0: remove small values and 0's from the weight_tensor for numerical stability
      auto weight_tensor_exp = weight_tensor.broadcast(Eigen::array<int, 4>({ batch_size, memory_size, 1, 1 }));
      auto weight_tensor_exp_clipped_neg = (weight_tensor_exp > weight_tensor_exp.constant(1 / this->min_) &&
        weight_tensor_exp < weight_tensor_exp.constant(TensorT(0))).select(
          weight_tensor_exp.constant(1 / this->min_), weight_tensor_exp);
      auto weight_tensor_exp_clipped_pos = (weight_tensor_exp <= weight_tensor_exp.constant(1 / this->max_) &&
        weight_tensor_exp > weight_tensor_exp.constant(TensorT(0))).select(
          weight_tensor_exp_clipped_neg.constant(1 / this->max_), weight_tensor_exp_clipped_neg);
      
			// step 1: compute the weight-normalized source net input expanded across batch and memory
      // NOTE for numerical stability we multiply by the weight_tensor and then divide by the square of the weight tensor plus a small number to avoid dividing by 0
			//auto input_normalized_tensor = source_input_tensor.broadcast(Eigen::array<int, 4>({ 1, 1, 1, sink_layer_size })) * weight_tensor_exp / (weight_tensor_exp*weight_tensor_exp + weight_tensor_exp.constant(this->eps_));
      auto input_normalized_tensor = (weight_tensor_exp != weight_tensor_exp.constant(TensorT(0))).select(
        source_input_tensor.broadcast(Eigen::array<int, 4>({ 1, 1, 1, sink_layer_size })) / weight_tensor_exp_clipped_pos, weight_tensor_exp.constant(TensorT(0)));
			
      // step 2: scale to the sink error
			auto scaled_error = -sink_error_tensor.broadcast(Eigen::array<int, 4>({ 1, 1, source_layer_size, 1 })) * input_normalized_tensor;
			
      // step 3: sum along the memory and average along the batch dimensions
			weight_error_tensor.device(device) += (scaled_error.sum(Eigen::array<int, 2>({ 0, 1 })) * weight_error_tensor.constant(TensorT(1) / (TensorT)batch_size)).clip(this->min_, this->max_);
		};
		std::string getName() const { return "ProdWeightGradTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationWeightGradTensorOp<TensorT, DeviceT>>(this));
	//	}
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
      operator_(sink_error, source_output, weight, source_input, weight_error, n_input_nodes, batch_size, memory_size, source_layer_size, sink_layer_size, device);
    };
    template<typename TT = TensorT, std::enable_if_t<std::is_same<TT, double>::value, int> = 0>
    void operator_(TT* sink_error, TT* source_output, TT* weight, TT* source_input, TT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
      Eigen::TensorMap<Eigen::Tensor<TT, 4>> sink_error_tensor(sink_error, batch_size, memory_size, 1, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 4>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TT, 2>> weight_error_tensor(weight_error, source_layer_size, sink_layer_size);
      auto sink_error_bcast = sink_error_tensor.broadcast(Eigen::array<Eigen::Index, 4>({ 1, 1, source_layer_size, 1 }));
      auto source_output_bcast = source_output_tensor.broadcast(Eigen::array<Eigen::Index, 4>({ 1, 1, 1, sink_layer_size }));
      auto tmp = -(source_output_bcast * sink_error_bcast).sum(Eigen::array<int, 2>({ 0, 1 }));
      weight_error_tensor.device(device) += (tmp * tmp.constant(TT(1) / (TT)batch_size)).clip(this->min_, this->max_);
    }
    template<typename TT = TensorT, std::enable_if_t<!std::is_same<TT, double>::value, int> = 0>
    void operator_(TT* sink_error, TT* source_output, TT* weight, TT* source_input, TT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 2>> weight_error_tensor(weight_error, source_layer_size, sink_layer_size);
      Eigen::array<Eigen::IndexPair<int>, 2> double_contraction_product_dims = { Eigen::IndexPair<int>(1,1), Eigen::IndexPair<int>(0,0) };
      auto tmp = -source_output_tensor.contract(sink_error_tensor, double_contraction_product_dims);
      // NOTE: Double contraction along the memory and batch (equivalent to a double sum along the products of the batch and memory dimensions)
      weight_error_tensor.device(device) += (tmp * tmp.constant(TT(1) / (TT)batch_size)).clip(this->min_, this->max_);
      // NOTE: Requires a correction by dividing by the batch size
    }
		std::string getName() const { return "MaxWeightGradTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationWeightGradTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

  /**
  @brief Min integration error function
  */
  template<typename TensorT, typename DeviceT>
  class MinWeightGradTensorOp : public IntegrationWeightGradTensorOp<TensorT, DeviceT>
  {
  public:
    MinWeightGradTensorOp() {};
    ~MinWeightGradTensorOp() {};
    void operator()(TensorT* sink_error, TensorT* source_output, TensorT* weight, TensorT* source_input, TensorT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
      operator_(sink_error, source_output, weight, source_input, weight_error, n_input_nodes, batch_size, memory_size, source_layer_size, sink_layer_size, device);
    };
    template<typename TT = TensorT, std::enable_if_t<std::is_same<TT, double>::value, int> = 0>
    void operator_(TT* sink_error, TT* source_output, TT* weight, TT* source_input, TT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
      Eigen::TensorMap<Eigen::Tensor<TT, 4>> sink_error_tensor(sink_error, batch_size, memory_size, 1, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 4>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TT, 2>> weight_error_tensor(weight_error, source_layer_size, sink_layer_size);
      auto sink_error_bcast = sink_error_tensor.broadcast(Eigen::array<Eigen::Index, 4>({ 1, 1, source_layer_size, 1 }));
      auto source_output_bcast = source_output_tensor.broadcast(Eigen::array<Eigen::Index, 4>({ 1, 1, 1, sink_layer_size }));
      auto tmp = -(source_output_bcast * sink_error_bcast).sum(Eigen::array<int, 2>({ 0, 1 }));
      weight_error_tensor.device(device) += (tmp * tmp.constant(TT(1) / (TT)batch_size)).clip(this->min_, this->max_);
    }
    template<typename TT = TensorT, std::enable_if_t<!std::is_same<TT, double>::value, int> = 0>
    void operator_(TT* sink_error, TT* source_output, TT* weight, TT* source_input, TT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 2>> weight_error_tensor(weight_error, source_layer_size, sink_layer_size);
      Eigen::array<Eigen::IndexPair<int>, 2> double_contraction_product_dims = { Eigen::IndexPair<int>(1,1), Eigen::IndexPair<int>(0,0) };
      auto tmp = -source_output_tensor.contract(sink_error_tensor, double_contraction_product_dims);
      // NOTE: Double contraction along the memory and batch (equivalent to a double sum along the products of the batch and memory dimensions)
      weight_error_tensor.device(device) += (tmp * tmp.constant(TT(1) / (TT)batch_size)).clip(this->min_, this->max_);
      // NOTE: Requires a correction by dividing by the batch size
    }
    std::string getName() const { return "MinWeightGradTensorOp"; };
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(cereal::base_class<IntegrationWeightGradTensorOp<TensorT, DeviceT>>(this));
    //	}
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
			weight_error_tensor.device(device) = weight_error_tensor.constant(TensorT(0));
		};
		std::string getName() const { return "CountWeightGradTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationWeightGradTensorOp<TensorT, DeviceT>>(this));
	//	}
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
      operator_(sink_error, source_output, weight, source_input, weight_error, n_input_nodes, batch_size, memory_size, source_layer_size, sink_layer_size, device);
    };
    template<typename TT = TensorT, std::enable_if_t<std::is_same<TT, double>::value, int> = 0>
    void operator_(TT* sink_error, TT* source_output, TT* weight, TT* source_input, TT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
      Eigen::TensorMap<Eigen::Tensor<TT, 4>> sink_error_tensor(sink_error, batch_size, memory_size, 1, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 4>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TT, 2>> weight_error_tensor(weight_error, source_layer_size, sink_layer_size);
      auto sink_error_bcast = sink_error_tensor.broadcast(Eigen::array<Eigen::Index, 4>({ 1, 1, source_layer_size, 1 }));
      auto source_output_bcast = source_output_tensor.broadcast(Eigen::array<Eigen::Index, 4>({ 1, 1, 1, sink_layer_size }));
      auto tmp = -(source_output_bcast * sink_error_bcast).sum(Eigen::array<int, 2>({ 0, 1 }));
      weight_error_tensor.device(device) += (tmp * weight_error_tensor.constant(TT(1) / (TT)batch_size).eval() * weight_error_tensor.constant(TT(1) / (TT)n_input_nodes)).clip(this->min_, this->max_);;
    }
    template<typename TT = TensorT, std::enable_if_t<!std::is_same<TT, double>::value, int> = 0>
    void operator_(TT* sink_error, TT* source_output, TT* weight, TT* source_input, TT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 2>> weight_error_tensor(weight_error, source_layer_size, sink_layer_size);
      Eigen::array<Eigen::IndexPair<int>, 2> double_contraction_product_dims = { Eigen::IndexPair<int>(1,1), Eigen::IndexPair<int>(0,0) };
      auto tmp = -source_output_tensor.contract(sink_error_tensor, double_contraction_product_dims);
      // NOTE: Double contraction along the memory and batch (equivalent to a double sum along the products of the batch and memory dimensions)
      weight_error_tensor.device(device) += (tmp * weight_error_tensor.constant(TT(1) / (TT)batch_size).eval() * weight_error_tensor.constant(TT(1) / (TT)n_input_nodes)).clip(this->min_, this->max_);;
      // NOTE: Requires a correction by dividing by the batch size
    }
		std::string getName() const { return "MeanWeightGradTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationWeightGradTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief VarMod integration error function
	*/
	template<typename TensorT, typename DeviceT>
	class VarModWeightGradTensorOp : public IntegrationWeightGradTensorOp<TensorT, DeviceT>
	{
	public:
		VarModWeightGradTensorOp() {};
		~VarModWeightGradTensorOp() {};
    void operator()(TensorT* sink_error, TensorT* source_output, TensorT* weight, TensorT* source_input, TensorT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
      operator_(sink_error, source_output, weight, source_input, weight_error, n_input_nodes, batch_size, memory_size, source_layer_size, sink_layer_size, device);
    };
    template<typename TT = TensorT, std::enable_if_t<std::is_same<TT, double>::value, int> = 0>
    void operator_(TT* sink_error, TT* source_output, TT* weight, TT* source_input, TT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
      Eigen::TensorMap<Eigen::Tensor<TT, 4>> sink_error_tensor(sink_error, batch_size, memory_size, 1, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 4>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size, 1);
      Eigen::TensorMap<Eigen::Tensor<TT, 2>> weight_error_tensor(weight_error, source_layer_size, sink_layer_size);
      auto sink_error_bcast = sink_error_tensor.broadcast(Eigen::array<Eigen::Index, 4>({ 1, 1, source_layer_size, 1 }));
      auto source_output_bcast = source_output_tensor.broadcast(Eigen::array<Eigen::Index, 4>({ 1, 1, 1, sink_layer_size }));
      auto tmp = -(source_output_bcast * sink_error_bcast).sum(Eigen::array<int, 2>({ 0, 1 }));
      weight_error_tensor.device(device) += (tmp * weight_error_tensor.constant(TT(1) / (TT)batch_size)
        * weight_error_tensor.constant(TT(1) / (TT)n_input_nodes).eval() * weight_error_tensor.constant((TT)2)).clip(this->min_, this->max_);;
    }
    template<typename TT = TensorT, std::enable_if_t<!std::is_same<TT, double>::value, int> = 0>
    void operator_(TT* sink_error, TT* source_output, TT* weight, TT* source_input, TT* weight_error, const int& n_input_nodes, const int& batch_size, const int& memory_size, const int& source_layer_size, const int& sink_layer_size, DeviceT& device) {
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> sink_error_tensor(sink_error, batch_size, memory_size, sink_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 3>> source_output_tensor(source_output, batch_size, memory_size, source_layer_size);
      Eigen::TensorMap<Eigen::Tensor<TT, 2>> weight_error_tensor(weight_error, source_layer_size, sink_layer_size);
      Eigen::array<Eigen::IndexPair<int>, 2> double_contraction_product_dims = { Eigen::IndexPair<int>(1,1), Eigen::IndexPair<int>(0,0) };
      auto tmp = -source_output_tensor.contract(sink_error_tensor, double_contraction_product_dims);
      // NOTE: Double contraction along the memory and batch (equivalent to a double sum along the products of the batch and memory dimensions)
      weight_error_tensor.device(device) += (tmp * weight_error_tensor.constant(TT(1) / (TT)batch_size)
        * weight_error_tensor.constant(TT(1) / (TT)n_input_nodes).eval() * weight_error_tensor.constant((TT)2)).clip(this->min_, this->max_);;
      // NOTE: Requires a correction by dividing by the batch size
    }
		std::string getName() const { return "VarModWeightGradTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationWeightGradTensorOp<TensorT, DeviceT>>(this));
	//	}
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
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<IntegrationWeightGradTensorOp<TensorT, DeviceT>>(this));
	//	}
	};
}
//CEREAL_REGISTER_TYPE(SmartPeak::SumTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ProdTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::MaxTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::MeanTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::VarModTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::CountTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SumErrorTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ProdErrorTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::MaxErrorTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::MeanErrorTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::VarModErrorTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::CountErrorTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SumWeightGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ProdWeightGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::MaxWeightGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::CountWeightGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::MeanWeightGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::VarModWeightGradTensorOp<float, Eigen::DefaultDevice>);
//
//#if COMPILE_WITH_CUDA
//CEREAL_REGISTER_TYPE(SmartPeak::SumTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ProdTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::MaxTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::MeanTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::VarModTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::CountTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SumErrorTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ProdErrorTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::MaxErrorTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::MeanErrorTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::VarModErrorTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::CountErrorTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SumWeightGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ProdWeightGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::MaxWeightGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::CountWeightGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::MeanWeightGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::VarModWeightGradTensorOp<float, Eigen::GpuDevice>);
//#endif

#endif //SMARTPEAK_TENSORINTEGRATIONFUNCTION_H