/**TODO:  Add copyright*/

#ifndef SMARTPEAK_ACTIVATIONTENSORFUNCTION_H
#define SMARTPEAK_ACTIVATIONTENSORFUNCTION_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <SmartPeak/ml/ActivationFunction.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/MatrixFunctions>

//#include <cereal/access.hpp>  // serialiation of private members
//#undef min // clashes with std::limit on windows in polymorphic.hpp
//#undef max // clashes with std::limit on windows in polymorphic.hpp
//#include <cereal/types/polymorphic.hpp>

namespace SmartPeak
{
  /**
    @brief Base class for all activation function wrappers.
  */
  template<typename TensorT, typename DeviceT>
  class ActivationTensorOp
  {
  public: 
		ActivationTensorOp() = default;
    ActivationTensorOp(const TensorT& eps, const TensorT& min, const TensorT& max) : eps_(eps), min_(min), max_(max) {};
		virtual ~ActivationTensorOp() = default;
		virtual std::string getName() const = 0;
		virtual void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const = 0;
    void setEps(const TensorT& eps) { eps_ = eps; }
    void setMin(const TensorT& min) { min_ = min; }
    void setMax(const TensorT& max) { max_ = max; }
    TensorT getEps() const { return eps_; }
    TensorT getMin() const { return min_; }
    TensorT getMax() const { return max_; }
  protected:
    TensorT eps_ = TensorT(1e-24); ///< threshold to clip between min and max
    TensorT min_ = TensorT(-1e9);
    TensorT max_ = TensorT(1e9);
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {}
  };

  /**
    @brief Rectified Linear Unit (ReLU) activation function

    References:
    R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). 
      Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. 
      Nature. 405. pp. 947–951.
  */
  template<typename TensorT, typename DeviceT>
  class ReLUTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
	public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			auto result = (x.chip(time_step, 1) >= x.chip(time_step, 1).constant(TensorT(0))).select(x.chip(time_step, 1), x.chip(time_step, 1).constant(TensorT(0)));
			out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
      //std::cout << "[ReLUTensorOp] Time step " << time_step << " : " << out.chip(time_step, 1) << std::endl;  // DEBUGGING...
		};
    std::string getName() const{return "ReLUTensorOp";};
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
  };

  /**
    @brief Rectified Linear Unit (ReLU) gradient

    References:
    R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). 
      Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit. 
      Nature. 405. pp. 947–951.
  */
  template<typename TensorT, typename DeviceT>
  class ReLUGradTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			auto result = (x.chip(time_step, 1) >= x.chip(time_step, 1).constant(TensorT(0))).select(x.chip(time_step, 1).constant(TensorT(1)), x.chip(time_step, 1).constant(TensorT(0)));
			out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
		};
    std::string getName() const{return "ReLUGradTensorOp";};
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
  };

  /**
    @brief Exponential Linear Unit (ELU) activation function

    References:
    Clevert, Djork-Arné; Unterthiner, Thomas; Hochreiter, Sepp (2015). 
      "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)".
      arXiv:1511.07289
  */
  template<typename TensorT, typename DeviceT>
  class ELUTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
  public:
    ELUTensorOp() = default;
    ~ELUTensorOp() = default;
    ELUTensorOp(const TensorT& eps, const TensorT& min, const TensorT& max, const TensorT& alpha) : ActivationTensorOp(eps, min, max), alpha_(alpha) {};
    ELUTensorOp(const TensorT& alpha): alpha_(alpha){}; 
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			auto result = (x.chip(time_step, 1) > x.chip(time_step, 1).constant(TensorT(0))).select(
				x.chip(time_step, 1), 
				x.chip(time_step, 1).constant(alpha_) * (x.chip(time_step, 1).exp() - x.chip(time_step, 1).constant(TensorT(1))));
			out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
		};
    void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
    TensorT getAlpha() const { return alpha_; };
    std::string getName() const{return "ELUTensorOp";};
	private:
		//friend class cereal::access;
		//template<class Archive>
		//void serialize(Archive& archive) {
		//	archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this), alpha_);
		//}
    TensorT alpha_ = 1;
  };

  /**
    @brief Exponential Linear Unit (ELU) gradient

    References:
    Clevert, Djork-Arné; Unterthiner, Thomas; Hochreiter, Sepp (2015). 
      "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)".
      arXiv:1511.07289
  */
  template<typename TensorT, typename DeviceT>
  class ELUGradTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
  public:
    ELUGradTensorOp() = default;
    ~ELUGradTensorOp() = default;
    ELUGradTensorOp(const TensorT& eps, const TensorT& min, const TensorT& max, const TensorT& alpha) : ActivationTensorOp(eps, min, max), alpha_(alpha) {};
    ELUGradTensorOp(const TensorT& alpha): alpha_(alpha){}; 
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			auto result = (x.chip(time_step, 1) > x.chip(time_step, 1).constant(TensorT(0))).select(
				x.chip(time_step, 1).constant(TensorT(1)),
				(x.chip(time_step, 1) > x.chip(time_step, 1).constant(TensorT(0))).select(x.chip(time_step, 1), x.chip(time_step, 1).constant(alpha_) * (x.chip(time_step, 1).exp() - x.chip(time_step, 1).constant(TensorT(1)))) + x.chip(time_step, 1).constant(alpha_));
			out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
		};
    void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
    TensorT getAlpha() const { return alpha_; };
    std::string getName() const{return "ELUGradTensorOp";};
	private:
		//friend class cereal::access;
		//template<class Archive>
		//void serialize(Archive& archive) {
		//	archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this), alpha_);
		//}
    TensorT alpha_ = 1;
  };

  /**
    @brief Sigmoid activation function
  */
  template<typename TensorT, typename DeviceT>
  class SigmoidTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			auto result = x.chip(time_step, 1).sigmoid();
			out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
		};
    std::string getName() const{return "SigmoidTensorOp";};
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
  };

  /**
    @brief Sigmoid gradient
  */
  template<typename TensorT, typename DeviceT>
  class SigmoidGradTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			auto result = x.chip(time_step, 1).sigmoid() * (x.chip(time_step, 1).constant(TensorT(1)) - x.chip(time_step, 1).sigmoid());
			out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
		};
    std::string getName() const{return "SigmoidGradTensorOp";};
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
  };
  
  /**
    @brief Hyperbolic Tangent activation function
  */
  template<typename TensorT, typename DeviceT>
  class TanHTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			auto result = x.chip(time_step, 1).tanh();
			out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
		};
    std::string getName() const{return "TanHTensorOp";};
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
  };

  /**
    @brief Hyperbolic Tangent gradient
  */
  template<typename TensorT, typename DeviceT>
  class TanHGradTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			auto result = x.chip(time_step, 1).constant(TensorT(1)) - (x.chip(time_step, 1).tanh()).pow((TensorT)2);
			out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
		};
    std::string getName() const{return "TanHGradTensorOp";};
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
  };
  
  /**
    @brief Rectified Hyperbolic Tangent activation function
  */
  template<typename TensorT, typename DeviceT>
  class ReTanHTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(ReTanHOp<TensorT>());
			//out.chip(time_step, 1).device(device) = [TODO]
		};
    std::string getName() const{return "ReTanHTensorOp";};
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
  };

  /**
    @brief Rectified Hyperbolic Tangent gradient
  */
  template<typename TensorT, typename DeviceT>
  class ReTanHGradTensorOp: public ActivationTensorOp<TensorT, DeviceT>
  {
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//out.chip(time_step, 1).device(device) = x.chip(time_step, 1).unaryExpr(ReTanHGradOp<TensorT>());
			//out.chip(time_step, 1).device(device) = [TODO]
		};
    std::string getName() const{return "ReTanHGradTensorOp";};
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
  };

	/**
	@brief Linear activation function
	*/
	template<typename TensorT, typename DeviceT>
	class LinearTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			auto result = x.chip(time_step, 1); 
			out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
		};
		std::string getName() const { return "LinearTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief Linear gradient
	*/
	template<typename TensorT, typename DeviceT>
	class LinearGradTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			out.chip(time_step, 1).device(device) = x.chip(time_step, 1).constant(TensorT(1));
		};
		std::string getName() const { return "LinearGradTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief Inverse activation function
	*/
	template<typename TensorT, typename DeviceT>
	class InverseTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
      // Temporary memory for computation
      TensorT* tmp_data;
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        tmp_data = new TensorT[batch_size * layer_size];
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        size_t bytes = batch_size * layer_size * sizeof(TensorT);
        assert(cudaMalloc((void**)(&tmp_data), bytes) == cudaSuccess);
      }
#endif
      // Cap small values by selection
      auto x_clipped_neg = (x.chip(time_step, 1) > x.chip(time_step, 1).constant(1 / this->getMin()) &&
        x.chip(time_step, 1) < x.chip(time_step, 1).constant(TensorT(0))).select(
          x.chip(time_step, 1).constant(1 / this->getMin()), x.chip(time_step, 1));
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x_clipped_pos(tmp_data, batch_size, layer_size);
      x_clipped_pos.device(device) = (x_clipped_neg <= x_clipped_neg.constant(1 / this->getMax()) &&
        x_clipped_neg > x_clipped_neg.constant(TensorT(0))).select(
          x_clipped_neg.constant(1 / this->getMax()), x_clipped_neg);
      // Remove 0 by selection
      auto result = (x_clipped_pos != x_clipped_pos.constant(TensorT(0))).select(
        x_clipped_pos.constant(TensorT(1)) / x_clipped_pos, x_clipped_pos.constant(TensorT(0)));
      out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
      // Deallocate temporary memory
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        delete[] tmp_data;
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        assert(cudaFree(tmp_data) == cudaSuccess);
      }
#endif
		};
		std::string getName() const { return "InverseTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief Inverse gradient
	*/
	template<typename TensorT, typename DeviceT>
	class InverseGradTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
      // Temporary memory for computation
      TensorT* tmp_data;
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        tmp_data = new TensorT[batch_size * layer_size];
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        size_t bytes = batch_size * layer_size * sizeof(TensorT);
        assert(cudaMalloc((void**)(&tmp_data), bytes) == cudaSuccess);
      }
#endif
      // Cap small values by selection
      auto x_clipped_neg = (x.chip(time_step, 1) > x.chip(time_step, 1).constant(1 / this->getMin()) &&
        x.chip(time_step, 1) < x.chip(time_step, 1).constant(TensorT(0))).select(
          x.chip(time_step, 1).constant(1 / this->getMin()), x.chip(time_step, 1));
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x_clipped_pos(tmp_data, batch_size, layer_size);
      x_clipped_pos.device(device) = (x_clipped_neg <= x_clipped_neg.constant(1 / this->getMax()) &&
        x_clipped_neg > x_clipped_neg.constant(TensorT(0))).select(
          x_clipped_neg.constant(1 / this->getMax()), x_clipped_neg);
      // Remove 0 by selection
      auto result = (x_clipped_pos != x_clipped_pos.constant(TensorT(0))).select(
        x_clipped_pos.constant(TensorT(-1)) / x_clipped_pos.pow(2), x_clipped_pos.constant(TensorT(0)));
			out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
      // Deallocate temporary memory
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        delete[] tmp_data;
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        assert(cudaFree(tmp_data) == cudaSuccess);
      }
#endif
		};
		std::string getName() const { return "InverseGradTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief Exponential activation function
	*/
	template<typename TensorT, typename DeviceT>
	class ExponentialTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
      TensorT maxT = log(this->getMax());
			auto result = x.chip(time_step, 1).clip(this->getMin(), maxT).exp();
			out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
		};
		std::string getName() const { return "ExponentialTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief Exponential gradient
	*/
	template<typename TensorT, typename DeviceT>
	class ExponentialGradTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
      TensorT maxT = log(this->getMax());
      auto result = x.chip(time_step, 1).clip(this->getMin(), maxT).exp();
			out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
		};
		std::string getName() const { return "ExponentialGradTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief Log activation function
	*/
	template<typename TensorT, typename DeviceT>
	class LogTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			auto result = x.chip(time_step, 1).clip(this->getEps(), this->getMax()).log();
			out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
		};
		std::string getName() const { return "LogTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief Log gradient
	*/
	template<typename TensorT, typename DeviceT>
	class LogGradTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
      // Temporary memory for computation
      TensorT* tmp_data;
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        tmp_data = new TensorT[batch_size* layer_size];
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        size_t bytes = batch_size * layer_size * sizeof(TensorT);
        assert(cudaMalloc((void**)(&tmp_data), bytes) == cudaSuccess);
      }
#endif
      // Cap small values by selection
      auto x_clipped_neg = (x.chip(time_step, 1) > x.chip(time_step, 1).constant(1/this->getMin()) &&
        x.chip(time_step, 1) < x.chip(time_step, 1).constant(TensorT(0))).select(
        x.chip(time_step, 1).constant(1/this->getMin()), x.chip(time_step, 1));
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> x_clipped_pos(tmp_data, batch_size, layer_size);
      x_clipped_pos.device(device) = (x_clipped_neg <= x_clipped_neg.constant(1/this->getMax()) &&
        x_clipped_neg > x_clipped_neg.constant(TensorT(0))).select(
        x_clipped_neg.constant(1/this->getMax()), x_clipped_neg);
      // Remove 0 by selection
      auto result = (x_clipped_pos != x_clipped_pos.constant(TensorT(0))).select(
        x_clipped_pos.constant(TensorT(1)) / x_clipped_pos, x_clipped_pos.constant(TensorT(0)));
			out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
      // Deallocate temporary memory
      if (typeid(device).name() == typeid(Eigen::DefaultDevice).name()) {
        delete[] tmp_data;
      }
#if COMPILE_WITH_CUDA
      else if (typeid(device).name() == typeid(Eigen::GpuDevice).name()) {
        assert(cudaFree(tmp_data) == cudaSuccess);
      }
#endif
		};
		std::string getName() const { return "LogGradTensorOp"; };
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
	//	}
	};

	/**
	@brief Pow activation function
	*/
	template<typename TensorT, typename DeviceT>
	class PowTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
  public:
    PowTensorOp() = default;
    ~PowTensorOp() = default;
    PowTensorOp(const TensorT& eps, const TensorT& min, const TensorT& max, const TensorT& base) : ActivationTensorOp(eps, min, max), base_(base) {};
		PowTensorOp(const TensorT& base): base_(base){};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);      
      TensorT maxT = (base_ >= TensorT(1))? pow(this->getMax(), 1 / base_): this->getMax();
      TensorT minT = ((base_ < TensorT(1) && base_ > TensorT(0)) || (base_ > TensorT(-1) && base_ < TensorT(0))) ? TensorT(0): this->getMin();
			auto result = x.chip(time_step, 1).clip(minT, maxT).pow(base_);
      // NOTE there is still the case where base_ < 0 and x == 0 to deal with
			out.chip(time_step, 1).device(device) = (result == result).select(result.clip(this->getMin(), this->getMax()), result.constant(TensorT(0)));
		};
		std::string getName() const { return "PowTensorOp"; };
	private:
		//friend class cereal::access;
		//template<class Archive>
		//void serialize(Archive& archive) {
		//	archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this), base_);
		//}
		TensorT base_;
	};

	/**
	@brief Pow gradient
	*/
	template<typename TensorT, typename DeviceT>
	class PowGradTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
  public:
    PowGradTensorOp() = default;
    ~PowGradTensorOp() = default;
    PowGradTensorOp(const TensorT& eps, const TensorT& min, const TensorT& max, const TensorT& base) : ActivationTensorOp(eps, min, max), base_(base) {};
		PowGradTensorOp(const TensorT& base) : base_(base) {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
      TensorT maxT = (base_ >= TensorT(2)) ? pow(this->getMax(), 1 / (base_ - TensorT(1))) : this->getMax();
      TensorT minT = ((base_ < TensorT(2) && base_ > TensorT(1)) || (base_ > TensorT(0) && base_ < TensorT(1))) ? TensorT(0) : this->getMin();
			auto result = x.chip(time_step, 1).constant(base_) * x.chip(time_step, 1).clip(minT, maxT).pow(base_ - TensorT(1));
      // NOTE there is still the case where base_ < 0 and x == 0 to deal with
      out.chip(time_step, 1).device(device) = (result == result).select(result.clip(this->getMin(), this->getMax()), result.constant(TensorT(0)));
		};
		std::string getName() const { return "PowGradTensorOp"; };
	private:
		//friend class cereal::access;
		//template<class Archive>
		//void serialize(Archive& archive) {
		//	archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this), base_);
		//}
		TensorT base_;
	};

	/**
		@brief LeakyReLU activation function

		default alpha = 1e-2
	*/
	template<typename TensorT, typename DeviceT>
	class LeakyReLUTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
  public:
    LeakyReLUTensorOp() = default;
    ~LeakyReLUTensorOp() = default;
    LeakyReLUTensorOp(const TensorT& eps, const TensorT& min, const TensorT& max, const TensorT& alpha) : ActivationTensorOp(eps, min, max), alpha_(alpha) {};
		LeakyReLUTensorOp(const TensorT& alpha) : alpha_(alpha) {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			auto result = (x.chip(time_step, 1) >= x.chip(time_step, 1).constant(TensorT(0))).select(
				x.chip(time_step, 1), x.chip(time_step, 1) * x.chip(time_step, 1).constant(alpha_));
			out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
		};
		void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
		TensorT getAlpha() const { return alpha_; };
		std::string getName() const { return "LeakyReLUTensorOp"; };
	private:
		//friend class cereal::access;
		//template<class Archive>
		//void serialize(Archive& archive) {
		//	archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this), alpha_);
		//}
		TensorT alpha_ = 1e-2;
	};

	/**
		@brief LeakyReLU gradient
	*/
	template<typename TensorT, typename DeviceT>
	class LeakyReLUGradTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
  public:
    LeakyReLUGradTensorOp() = default;
    ~LeakyReLUGradTensorOp() = default;
    LeakyReLUGradTensorOp(const TensorT& eps, const TensorT& min, const TensorT& max, const TensorT& alpha) : ActivationTensorOp(eps, min, max), alpha_(alpha) {};
		LeakyReLUGradTensorOp(const TensorT& alpha) : alpha_(alpha) {};
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			auto result = (x.chip(time_step, 1) >= x.chip(time_step, 1).constant(TensorT(0))).select(
				x.chip(time_step, 1).constant(TensorT(1)), x.chip(time_step, 1).constant(alpha_));
			out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
		};
		void setAlpha(const TensorT& alpha) { alpha_ = alpha; };
		TensorT getAlpha() const { return alpha_; };
		std::string getName() const { return "LeakyReLUGradTensorOp"; };
	private:
		//friend class cereal::access;
		//template<class Archive>
		//void serialize(Archive& archive) {
		//	archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this), alpha_);
		//}
		TensorT alpha_ = (TensorT)1e-2;
	};

	/**
	@brief Sin activation function
	*/
	template<typename TensorT, typename DeviceT>
	class SinTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//auto result = x.chip(time_step, 1).sin();
			//out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
		};
		std::string getName() const { return "SinTensorOp"; };
		//private:
		//	friend class cereal::access;
		//	template<class Archive>
		//	void serialize(Archive& archive) {
		//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
		//	}
	};

	/**
	@brief Sin gradient
	*/
	template<typename TensorT, typename DeviceT>
	class SinGradTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//auto result = x.chip(time_step, 1).cos();
			//out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
		};
		std::string getName() const { return "SinGradTensorOp"; };
		//private:
		//	friend class cereal::access;
		//	template<class Archive>
		//	void serialize(Archive& archive) {
		//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
		//	}
	};

	/**
	@brief Cos activation function
	*/
	template<typename TensorT, typename DeviceT>
	class CosTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//auto result = x.chip(time_step, 1).cos();
			//out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
		};
		std::string getName() const { return "CosTensorOp"; };
		//private:
		//	friend class cereal::access;
		//	template<class Archive>
		//	void serialize(Archive& archive) {
		//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
		//	}
	};

	/**
	@brief Cos gradient
	*/
	template<typename TensorT, typename DeviceT>
	class CosGradTensorOp : public ActivationTensorOp<TensorT, DeviceT>
	{
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
		void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
			//auto result = -x.chip(time_step, 1).sin();
			//out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax());
		};
		std::string getName() const { return "CosGradTensorOp"; };
		//private:
		//	friend class cereal::access;
		//	template<class Archive>
		//	void serialize(Archive& archive) {
		//		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
		//	}
	};

  /**
  @brief BatchNorm activation function
  */
  template<typename TensorT, typename DeviceT>
  class BatchNormTensorOp : public ActivationTensorOp<TensorT, DeviceT>
  {
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
      Eigen::TensorMap<Eigen::Tensor<TensorT, 4>> x(x_I, batch_size, 1, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
      auto mean = x.chip(time_step, 2).mean(Eigen::array<Eigen::Index, 1>({ 0 })).broadcast(Eigen::array<Eigen::Index, 2>({batch_size, 1}));  // 2 dims
      auto var = (x.chip(time_step, 2).chip(0, 1) - mean).pow(TensorT(2)) / mean.constant(TensorT(batch_size));
      auto result = (var <= var.constant(TensorT(0))).select(var.constant(TensorT(0)), (x.chip(time_step, 2).chip(0, 1) - mean) / var.sqrt());
      out.chip(time_step, 1).device(device) = result.clip(this->getMin(), this->getMax()).eval();
    };
    std::string getName() const { return "BatchNormTensorOp"; };
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
    //	}
  };

  /**
  @brief BatchNorm gradient

  d/dx((xi-mu)/var.sqrt())
  = ddx(xi-mu)/var.sqrt() + (xi-mu)*ddx(var.pow(-1/2))
  = ddx(xi-mu)/var.sqrt() + (xi-mu)*(-0.5*var.pow(-3/2))*ddx(var)
  = ddx(xi-mu)/var.sqrt() + (xi-mu)*(-0.5*var.pow(-3/2))*ddx(SUM(xi-mu)/m)
  = ddx(xi-mu)/var.sqrt() - (xi-mu)*(1/2m)*var.pow(-3/2)*ddx(SUM(xi-mu))
  = 0 - 0

  */
  template<typename TensorT, typename DeviceT>
  class BatchNormGradTensorOp : public ActivationTensorOp<TensorT, DeviceT>
  {
  public:
    using ActivationTensorOp<TensorT, DeviceT>::ActivationTensorOp;
    void operator()(TensorT* x_I, TensorT* x_O, const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
      //Eigen::TensorMap<Eigen::Tensor<TensorT, 5>> x(x_I, 1, batch_size, 1, memory_size, layer_size);
      //Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
      //auto mean = x.chip(time_step, 3).mean(Eigen::array<Eigen::Index, 1>({ 1 })).broadcast(Eigen::array<Eigen::Index, 3>({ 1, batch_size, 1 }));
      //auto var = ((x.chip(time_step, 3).chip(0, 2) - mean).pow(TensorT(2)) / mean.constant(TensorT(batch_size)));
      //auto x_mu = (x.chip(time_step, 3).chip(0, 2) - mean).sum(Eigen::array<Eigen::Index, 1>({ 1 })).broadcast(Eigen::array<Eigen::Index, 2>({ batch_size, 1 }));
      ////std::cout << "x_mu\n" << x_mu << std::endl;
      //auto result = x_mu * var.chip(0, 0).pow(-1 / 2) -
      //  x_mu.constant(1 / TensorT(2*batch_size)) * (x.chip(time_step, 4).chip(0, 2).chip(0, 0) - mean.chip(0, 0)) * var.chip(0, 0).pow(-3/2) * x_mu;
      ////std::cout << "result\n" << result << std::endl;

      //Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x(x_I, batch_size, memory_size, layer_size);
      //Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
      //auto x_chip = x.chip(time_step, 1);
      //auto result = -x_chip.constant(TensorT(batch_size)).pow(2) / (x_chip.constant(TensorT(batch_size)) - x_chip.constant(TensorT(1))) / x_chip.pow(2);
      ////std::cout << "x.chip(time_step, 4).chip(0, 3).chip(0, 1).chip(0, 0)\n" << x.chip(time_step, 4).chip(0, 3).chip(0, 1).chip(0, 0) << std::endl;

      //Eigen::TensorMap<Eigen::Tensor<TensorT, 6>> x(x_I, 1, 1, batch_size, 1, memory_size, layer_size);
      //Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> out(x_O, batch_size, memory_size, layer_size);
      //auto mean = x.chip(time_step, 4).mean(Eigen::array<Eigen::Index, 1>({ 2 })).broadcast(Eigen::array<Eigen::Index, 4>({1, 1, batch_size, 1 })); 
      //auto var = ((x.chip(time_step, 4).chip(0, 3) - mean).pow(TensorT(2)) / mean.constant(TensorT(batch_size))).chip(0, 1).chip(0, 0);
      //auto x_mu = (x.chip(time_step, 4).chip(0, 3) - mean).sum(Eigen::array<Eigen::Index, 2>({ 2, 3 })).broadcast(Eigen::array<Eigen::Index, 2>({ batch_size, layer_size }));
      //std::cout << "x_mu\n" << x_mu << std::endl;
      //auto dvar = x_mu * var.constant(TensorT(-0.5)) * var.pow(TensorT(-3 / 2));
      //std::cout << "dvar\n" << dvar << std::endl;
      //auto dmu = - 1 / var.sqrt() +  dvar * dvar.constant(TensorT(-2) / TensorT(batch_size)) * x_mu;
      //std::cout << "dmu\n" << dmu << std::endl;
      //auto result = 1 / var.sqrt() + dmu / dvar.constant(TensorT(batch_size)) +
      //  dvar * dvar.constant(TensorT(2) / TensorT(batch_size)) * (x.chip(time_step, 4).chip(0, 3).chip(0, 1).chip(0, 0) - mean.chip(0, 1).chip(0, 0));

      out.chip(time_step, 1).device(device) = (result == result).select(result.clip(this->getMin(), this->getMax()), result.constant(TensorT(0))).eval();
    };
    std::string getName() const { return "BatchNormGradTensorOp"; };
    //private:
    //	friend class cereal::access;
    //	template<class Archive>
    //	void serialize(Archive& archive) {
    //		archive(cereal::base_class<ActivationTensorOp<TensorT, DeviceT>>(this));
    //	}
  };

  template<typename TensorT, typename DeviceT>
  struct GradientCheckTensorOp {
    void operator()(TensorT* x_I, TensorT* x_f_plus, TensorT* x_f_neg, TensorT* x_b, TensorT* diff,
      const int& batch_size, const int& memory_size, const int& layer_size, const int& time_step, DeviceT& device) const {
      // create the forward propogation offsets
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x_I_values(x_I, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x_f_plus_values(x_f_plus, batch_size, memory_size, layer_size);
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x_f_neg_values(x_f_neg, batch_size, memory_size, layer_size);
      x_f_plus_values.device(device) = x_I_values + x_I_values.constant(eps_);
      x_f_neg_values.device(device) = x_I_values - x_I_values.constant(eps_);

      // calculate the approximate gradient
      forward_->operator()(x_f_plus, x_f_plus, batch_size, memory_size, layer_size, time_step, device);
      forward_->operator()(x_f_neg, x_f_neg, batch_size, memory_size, layer_size, time_step, device);
      auto gradapprox = (x_f_plus_values.chip(time_step, 1) - x_f_neg_values.chip(time_step, 1)) / x_f_plus_values.chip(time_step, 1).constant(TensorT(2) * eps_);
      std::cout << "gradapprox\n" << gradapprox << std::endl;

      // calculate the true gradient
      reverse_->operator()(x_I, x_b, batch_size, memory_size, layer_size, time_step, device);

      // calculate the normalized difference across each batch
      Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> x_b_values(x_b, batch_size, memory_size, layer_size);
      std::cout << "x_b_values\n" << x_b_values.chip(time_step, 1) << std::endl;
      auto numerator = (x_b_values.chip(time_step, 1) - gradapprox).pow(2).sum().sqrt();
      auto denominator = x_b_values.chip(time_step, 1).pow(2).sum().sqrt() + gradapprox .pow(2).sum().sqrt();
      Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> diff_value(diff);
      auto result = (denominator == denominator.constant(0)).select(denominator.constant(0), numerator / denominator);
      diff_value.device(device) = result;
    }
    TensorT eps_ = TensorT(1e-7);
    std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> forward_ = nullptr;
    std::shared_ptr<ActivationTensorOp<TensorT, DeviceT>> reverse_ = nullptr;
  };
}
//CEREAL_REGISTER_TYPE(SmartPeak::ReLUTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ReLUGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ELUTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ELUGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SigmoidTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SigmoidGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::TanHTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::TanHGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ReTanHTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ReTanHGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LinearTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LinearGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::InverseTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::InverseGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ExponentialTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ExponentialGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LogTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LogGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::PowTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::PowGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LeakyReLUTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LeakyReLUGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SinTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SinGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::CosTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::CosGradTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::BatchNormTensorOp<float, Eigen::DefaultDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::BatchNormGradTensorOp<float, Eigen::DefaultDevice>);
//
//#if COMPILE_WITH_CUDA
//CEREAL_REGISTER_TYPE(SmartPeak::ReLUTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ReLUGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ELUTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ELUGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SigmoidTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SigmoidGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::TanHTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::TanHGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ReTanHTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ReTanHGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LinearTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LinearGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::InverseTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::InverseGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ExponentialTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::ExponentialGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LogTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LogGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::PowTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::PowGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LeakyReLUTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::LeakyReLUGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SinTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::SinGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::CosTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::CosGradTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::BatchNormTensorOp<float, Eigen::GpuDevice>);
//CEREAL_REGISTER_TYPE(SmartPeak::BatchNormGradTensorOp<float, Eigen::GpuDevice>);
//#endif
//
//// TODO: double, int, etc.,

#endif //SMARTPEAK_ACTIVATIONTENSORFUNCTION_H