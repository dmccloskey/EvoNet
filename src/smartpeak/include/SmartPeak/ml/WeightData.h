/**TODO:  Add copyright*/

#ifndef SMARTPEAK_WEIGHTDATA_H
#define SMARTPEAK_WEIGHTDATA_H

#if EVONET_CUDA_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>

namespace SmartPeak
{
  /**
    @brief WeightData
  */
	template<typename TensorT>
  class WeightData
  {
public:
    WeightData() = default; ///< Default constructor
    WeightData(const WeightData& other)
		{
			h_weight_ = other.h_weight_;
			d_weight_ = other.d_weight_;
		};
    ~WeightData() = default; ///< Default destructor

    inline bool operator==(const WeightData& other) const
    {
			return
				std::tie(
					h_weight_,
					d_weight_
				) == std::tie(
					other.h_weight_,
					other.d_weight_
        );
    }

    inline bool operator!=(const WeightData& other) const
    {
      return !(*this == other);
    }

    inline WeightData& operator=(const WeightData& other)
    {
			h_weight_ = other.h_weight_;
			d_weight_ = other.d_weight_;
      return *this;
    }

		virtual void setWeight(const TensorT& weight) = 0; ///< weight setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> getWeight() { std::shared_ptr<TensorT> h_weight = h_weight_; Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> weight(h_weight.get()); return weight; }; ///< weight copy getter
		std::shared_ptr<TensorT> getHWeightPointer() { return h_weight_; }; ///< weight pointer getter
		std::shared_ptr<TensorT> getDWeightPointer() { return d_weight_; }; ///< weight pointer getter

		size_t getTensorSize() { return sizeof(TensorT); }; ///< Get the size of each tensor in bytes

protected:
		std::shared_ptr<TensorT> h_weight_ = nullptr;
		std::shared_ptr<TensorT> d_weight_ = nullptr;
  };

	template<typename TensorT>
	class WeightDataCpu : public WeightData<TensorT> {
	public:
		void setWeight(const TensorT& weight) {
			TensorT* h_weight = new TensorT;
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> weight_copy(h_weight);
			weight_copy.setConstant(weight);
			h_weight_.reset(std::move(h_weight));
		}; ///< weight setter
	};

#if EVONET_CUDA_CUDA

	template<typename TensorT>
	class WeightDataGpu : public WeightData<TensorT> {
	public:
		void setWeight(const TensorT& weight) {
			// allocate cuda and pinned host memory
			TensorT* d_weight;
			TensorT* h_weight;
			assert(cudaMalloc((void**)(&d_weight), getTensorSize()) == cudaSuccess);
			assert(cudaHostAlloc((void**)(&h_weight), getTensorSize(), cudaHostAllocDefault) == cudaSuccess);
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 0>> weight_copy(h_weight);
			weight_copy.setConstant(weight);
			// define the deleters
			auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
			auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
			h_weight_.reset(h_weight, h_deleter); 
			d_weight_.reset(d_weight, d_deleter);
		}; ///< weight setter
	};
#endif
}

#endif //SMARTPEAK_WEIGHTDATA_H