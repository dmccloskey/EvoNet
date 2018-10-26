/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELERRORDATA_H
#define SMARTPEAK_MODELERRORDATA_H

#if COMPILE_WITH_CUDA
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
    @brief Network ModelErrorData
  */
	template<typename TensorT>
  class ModelErrorData
  {
public:
    ModelErrorData() = default; ///< Default constructor
    ModelErrorData(const ModelErrorData& other)
		{
			h_error_ = other.h_error_;
			d_error_ = other.d_error_;
		};
    ~ModelErrorData() = default; ///< Default destructor

    inline bool operator==(const ModelErrorData& other) const
    {
      return
        std::tie(
       
        ) == std::tie(

        )
      ;
    }

    inline bool operator!=(const ModelErrorData& other) const
    {
      return !(*this == other);
    }

    inline ModelErrorData& operator=(const ModelErrorData& other)
    { 
			h_error_ = other.h_error_;
			d_error_ = other.d_error_;
      return *this;
    }

		void setBatchSize(const size_t& batch_size) { batch_size_ = batch_size; }
		void setMemorySize(const size_t& memory_size) { memory_size_ = memory_size; }
		size_t getBatchSize() const { return batch_size_; }
		size_t getMemorySize() const	{ return memory_size_; }

    virtual void setError(const Eigen::Tensor<TensorT, 2>& error) = 0; ///< error setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> getError() { Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error(h_error_.get(), batch_size_, memory_size_); return error; }; ///< error copy getter
		std::shared_ptr<TensorT> getHErrorPointer() { return h_error_; }; ///< error pointer getter
		std::shared_ptr<TensorT> getDErrorPointer() { return d_error_; }; ///< error pointer getter

		size_t getTensorSize() { return batch_size_ * memory_size_ * sizeof(TensorT); }; ///< Get the size of each tensor in bytes

protected:
		size_t batch_size_ = 1; ///< Mini batch size
		size_t memory_size_ = 2; ///< Memory size
    /**
      @brief output, error and derivative have the following dimensions:
        rows: # of samples, cols: # of time steps
        where the number of samples spans 0 to n samples
        and the number of time steps spans m time points to 0
    */		
		std::shared_ptr<TensorT> h_error_ = nullptr;
		std::shared_ptr<TensorT> d_error_ = nullptr;
  };

	template<typename TensorT>
	class ModelErrorDataCpu : public ModelErrorData<TensorT> {
	public:
		void setError(const Eigen::Tensor<TensorT, 2>& error) {
			TensorT* h_error = new TensorT[this->batch_size_*this->memory_size_];
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_copy(h_error, this->batch_size_, this->memory_size_);
			error_copy = error;
			this->h_error_.reset(h_error);
		}; ///< error setter
	};

#if COMPILE_WITH_CUDA

	template<typename TensorT>
	class ModelErrorDataGpu : public ModelErrorData<TensorT> {
	public:
		void setError(const Eigen::Tensor<TensorT, 2>& error) {
			// allocate cuda and pinned host memory
			TensorT* d_error;
			TensorT* h_error;
			assert(cudaMalloc((void**)(&d_error), getTensorSize()) == cudaSuccess);
			assert(cudaHostAlloc((void**)(&h_error), getTensorSize(), cudaHostAllocDefault) == cudaSuccess);
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_copy(h_error, this->batch_size_, this->memory_size_);
			error_copy = error;
			// define the deleters
			auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
			auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
			this->h_error_.reset(h_error, h_deleter);
			this->d_error_.reset(d_error, d_deleter);
		}; ///< error setter
	};
#endif
}

#endif //SMARTPEAK_MODELERRORDATA_H