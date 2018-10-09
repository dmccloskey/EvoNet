/**TODO:  Add copyright*/

#ifndef SMARTPEAK_NODEDATA_H
#define SMARTPEAK_NODEDATA_H

#ifndef EVONET_CUDA
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
    @brief Network NodeData
  */
	template<typename TensorT>
  class NodeData
  {
public:
    NodeData() = default; ///< Default constructor
    NodeData(const NodeData& other)
		{
			input_ = other.input_;
			output_ = other.output_;
			error_ = other.error_;
			derivative_ = other.derivative_;
			dt_ = other.dt_;
			h_input_ = other.h_input_;
			h_output_ = other.h_output_;
			h_error_ = other.h_error_;
			h_derivative_ = other.h_derivative_;
			h_dt_ = other.h_dt_;
			d_input_ = other.d_input_;
			d_output_ = other.d_output_;
			d_error_ = other.d_error_;
			d_derivative_ = other.d_derivative_;
			d_dt_ = other.d_dt_;
		};
    ~NodeData() = default; ///< Default destructor

    inline bool operator==(const NodeData& other) const
    {
      return
        std::tie(
       
        ) == std::tie(

        )
      ;
    }

    inline bool operator!=(const NodeData& other) const
    {
      return !(*this == other);
    }

    inline NodeData& operator=(const NodeData& other)
    { 
			input_ = other.input_;
      output_ = other.output_;
      error_ = other.error_;
      derivative_ = other.derivative_;
      dt_ = other.dt_;
			h_input_ = other.h_input_;
			h_output_ = other.h_output_;
			h_error_ = other.h_error_;
			h_derivative_ = other.h_derivative_;
			h_dt_ = other.h_dt_;
			d_input_ = other.d_input_;
			d_output_ = other.d_output_;
			d_error_ = other.d_error_;
			d_derivative_ = other.d_derivative_;
			d_dt_ = other.d_dt_;
      return *this;
    }

		void setBatchSize(const size_t& batch_size) { batch_size_ = batch_size; }
		void setMemorySize(const size_t& memory_size) { memory_size_ = memory_size; }
		size_t getBatchSize() const { return batch_size_; }
		size_t getMemorySize() const	{ return memory_size_; }

		virtual void setInput(TensorT* input) = 0; ///< input setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> getInput() { std::shared_ptr<TensorT> h_input = h_input_; Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> input(h_input.get(), batch_size_, memory_size_); return input; }; ///< input copy getter
		std::shared_ptr<TensorT> getHInputPointer() { return h_input_; }; ///< input pointer getter
		std::shared_ptr<TensorT> getDInputPointer() { return d_input_; }; ///< input pointer getter

    virtual void setOutput(TensorT* output) = 0; ///< output setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> getOutput() { Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> output(h_output_.get(), batch_size_, memory_size_); return output; }; ///< output copy getter
		std::shared_ptr<TensorT> getHOutputPointer() { return h_output_; }; ///< output pointer getter
		std::shared_ptr<TensorT> getDOutputPointer() { return d_output_; }; ///< output pointer getter

    virtual void setError(TensorT* error) = 0; ///< error setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> getError() { Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error(h_error_.get(), batch_size_, memory_size_); return error; }; ///< error copy getter
		std::shared_ptr<TensorT> getHErrorPointer() { return h_error_; }; ///< error pointer getter
		std::shared_ptr<TensorT> getDErrorPointer() { return d_error_; }; ///< error pointer getter

    virtual void setDerivative(TensorT* derivative) = 0; ///< derivative setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> getDerivative() { Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> derivative(h_derivative_.get(), batch_size_, memory_size_); return derivative; }; ///< derivative copy getter
		std::shared_ptr<TensorT> getHDerivativePointer() { return h_derivative_; }; ///< derivative pointer getter
		std::shared_ptr<TensorT> getDDerivativePointer() { return d_derivative_; }; ///< derivative pointer getter

    virtual void setDt(TensorT* dt) = 0; ///< dt setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> getDt() { Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> dt(h_dt_.get(), batch_size_, memory_size_); return dt;	}; ///< dt copy getter
		std::shared_ptr<TensorT> getHDtPointer() { return h_dt_; }; ///< dt pointer getter
		std::shared_ptr<TensorT> getDDtPointer() { return d_dt_; }; ///< dt pointer getter

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
		std::shared_ptr<TensorT> h_input_ = nullptr;
		std::shared_ptr<TensorT> h_output_ = nullptr;
		std::shared_ptr<TensorT> h_error_ = nullptr;
		std::shared_ptr<TensorT> h_derivative_ = nullptr;
		std::shared_ptr<TensorT> h_dt_ = nullptr;
		std::shared_ptr<TensorT> d_input_ = nullptr;
		std::shared_ptr<TensorT> d_output_ = nullptr;
		std::shared_ptr<TensorT> d_error_ = nullptr;
		std::shared_ptr<TensorT> d_derivative_ = nullptr;
		std::shared_ptr<TensorT> d_dt_ = nullptr;
  };

	template<typename TensorT>
	class NodeDataCpu : public NodeData<TensorT> {
	public:
		void setInput(TensorT* input) { h_input_.reset(std::move(input)); }; ///< input setter
		void setOutput(TensorT* output) { h_output_.reset(std::move(output)); }; ///< output setter
		void setError(TensorT* error) { h_error_.reset(std::move(error)); }; ///< error setter
		void setDerivative(TensorT* derivative) { h_derivative_.reset(std::move(derivative)); }; ///< derivative setter
		void setDt(TensorT* dt) { h_dt_.reset(std::move(dt)); }; ///< dt setter
	};

#ifndef EVONET_CUDA

	template<typename TensorT>
	class NodeDataGpu : public NodeData<TensorT> {
	public:
		void setInput(TensorT* input) {
			// allocate cuda and pinned host memory
			TensorT* d_input;
			TensorT* h_input;
			assert(cudaMalloc((void**)(&d_input), getTensorSize()) == cudaSuccess);
			assert(cudaHostAlloc((void**)(&h_input), getTensorSize(), cudaHostAllocDefault) == cudaSuccess);
			// define the deleters
			auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
			auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
			h_input = std::move(input);
			h_input_.reset(h_input, h_deleter); 
			d_input_.reset(d_input, d_deleter);
		}; ///< input setter
		void setOutput(TensorT* output) {
			// allocate cuda and pinned host memory
			TensorT* d_output;
			TensorT* h_output;
			assert(cudaMalloc((void**)(&d_output), getTensorSize()) == cudaSuccess);
			assert(cudaHostAlloc((void**)(&h_output), getTensorSize(), cudaHostAllocDefault) == cudaSuccess);
			// define the deleters
			auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
			auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
			h_output = std::move(output);
			h_output_.reset(h_output, h_deleter);
			d_output_.reset(d_output, d_deleter);
		}; ///< output setter
		void setError(TensorT* error) {
			// allocate cuda and pinned host memory
			TensorT* d_error;
			TensorT* h_error;
			assert(cudaMalloc((void**)(&d_error), getTensorSize()) == cudaSuccess);
			assert(cudaHostAlloc((void**)(&h_error), getTensorSize(), cudaHostAllocDefault) == cudaSuccess);
			// define the deleters
			auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
			auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
			h_error = std::move(error);
			h_error_.reset(h_error, h_deleter);
			d_error_.reset(d_error, d_deleter);
		}; ///< error setter
		void setDerivative(TensorT* derivative) {
			// allocate cuda and pinned host memory
			TensorT* d_derivative;
			TensorT* h_derivative;
			assert(cudaMalloc((void**)(&d_derivative), getTensorSize()) == cudaSuccess);
			assert(cudaHostAlloc((void**)(&h_derivative), getTensorSize(), cudaHostAllocDefault) == cudaSuccess);
			// define the deleters
			auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
			auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
			h_derivative = std::move(derivative);
			h_derivative_.reset(h_derivative, h_deleter);
			d_derivative_.reset(d_derivative, d_deleter);
		}; ///< derivative setter
		void setDt(TensorT* dt) {
			// allocate cuda and pinned host memory
			TensorT* d_dt;
			TensorT* h_dt;
			assert(cudaMalloc((void**)(&d_dt), getTensorSize()) == cudaSuccess);
			assert(cudaHostAlloc((void**)(&h_dt), getTensorSize(), cudaHostAllocDefault) == cudaSuccess);
			// define the deleters
			auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
			auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
			h_dt = std::move(dt);
			h_dt_.reset(h_dt, h_deleter);
			d_dt_.reset(d_dt, d_deleter);
		}; ///< dt setter
	};
#endif
}

#endif //SMARTPEAK_NODEDATA_H