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

//#include <cereal/access.hpp>  // serialiation of private members
//#include <cereal/types/memory.hpp>
//#undef min // clashes with std::limit on windows in polymorphic.hpp
//#undef max // clashes with std::limit on windows in polymorphic.hpp
//#include <cereal/types/polymorphic.hpp>

namespace SmartPeak
{
  /**
    @brief Network ModelErrorData
  */
	template<typename TensorT, typename DeviceT>
  class ModelErrorData
  {
public:
    ModelErrorData() = default; ///< Default constructor
    ModelErrorData(const ModelErrorData& other)
		{
      batch_size_ = other.batch_size_;
      memory_size_ = other.memory_size_;
      n_metrics_ = other.n_metrics_;
			h_error_ = other.h_error_;
			d_error_ = other.d_error_;
      h_metric_ = other.h_metric_;
      d_metric_ = other.d_metric_;
      h_error_updated_ = other.h_error_updated_;
      d_error_updated_ = other.d_error_updated_;
      h_metric_updated_ = other.h_metric_updated_;
      d_metric_updated_ = other.d_metric_updated_;
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
      batch_size_ = other.batch_size_;
      memory_size_ = other.memory_size_;
      n_metrics_ = other.n_metrics_;
      h_error_ = other.h_error_;
      d_error_ = other.d_error_;
      h_metric_ = other.h_metric_;
      d_metric_ = other.d_metric_;
      h_error_updated_ = other.h_error_updated_;
      d_error_updated_ = other.d_error_updated_;
      h_metric_updated_ = other.h_metric_updated_;
      d_metric_updated_ = other.d_metric_updated_;
      return *this;
    }

		void setBatchSize(const size_t& batch_size) { (batch_size <= 0) ? batch_size_ = 1 : batch_size_ = batch_size; }
		void setMemorySize(const size_t& memory_size) { (memory_size <= 0) ? memory_size_ = 1 : memory_size_ = memory_size; }
    void setNMetrics(const size_t& n_metrics) { (n_metrics <= 0) ? n_metrics_ = 1: n_metrics_ = n_metrics; }
		size_t getBatchSize() const { return batch_size_; }
		size_t getMemorySize() const	{ return memory_size_; }
    size_t getNMetrics() const { return n_metrics_; }

    virtual void setError(const Eigen::Tensor<TensorT, 2>& error) = 0; ///< error setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> getError() { Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error(h_error_.get(), batch_size_, memory_size_); return error; }; ///< error copy getter
		std::shared_ptr<TensorT[]> getHErrorPointer() { return h_error_; }; ///< error pointer getter
		std::shared_ptr<TensorT[]> getDErrorPointer() { return d_error_; }; ///< error pointer getter

    virtual void setMetric(const Eigen::Tensor<TensorT, 2>& metric) = 0; ///< metric setter
    Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> getMetric() { Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> metric(h_metric_.get(), n_metrics_, memory_size_); return metric; }; ///< metric copy getter
    std::shared_ptr<TensorT[]> getHMetricPointer() { return h_metric_; }; ///< metric pointer getter
    std::shared_ptr<TensorT[]> getDMetricPointer() { return d_metric_; }; ///< metric pointer getter

		size_t getErrorTensorSize() { return batch_size_ * memory_size_ * sizeof(TensorT); }; ///< Get the size of each tensor in bytes
    size_t getMetricTensorSize() { return n_metrics_ * memory_size_ * sizeof(TensorT); }; ///< Get the size of each tensor in bytes

		void initModelErrorData(const int& batch_size, const int& memory_size, const int& n_metrics);

		virtual bool syncHAndDError(DeviceT& device) = 0;
    virtual bool syncHAndDMetric(DeviceT& device) = 0;

		std::pair<bool, bool> getErrorStatus() { return std::make_pair(h_error_updated_, d_error_updated_); };
    std::pair<bool, bool> getMetricStatus() { return std::make_pair(h_metric_updated_, d_metric_updated_); };

protected:
		size_t batch_size_ = 1; ///< Mini batch size
		size_t memory_size_ = 2; ///< Memory size
    size_t n_metrics_ = 2; ///< The number of model metrics
    /**
      @brief output, error and derivative have the following dimensions:
        rows: # of samples, cols: # of time steps
        where the number of samples spans 0 to n samples
        and the number of time steps spans m time points to 0
    */		
		std::shared_ptr<TensorT[]> h_error_ = nullptr;
		std::shared_ptr<TensorT[]> d_error_ = nullptr;
    std::shared_ptr<TensorT[]> h_metric_ = nullptr;
    std::shared_ptr<TensorT[]> d_metric_ = nullptr;
		bool h_error_updated_ = false;
		bool d_error_updated_ = false;
    bool h_metric_updated_ = false;
    bool d_metric_updated_ = false;
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(batch_size_, memory_size_, h_error_, d_error_, h_error_updated_, d_error_updated_);
	//	}
  };

	template<typename TensorT, typename DeviceT>
	inline void ModelErrorData<TensorT, DeviceT>::initModelErrorData(const int& batch_size, const int& memory_size, const int& n_metrics)
	{
		setBatchSize(batch_size);	setMemorySize(memory_size); setNMetrics(n_metrics);
		Eigen::Tensor<TensorT, 2> zero(batch_size, memory_size); zero.setZero();
		setError(zero);
    Eigen::Tensor<TensorT, 2> zero_metric(n_metrics, memory_size); zero_metric.setZero();
    setMetric(zero_metric);
	}

	template<typename TensorT>
	class ModelErrorDataCpu : public ModelErrorData<TensorT, Eigen::DefaultDevice> {
	public:
		void setError(const Eigen::Tensor<TensorT, 2>& error) override {
			TensorT* h_error = new TensorT[this->batch_size_*this->memory_size_];
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_copy(h_error, this->batch_size_, this->memory_size_);
			error_copy = error;
			this->h_error_.reset(h_error);
			this->h_error_updated_ = true;
			this->d_error_updated_ = true;
		}; ///< error setter
		bool syncHAndDError(Eigen::DefaultDevice& device) override { return true; }
    void setMetric(const Eigen::Tensor<TensorT, 2>& metric) override {
      TensorT* h_metric = new TensorT[this->n_metrics_*this->memory_size_];
      // copy the tensor
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> metric_copy(h_metric, this->n_metrics_, this->memory_size_);
      metric_copy = metric;
      this->h_metric_.reset(h_metric);
      this->h_metric_updated_ = true;
      this->d_metric_updated_ = true;
    }; ///< metric setter
    bool syncHAndDMetric(Eigen::DefaultDevice& device) override { return true; }
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ModelErrorData<TensorT, Eigen::DefaultDevice>>(this));
	//	}
	};

#if COMPILE_WITH_CUDA

	template<typename TensorT>
	class ModelErrorDataGpu : public ModelErrorData<TensorT, Eigen::GpuDevice> {
	public:
		void setError(const Eigen::Tensor<TensorT, 2>& error) override {
			// allocate cuda and pinned host memory
			TensorT* d_error;
			TensorT* h_error;
			assert(cudaMalloc((void**)(&d_error), getErrorTensorSize()) == cudaSuccess);
			assert(cudaHostAlloc((void**)(&h_error), getErrorTensorSize(), cudaHostAllocDefault) == cudaSuccess);
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_copy(h_error, this->batch_size_, this->memory_size_);
			error_copy = error;
			// define the deleters
      auto h_deleter = [&](TensorT* ptr) { assert(cudaFreeHost(ptr) == cudaSuccess); };
      auto d_deleter = [&](TensorT* ptr) { assert(cudaFree(ptr) == cudaSuccess); };
			this->h_error_.reset(h_error, h_deleter);
			this->d_error_.reset(d_error, d_deleter);
			this->h_error_updated_ = true;
			this->d_error_updated_ = false;
		}; ///< error setter
		bool syncHAndDError(Eigen::GpuDevice& device) override {
			if (this->h_error_updated_ && !this->d_error_updated_) {
				device.memcpyHostToDevice(this->d_error_.get(), this->h_error_.get(), getErrorTensorSize());
				this->d_error_updated_ = true;
				this->h_error_updated_ = false;
				return true;
			}
			else if (!this->h_error_updated_ && this->d_error_updated_) {
				device.memcpyDeviceToHost(this->h_error_.get(), this->d_error_.get(), getErrorTensorSize());
				this->h_error_updated_ = true;
				this->d_error_updated_ = false;
				return true;
			}
			else {
				std::cout << "Both host and device are syncHAndDronized." << std::endl;
				return false;
			}
		}
    void setMetric(const Eigen::Tensor<TensorT, 2>& metric) override {
      // allocate cuda and pinned host memory
      TensorT* d_metric;
      TensorT* h_metric;
      assert(cudaMalloc((void**)(&d_metric), getMetricTensorSize()) == cudaSuccess);
      assert(cudaHostAlloc((void**)(&h_metric), getMetricTensorSize(), cudaHostAllocDefault) == cudaSuccess);
      // copy the tensor
      Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> metric_copy(h_metric, this->n_metrics_, this->memory_size_);
      metric_copy = metric;
      // define the deleters
      auto h_deleter = [&](TensorT* ptr) { assert(cudaFreeHost(ptr) == cudaSuccess); };
      auto d_deleter = [&](TensorT* ptr) { assert(cudaFree(ptr) == cudaSuccess); };
      this->h_metric_.reset(h_metric, h_deleter);
      this->d_metric_.reset(d_metric, d_deleter);
      this->h_metric_updated_ = true;
      this->d_metric_updated_ = false;
    }; ///< metric setter
    bool syncHAndDMetric(Eigen::GpuDevice& device) override {
      if (this->h_metric_updated_ && !this->d_metric_updated_) {
        device.memcpyHostToDevice(this->d_metric_.get(), this->h_metric_.get(), getMetricTensorSize());
        this->d_metric_updated_ = true;
        this->h_metric_updated_ = false;
        return true;
      }
      else if (!this->h_metric_updated_ && this->d_metric_updated_) {
        device.memcpyDeviceToHost(this->h_metric_.get(), this->d_metric_.get(), getMetricTensorSize());
        this->h_metric_updated_ = true;
        this->d_metric_updated_ = false;
        return true;
      }
      else {
        std::cout << "Both host and device are syncHAndDronized." << std::endl;
        return false;
      }
    }
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<ModelErrorData<TensorT, Eigen::GpuDevice>>(this));
	//	}
	};
#endif
}

//CEREAL_REGISTER_TYPE(SmartPeak::ModelErrorDataCpu<float>);
//// TODO: add double, int, etc.
//#if COMPILE_WITH_CUDA
//CEREAL_REGISTER_TYPE(SmartPeak::ModelErrorDataGpu<float>);
//// TODO: add double, int, etc.
//#endif

#endif //SMARTPEAK_MODELERRORDATA_H