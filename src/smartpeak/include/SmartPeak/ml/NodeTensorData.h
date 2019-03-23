/**TODO:  Add copyright*/

#ifndef SMARTPEAK_NODETENSORDATA_H
#define SMARTPEAK_NODETENSORDATA_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <unsupported/Eigen/CXX11/Tensor>
#include <SmartPeak/ml/Node.h>
#include <memory>

//#include <cereal/access.hpp>  // serialiation of private members
//#include <cereal/types/memory.hpp>
//#undef min // clashes with std::limit on windows in polymorphic.hpp
//#undef max // clashes with std::limit on windows in polymorphic.hpp
//#include <cereal/types/polymorphic.hpp>

namespace SmartPeak
{
  /**
    @brief Network NodeMatrixData
  */
	template<typename TensorT, typename DeviceT>
  class NodeTensorData
  {
public:
    NodeTensorData() = default; ///< Default constructor
    NodeTensorData(const NodeTensorData& other)
		{
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
			batch_size_ = other.batch_size_;
			memory_size_ = other.memory_size_;
			layer_size_ = other.layer_size_;
			h_input_updated_ = other.h_input_updated_;
			h_output_updated_ = other.h_output_updated_;
			h_error_updated_ = other.h_error_updated_;
			h_derivative_updated_ = other.h_derivative_updated_;
			h_dt_updated_ = other.h_dt_updated_;
			d_input_updated_ = other.d_input_updated_;
			d_output_updated_ = other.d_output_updated_;
			d_error_updated_ = other.d_error_updated_;
			d_derivative_updated_ = other.d_derivative_updated_;
			d_dt_updated_ = other.d_dt_updated_;
		};
    ~NodeTensorData() = default; ///< Default destructor

    inline bool operator==(const NodeTensorData& other) const
    {
      return
        std::tie(
       
        ) == std::tie(

        )
      ;
    }

    inline bool operator!=(const NodeTensorData& other) const
    {
      return !(*this == other);
    }

    inline NodeTensorData& operator=(const NodeTensorData& other)
    { 
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
			batch_size_ = other.batch_size_;
			memory_size_ = other.memory_size_;
			layer_size_ = other.layer_size_;
			h_input_updated_ = other.h_input_updated_;
			h_output_updated_ = other.h_output_updated_;
			h_error_updated_ = other.h_error_updated_;
			h_derivative_updated_ = other.h_derivative_updated_;
			h_dt_updated_ = other.h_dt_updated_;
			d_input_updated_ = other.d_input_updated_;
			d_output_updated_ = other.d_output_updated_;
			d_error_updated_ = other.d_error_updated_;
			d_derivative_updated_ = other.d_derivative_updated_;
			d_dt_updated_ = other.d_dt_updated_;
      return *this;
    }

		void setBatchSize(const int& batch_size) { batch_size_ = batch_size; }
		void setMemorySize(const int& memory_size) { memory_size_ = memory_size; }
		void setLayerSize(const int& layer_size) { layer_size_ = layer_size; }
    void setLayerIntegration(const std::string& layer_integration) { layer_integration_ = layer_integration; }
		int getBatchSize() const { return batch_size_; }
		int getMemorySize() const	{ return memory_size_; }
		int getLayerSize() const { return layer_size_; }
    std::string getLayerIntegration() const { return layer_integration_; }

		virtual void setInput(const Eigen::Tensor<TensorT, 3>& input) = 0; ///< input setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> getInput() { std::shared_ptr<TensorT> h_input = h_input_; Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> input(h_input.get(), batch_size_, memory_size_, layer_size_); return input; }; ///< input copy getter
		std::shared_ptr<TensorT> getHInputPointer() { return h_input_; }; ///< input pointer getter
		std::shared_ptr<TensorT> getDInputPointer() { return d_input_; }; ///< input pointer getter

    virtual void setOutput(const Eigen::Tensor<TensorT, 3>& output) = 0; ///< output setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> getOutput() { Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> output(h_output_.get(), batch_size_, memory_size_, layer_size_); return output; }; ///< output copy getter
		std::shared_ptr<TensorT> getHOutputPointer() { return h_output_; }; ///< output pointer getter
		std::shared_ptr<TensorT> getDOutputPointer() { return d_output_; }; ///< output pointer getter

    virtual void setError(const Eigen::Tensor<TensorT, 3>& error) = 0; ///< error setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> getError() { Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> error(h_error_.get(), batch_size_, memory_size_, layer_size_); return error; }; ///< error copy getter
		std::shared_ptr<TensorT> getHErrorPointer() { return h_error_; }; ///< error pointer getter
		std::shared_ptr<TensorT> getDErrorPointer() { return d_error_; }; ///< error pointer getter

    virtual void setDerivative(const Eigen::Tensor<TensorT, 3>& derivative) = 0; ///< derivative setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> getDerivative() { Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> derivative(h_derivative_.get(), batch_size_, memory_size_, layer_size_); return derivative; }; ///< derivative copy getter
		std::shared_ptr<TensorT> getHDerivativePointer() { return h_derivative_; }; ///< derivative pointer getter
		std::shared_ptr<TensorT> getDDerivativePointer() { return d_derivative_; }; ///< derivative pointer getter

    virtual void setDt(const Eigen::Tensor<TensorT, 3>& dt) = 0; ///< dt setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> getDt() { Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> dt(h_dt_.get(), batch_size_, memory_size_, layer_size_); return dt;	}; ///< dt copy getter
		std::shared_ptr<TensorT> getHDtPointer() { return h_dt_; }; ///< dt pointer getter
		std::shared_ptr<TensorT> getDDtPointer() { return d_dt_; }; ///< dt pointer getter

		size_t getTensorSize() { return batch_size_ * memory_size_ * layer_size_ * sizeof(TensorT); }; ///< Get the size of each tensor in bytes

		void initNodeTensorData(const int& batch_size, const int& memory_size, const int& layer_size, const NodeType& node_type, const std::string& node_integration, const bool& train); ///< initialize the node according to node type

		virtual bool syncHAndDInput(DeviceT& device) = 0;
		virtual bool syncHAndDOutput(DeviceT& device) = 0;
		virtual bool syncHAndDError(DeviceT& device) = 0;
		virtual bool syncHAndDDerivative(DeviceT& device) = 0;
		virtual bool syncHAndDDt(DeviceT& device) = 0;

		std::pair<bool, bool> getInputStatus() { return std::make_pair(h_input_updated_, d_input_updated_);	};
		std::pair<bool, bool> getOutputStatus() { return std::make_pair(h_output_updated_, d_output_updated_); };
		std::pair<bool, bool> getErrorStatus() { return std::make_pair(h_error_updated_, d_error_updated_); };
		std::pair<bool, bool> getDerivativeStatus() { return std::make_pair(h_derivative_updated_, d_derivative_updated_); };
		std::pair<bool, bool> getDtStatus() { return std::make_pair(h_dt_updated_, d_dt_updated_); };

protected:
		int batch_size_ = 1; ///< Mini batch size
		int memory_size_ = 2; ///< Memory size
		int layer_size_ = 1; ///< Layer size
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
		std::shared_ptr<TensorT> h_dt_ = nullptr; // [TODO: change to drop probability]
		std::shared_ptr<TensorT> d_input_ = nullptr;
		std::shared_ptr<TensorT> d_output_ = nullptr;
		std::shared_ptr<TensorT> d_error_ = nullptr;
		std::shared_ptr<TensorT> d_derivative_ = nullptr;
		std::shared_ptr<TensorT> d_dt_ = nullptr;

		bool h_input_updated_ = false;
		bool h_output_updated_ = false;
		bool h_error_updated_ = false;
		bool h_derivative_updated_ = false;
		bool h_dt_updated_ = false;

		bool d_input_updated_ = false;
		bool d_output_updated_ = false;
		bool d_error_updated_ = false;
		bool d_derivative_updated_ = false;
		bool d_dt_updated_ = false;

    std::string layer_integration_;

	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(batch_size_, memory_size_, layer_size_, 
	//		h_input_, h_output_, h_error_, h_derivative_, h_dt_,
	//		d_input_, d_output_, d_error_, d_derivative_, d_dt_,
	//		h_input_updated_, h_output_updated_, h_error_updated_, h_derivative_updated_, h_dt_updated_,
	//		d_input_updated_, d_output_updated_, d_error_updated_, d_derivative_updated_, d_dt_updated_);
	//	}
  };

	template<typename TensorT, typename DeviceT>
	inline void NodeTensorData<TensorT, DeviceT>::initNodeTensorData(const int& batch_size, const int& memory_size, const int& layer_size, const NodeType& node_type, const std::string& node_integration, const bool& train)
	{
		setBatchSize(batch_size);	setMemorySize(memory_size);	setLayerSize(layer_size);
    setLayerIntegration(node_integration);
		// Template zero and one tensor
		Eigen::Tensor<TensorT, 3> zero(batch_size, memory_size, layer_size); zero.setConstant(0);
		Eigen::Tensor<TensorT, 3> one(batch_size, memory_size, layer_size);	one.setConstant(1);
		// set the input, error, and derivatives
		setError(zero);	setDerivative(zero);
		setDt(one);
		//// set Drop probabilities [TODO: broke when adding NodeData...]
		//if (train) {
		//	setDt(one.unaryExpr(RandBinaryOp<TensorT>(getDropProbability())));
		//} else {
		//	setDt(one);
		//}
		// corrections for specific node types
		if (node_type == NodeType::bias) {
			setOutput(one);
      setInput(zero);
		}
		else if (node_type == NodeType::input) {
			setOutput(zero);  // Check for `node_integration` == "ProdOp" and set to 1
      setInput(zero);
		}
		else if (node_type == NodeType::zero) {
			setOutput(zero);
      setInput(zero);
		}
    else if (node_integration == "ProdOp" || node_integration == "ProdSCOp") {
      setOutput(zero);
      setInput(one);
    }
		else {
			setOutput(zero);
      setInput(zero);
		}
	}

	template<typename TensorT>
	class NodeTensorDataCpu : public NodeTensorData<TensorT, Eigen::DefaultDevice> {
	public:
		void setInput(const Eigen::Tensor<TensorT, 3>& input) {
			TensorT* h_input = new TensorT[this->batch_size_*this->memory_size_*this->layer_size_];
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> input_copy(h_input, this->batch_size_, this->memory_size_, this->layer_size_);
			input_copy = input;
			//auto h_deleter = [&](TensorT* ptr) { delete[] ptr; };
			//this->h_input_.reset(h_input, h_deleter);
			this->h_input_.reset(h_input);
			this->h_input_updated_ = true;
			this->d_input_updated_ = true;
		}; ///< input setter
		void setOutput(const Eigen::Tensor<TensorT, 3>& output) {
			TensorT* h_output = new TensorT[this->batch_size_*this->memory_size_*this->layer_size_];
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> output_copy(h_output, this->batch_size_, this->memory_size_, this->layer_size_);
			output_copy = output;
			//auto h_deleter = [&](TensorT* ptr) { delete[] ptr; };
			//this->h_output_.reset(h_output, h_deleter);
			this->h_output_.reset(h_output);
			this->h_output_updated_ = true;
			this->d_output_updated_ = true;
		}; ///< output setter
		void setError(const Eigen::Tensor<TensorT, 3>& error) {
			TensorT* h_error = new TensorT[this->batch_size_*this->memory_size_*this->layer_size_];
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> error_copy(h_error, this->batch_size_, this->memory_size_, this->layer_size_);
			error_copy = error;
			//auto h_deleter = [&](TensorT* ptr) { delete[] ptr; };
			//this->h_error_.reset(h_error, h_deleter);
			this->h_error_.reset(h_error);
			this->h_error_updated_ = true;
			this->d_error_updated_ = true;
		}; ///< error setter
		void setDerivative(const Eigen::Tensor<TensorT, 3>& derivative) {
			TensorT* h_derivative = new TensorT[this->batch_size_*this->memory_size_*this->layer_size_];
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> derivative_copy(h_derivative, this->batch_size_, this->memory_size_, this->layer_size_);
			derivative_copy = derivative;
			//auto h_deleter = [&](TensorT* ptr) { delete[] ptr; };
			//this->h_derivative_.reset(h_derivative, h_deleter);
			this->h_derivative_.reset(h_derivative);
			this->h_derivative_updated_ = true;
			this->d_derivative_updated_ = true;
		}; ///< derivative setter
		void setDt(const Eigen::Tensor<TensorT, 3>& dt) {
			TensorT* h_dt = new TensorT[this->batch_size_*this->memory_size_*this->layer_size_];
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> dt_copy(h_dt, this->batch_size_, this->memory_size_, this->layer_size_);
			dt_copy = dt;
			//auto h_deleter = [&](TensorT* ptr) { delete[] ptr; };
			//this->h_dt_.reset(h_dt, h_deleter);
			this->h_dt_.reset(h_dt);
			this->h_dt_updated_ = true;
			this->d_dt_updated_ = true;
		}; ///< dt setter
		bool syncHAndDInput(Eigen::DefaultDevice& device) { return true; }
		bool syncHAndDOutput(Eigen::DefaultDevice& device) {	return true; }
		bool syncHAndDError(Eigen::DefaultDevice& device) { return true; }
		bool syncHAndDDerivative(Eigen::DefaultDevice& device) { return true; }
		bool syncHAndDDt(Eigen::DefaultDevice& device) { return true; }
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<NodeTensorData<TensorT, Eigen::DefaultDevice>>(this));
	//	}
	};

#if COMPILE_WITH_CUDA

	template<typename TensorT>
	class NodeTensorDataGpu : public NodeTensorData<TensorT, Eigen::GpuDevice> {
	public:
		void setInput(const Eigen::Tensor<TensorT, 3>& input) {
			// allocate cuda and pinned host memory
			TensorT* d_input;
			TensorT* h_input;
			assert(cudaMalloc((void**)(&d_input), getTensorSize()) == cudaSuccess);
			assert(cudaHostAlloc((void**)(&h_input), getTensorSize(), cudaHostAllocDefault ) == cudaSuccess);
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> input_copy(h_input, this->batch_size_, this->memory_size_, this->layer_size_);
			input_copy = input;
			// define the deleters
			auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
			auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
			this->h_input_.reset(h_input, h_deleter); 
			this->d_input_.reset(d_input, d_deleter);
			this->h_input_updated_ = true;
			this->d_input_updated_ = false;
		}; ///< input setter
		void setOutput(const Eigen::Tensor<TensorT, 3>& output) {
			// allocate cuda and pinned host memory
			TensorT* d_output;
			TensorT* h_output;
			assert(cudaMalloc((void**)(&d_output), getTensorSize()) == cudaSuccess);
			assert(cudaHostAlloc((void**)(&h_output), getTensorSize(), cudaHostAllocDefault ) == cudaSuccess);
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> output_copy(h_output, this->batch_size_, this->memory_size_, this->layer_size_);
			output_copy = output;
			// define the deleters
			auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
			auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
			this->h_output_.reset(h_output, h_deleter);
			this->d_output_.reset(d_output, d_deleter);
			this->h_output_updated_ = true;
			this->d_output_updated_ = false;
		}; ///< output setter
		void setError(const Eigen::Tensor<TensorT, 3>& error) {
			// allocate cuda and pinned host memory
			TensorT* d_error;
			TensorT* h_error;
			assert(cudaMalloc((void**)(&d_error), getTensorSize()) == cudaSuccess);
			assert(cudaHostAlloc((void**)(&h_error), getTensorSize(), cudaHostAllocDefault ) == cudaSuccess);
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> error_copy(h_error, this->batch_size_, this->memory_size_, this->layer_size_);
			error_copy = error;
			// define the deleters
			auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
			auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
			this->h_error_.reset(h_error, h_deleter);
			this->d_error_.reset(d_error, d_deleter);
			this->h_error_updated_ = true;
			this->d_error_updated_ = false;
		}; ///< error setter
		void setDerivative(const Eigen::Tensor<TensorT, 3>& derivative) {
			// allocate cuda and pinned host memory
			TensorT* d_derivative;
			TensorT* h_derivative;
			assert(cudaMalloc((void**)(&d_derivative), getTensorSize()) == cudaSuccess);
			assert(cudaHostAlloc((void**)(&h_derivative), getTensorSize(), cudaHostAllocDefault ) == cudaSuccess);
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> derivative_copy(h_derivative, this->batch_size_, this->memory_size_, this->layer_size_);
			derivative_copy = derivative;
			// define the deleters
			auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
			auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
			this->h_derivative_.reset(h_derivative, h_deleter);
			this->d_derivative_.reset(d_derivative, d_deleter);
			this->h_derivative_updated_ = true;
			this->d_derivative_updated_ = false;
		}; ///< derivative setter
		void setDt(const Eigen::Tensor<TensorT, 3>& dt) {
			// allocate cuda and pinned host memory
			TensorT* d_dt;
			TensorT* h_dt;
			assert(cudaMalloc((void**)(&d_dt), getTensorSize()) == cudaSuccess);
			assert(cudaHostAlloc((void**)(&h_dt), getTensorSize(), cudaHostAllocDefault ) == cudaSuccess);
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> dt_copy(h_dt, this->batch_size_, this->memory_size_, this->layer_size_);
			dt_copy = dt;
			// define the deleters
			auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
			auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
			this->h_dt_.reset(h_dt, h_deleter);
			this->d_dt_.reset(d_dt, d_deleter);
			this->h_dt_updated_ = true;
			this->d_dt_updated_ = false;
		}; ///< dt setter
		bool syncHAndDInput(Eigen::GpuDevice& device){
			if (this->h_input_updated_ && !this->d_input_updated_) {
				device.memcpyHostToDevice(this->d_input_.get(), this->h_input_.get(), getTensorSize());
				this->d_input_updated_ = true;
				this->h_input_updated_ = false;
				return true;
			}
			else if (!this->h_input_updated_ && this->d_input_updated_) {
				device.memcpyDeviceToHost(this->h_input_.get(), this->d_input_.get(), getTensorSize());
				this->h_input_updated_ = true;
				this->d_input_updated_ = false;
				return true;
			}
			else {
				std::cout << "Both host and device are syncHAndDronized." << std::endl;
				return false;
			}
		}
		bool syncHAndDOutput(Eigen::GpuDevice& device){
			if (this->h_output_updated_ && !this->d_output_updated_) {
				device.memcpyHostToDevice(this->d_output_.get(), this->h_output_.get(), getTensorSize());
				this->d_output_updated_ = true;
				this->h_output_updated_ = false;
				return true;
			}
			else if (!this->h_output_updated_ && this->d_output_updated_) {
				device.memcpyDeviceToHost(this->h_output_.get(), this->d_output_.get(), getTensorSize());
				this->h_output_updated_ = true;
				this->d_output_updated_ = false;
				return true;
			}
			else {
				std::cout << "Both host and device are syncHAndDronized." << std::endl;
				return false;
			}
		}
		bool syncHAndDError(Eigen::GpuDevice& device){
			if (this->h_error_updated_ && !this->d_error_updated_) {
				device.memcpyHostToDevice(this->d_error_.get(), this->h_error_.get(), getTensorSize());
				this->d_error_updated_ = true;
				this->h_error_updated_ = false;
				return true;
			}
			else if (!this->h_error_updated_ && this->d_error_updated_) {
				device.memcpyDeviceToHost(this->h_error_.get(), this->d_error_.get(), getTensorSize());
				this->h_error_updated_ = true;
				this->d_error_updated_ = false;
				return true;
			}
			else {
				std::cout << "Both host and device are syncHAndDronized." << std::endl;
				return false;
			}
		}
		bool syncHAndDDerivative(Eigen::GpuDevice& device){
			if (this->h_derivative_updated_ && !this->d_derivative_updated_) {
				device.memcpyHostToDevice(this->d_derivative_.get(), this->h_derivative_.get(), getTensorSize());
				this->d_derivative_updated_ = true;
				this->h_derivative_updated_ = false;
				return true;
			}
			else if (!this->h_derivative_updated_ && this->d_derivative_updated_) {
				device.memcpyDeviceToHost(this->h_derivative_.get(), this->d_derivative_.get(), getTensorSize());
				this->h_derivative_updated_ = true;
				this->d_derivative_updated_ = false;
				return true;
			}
			else {
				std::cout << "Both host and device are syncHAndDronized." << std::endl;
				return false;
			}
			return true;
		}
		bool syncHAndDDt(Eigen::GpuDevice& device){
			if (this->h_dt_updated_ && !this->d_dt_updated_) {
				device.memcpyHostToDevice(this->d_dt_.get(), this->h_dt_.get(), getTensorSize());
				this->d_dt_updated_ = true;
				this->h_dt_updated_ = false;
				return true;
			}
			else if (!this->h_dt_updated_ && this->d_dt_updated_) {
				device.memcpyDeviceToHost(this->h_dt_.get(), this->d_dt_.get(), getTensorSize());
				this->h_dt_updated_ = true;
				this->d_dt_updated_ = false;
				return true;
			}
			else {
				std::cout << "Both host and device are syncHAndDronized." << std::endl;
				return false;
			}
			return true;
		}
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<NodeTensorData<TensorT, Eigen::GpuDevice>>(this));
	//	}
	};
#endif
}

//CEREAL_REGISTER_TYPE(SmartPeak::NodeTensorDataCpu<float>);
//// TODO: add double, int, etc.
//#if COMPILE_WITH_CUDA
//CEREAL_REGISTER_TYPE(SmartPeak::NodeTensorDataGpu<float>);
//// TODO: add double, int, etc.
//#endif

#endif //SMARTPEAK_NODETENSORDATA_H