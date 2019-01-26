/**TODO:  Add copyright*/

#ifndef SMARTPEAK_WEIGHTTENSORDATA_H
#define SMARTPEAK_WEIGHTTENSORDATA_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <unsupported/Eigen/CXX11/Tensor>

//#include <cereal/access.hpp>  // serialiation of private members
//#include <cereal/types/memory.hpp>
//#undef min // clashes with std::limit on windows in polymorphic.hpp
//#undef max // clashes with std::limit on windows in polymorphic.hpp
//#include <cereal/types/polymorphic.hpp>

namespace SmartPeak
{
  /**
    @brief Network WeightMatrixData

		NOTES:
		- define the batch size and memory sizes
		- define the weight and solver param mapping
		- initialize tensors
  */
	template<typename TensorT, typename DeviceT>
  class WeightTensorData
  {
public:
    WeightTensorData() = default; ///< Default constructor
    WeightTensorData(const WeightTensorData& other)
		{
			h_weight_ = other.h_weight_;
			h_solver_params_ = other.h_solver_params_;
			h_error_ = other.h_error_;
			h_shared_weights_ = other.h_shared_weights_;
			d_weight_ = other.d_weight_;
			d_solver_params_ = other.d_solver_params_;
			d_error_ = other.d_error_;
			d_shared_weights_ = other.d_shared_weights_;
			layer1_size_ = other.layer1_size_;
			layer2_size_ = other.layer2_size_;
			n_solver_params_ = other.n_solver_params_;
			n_shared_weights_ = other.n_shared_weights_;
		};
    ~WeightTensorData() = default; ///< Default destructor

    inline bool operator==(const WeightTensorData& other) const
    {
      return
        std::tie(
       
        ) == std::tie(

        )
      ;
    }

    inline bool operator!=(const WeightTensorData& other) const
    {
      return !(*this == other);
    }

    inline WeightTensorData& operator=(const WeightTensorData& other)
    {
			h_weight_ = other.h_weight_;
			h_solver_params_ = other.h_solver_params_;
			h_error_ = other.h_error_;
			h_shared_weights_ = other.h_shared_weights_;
			d_weight_ = other.d_weight_;
			d_solver_params_ = other.d_solver_params_;
			d_error_ = other.d_error_;
			d_shared_weights_ = other.d_shared_weights_;
			layer1_size_ = other.layer1_size_;
			layer2_size_ = other.layer2_size_;
			n_solver_params_ = other.n_solver_params_;
			n_shared_weights_ = other.n_shared_weights_;
      return *this;
    }

		void setLayer1Size(const int& layer1_size) { layer1_size_ = layer1_size; }
		void setLayer2Size(const int& layer2_size) { layer2_size_ = layer2_size; }
		void setNSolverParams(const int& n_solver_params) { n_solver_params_ = n_solver_params; }
		void setNSharedWeights(const int& n_shared_weights) { n_shared_weights_ = n_shared_weights; }
		int getLayer1Size() const { return layer1_size_; }
		int getLayer2Size() const	{ return layer2_size_; }
		int getNSolverParams() const { return n_solver_params_; }
		int getNSharedWeights() const { return n_shared_weights_; }

		virtual void setWeight(const Eigen::Tensor<TensorT, 2>& weight) = 0; ///< weight setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> getWeight() { std::shared_ptr<TensorT> h_weight = h_weight_; Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight(h_weight.get(), layer1_size_, layer2_size_); return weight; }; ///< weight copy getter
		std::shared_ptr<TensorT> getHWeightPointer() { return h_weight_; }; ///< weight pointer getter
		std::shared_ptr<TensorT> getDWeightPointer() { return d_weight_; }; ///< weight pointer getter

    virtual void setSolverParams(const Eigen::Tensor<TensorT, 3>& solver_params) = 0; ///< solver_params setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> getSolverParams() { Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> solver_params(h_solver_params_.get(), layer1_size_, layer2_size_, n_solver_params_); return solver_params; }; ///< solver_params copy getter
		std::shared_ptr<TensorT> getHSolverParamsPointer() { return h_solver_params_; }; ///< solver_params pointer getter
		std::shared_ptr<TensorT> getDSolverParamsPointer() { return d_solver_params_; }; ///< solver_params pointer getter

    virtual void setError(const Eigen::Tensor<TensorT, 2>& error) = 0; ///< error setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> getError() { Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error(h_error_.get(), layer1_size_, layer2_size_); return error; }; ///< error copy getter
		std::shared_ptr<TensorT> getHErrorPointer() { return h_error_; }; ///< error pointer getter
		std::shared_ptr<TensorT> getDErrorPointer() { return d_error_; }; ///< error pointer getter

		virtual void setSharedWeights(const Eigen::Tensor<TensorT, 3>& shared_weights) = 0; ///< shared_weights setter
		Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> getSharedWeights() { Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> shared_weights(h_shared_weights_.get(), layer1_size_, layer2_size_, n_shared_weights_); return shared_weights; }; ///< shared_weights copy getter
		std::shared_ptr<TensorT> getHSharedWeightsPointer() { return h_shared_weights_; }; ///< shared_weights pointer getter
		std::shared_ptr<TensorT> getDSharedWeightsPointer() { return d_shared_weights_; }; ///< shared_weights pointer getter

		int getTensorSize() { return layer1_size_ * layer2_size_ * sizeof(TensorT); }; ///< Get the size of each tensor in bytes
		int getSolverParamsSize() { return layer1_size_ * layer2_size_ * n_solver_params_ * sizeof(TensorT); }; ///< Get the size of each tensor in bytes
		int getSharedWeightsSize() { return layer1_size_ * layer2_size_ * n_shared_weights_ * sizeof(TensorT); }; ///< Get the size of each tensor in bytes

		void initWeightTensorData(const int& layer1_size, const int&layer2_size, const std::vector<std::pair<int, int>>& weight_indices, 
			const std::map<std::string, std::vector<std::pair<int, int>>>& shared_weight_indices, const std::vector<TensorT>& weight_values, const bool& train, std::vector<TensorT>& solver_params);

		virtual bool syncHAndDError(DeviceT& device) = 0;
		virtual bool syncHAndDWeight(DeviceT& device) = 0;
		virtual bool syncHAndDSolverParams(DeviceT& device) = 0;
		virtual bool syncHAndDSharedWeights(DeviceT& device) = 0;

		std::pair<bool, bool> getErrorStatus() const { return std::make_pair(h_error_updated_, d_error_updated_); };
		void setErrorStatus(const bool& h_status, const bool& d_status) { h_error_updated_ = h_status; d_error_updated_ = d_status; };
		std::pair<bool, bool> getWeightStatus() const { return std::make_pair(h_weight_updated_, d_weight_updated_); };
		void setWeightStatus(const bool& h_status, const bool& d_status) { h_weight_updated_ = h_status; d_weight_updated_ = d_status; };
		std::pair<bool, bool> getSolverParamsStatus() const { return std::make_pair(h_solver_params_updated_, d_solver_params_updated_); };
		void setSolverParamsStatus(const bool& h_status, const bool& d_status) { h_solver_params_updated_ = h_status; d_solver_params_updated_ = d_status; };
		std::pair<bool, bool> getSharedWeightsStatus() const { return std::make_pair(h_shared_weights_updated_, d_shared_weights_updated_); };
		void setSharedWeightsStatus(const bool& h_status, const bool& d_status) { h_shared_weights_updated_ = h_status; d_shared_weights_updated_ = d_status; };
protected:
		int layer1_size_ = 1; ///< Layer1 size
		int layer2_size_ = 2; ///< Layer2 size
		int n_solver_params_ = 0; ///< The number of solver params
		int n_shared_weights_ = 0; ///< The number of shared weights in the layer

    /**
      @brief weight and error have the following dimensions:
        rows: # of layer1, cols: # of layer2
				while solver_params have the following dimensions:

    */		
		std::shared_ptr<TensorT> h_weight_;
		std::shared_ptr<TensorT> h_solver_params_;
		std::shared_ptr<TensorT> h_error_;
		std::shared_ptr<TensorT> h_shared_weights_;
		std::shared_ptr<TensorT> d_weight_;
		std::shared_ptr<TensorT> d_solver_params_;
		std::shared_ptr<TensorT> d_error_;
		std::shared_ptr<TensorT> d_shared_weights_;
		// [TODO: add drop probability]

		bool h_error_updated_ = false;
		bool h_weight_updated_ = false;
		bool h_solver_params_updated_ = false;
		bool h_shared_weights_updated_ = false;

		bool d_error_updated_ = false;
		bool d_weight_updated_ = false;
		bool d_solver_params_updated_ = false;
		bool d_shared_weights_updated_ = false;

	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(layer1_size_, layer2_size_, n_solver_params_, n_shared_weights_,
	//			h_weight_, h_solver_params_, h_error_, h_shared_weights_,
	//			d_weight_, d_solver_params_, d_error_, d_shared_weights_,
	//			h_error_updated_, h_weight_updated_, h_solver_params_updated_, h_shared_weights_updated_,
	//			d_error_updated_, d_weight_updated_, d_solver_params_updated_, d_shared_weights_updated_);
	//	}
  };

	template<typename TensorT, typename DeviceT>
	inline void WeightTensorData<TensorT, DeviceT>::initWeightTensorData(const int & layer1_size, const int & layer2_size, const std::vector<std::pair<int, int>>& weight_indices, 
		const std::map<std::string, std::vector<std::pair<int, int>>>& shared_weight_indices, const std::vector<TensorT>& weight_values, const bool & train, std::vector<TensorT>& solver_params)
	{
		assert(weight_indices.size() == weight_values.size());
		setLayer1Size(layer1_size);
		setLayer2Size(layer2_size);
		// TODO: implement checks to ensure Tensors are not too large
		// results in a std::bad_array_new_length

		// make the weight and error tensors
		Eigen::Tensor<TensorT, 2> zero(layer1_size, layer2_size); zero.setZero();
		Eigen::Tensor<TensorT, 2> weights(layer1_size, layer2_size); weights.setZero();
		for (size_t i = 0; i < weight_indices.size(); ++i) {
			weights(weight_indices[i].first, weight_indices[i].second) = weight_values[i];
		}
		setWeight(weights);
		setError(zero);

		// make the parameters
		setNSolverParams(solver_params.size());
		if (solver_params.size() > 0) {
			Eigen::Tensor<TensorT, 3> params(layer1_size, layer2_size, (int)solver_params.size());
			for (int i = 0; i < solver_params.size(); ++i) {
				params.chip(i, 2).setConstant(solver_params[i]);
			}
			setSolverParams(params);
		}

		// make the shared weighs tensor
		setNSharedWeights(shared_weight_indices.size());
		if (shared_weight_indices.size() > 0) {
			Eigen::Tensor<TensorT, 3> shared(layer1_size, layer2_size, (int)shared_weight_indices.size());
			shared.setZero();
			int iter = 0;
			for (const auto& weight_indices_map : shared_weight_indices) {
				for (const std::pair<int, int>& weight_index : weight_indices_map.second) {
					Eigen::array<int, 3> offsets = { weight_index.first, weight_index.second, iter };
					Eigen::array<int, 3> extents = { 1, 1, 1 };
					Eigen::Tensor<TensorT, 3> ones(1, 1, 1);
					ones.setConstant(1);
					shared.slice(offsets, extents) = ones;
				}
				++iter;
			}
			setSharedWeights(shared);
		}
	}

	template<typename TensorT>
	class WeightTensorDataCpu : public WeightTensorData<TensorT, Eigen::DefaultDevice> {
	public:
		void setWeight(const Eigen::Tensor<TensorT, 2>& weight) {
			TensorT* h_weight = new TensorT[this->layer1_size_*this->layer2_size_];
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight_copy(h_weight, this->layer1_size_, this->layer2_size_);
			weight_copy = weight;
			auto h_deleter = [&](TensorT* ptr) { delete[] ptr; };
			this->h_weight_.reset(h_weight, h_deleter);
			this->h_weight_updated_ = true;
			this->d_weight_updated_ = true;
		}; ///< weight setter
		void setSolverParams(const Eigen::Tensor<TensorT, 3>& solver_params) {
			TensorT* h_solver_params = new TensorT[this->layer1_size_*this->layer2_size_*this->n_solver_params_];
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> solver_params_copy(h_solver_params, this->layer1_size_, this->layer2_size_, this->n_solver_params_);
			solver_params_copy = solver_params;
			auto h_deleter = [&](TensorT* ptr) { delete[] ptr; };
			this->h_solver_params_.reset(h_solver_params, h_deleter);
			this->h_solver_params_updated_ = true;
			this->d_solver_params_updated_ = true;
		}; ///< solver_params setter
		void setError(const Eigen::Tensor<TensorT, 2>& error) {
			TensorT* h_error = new TensorT[this->layer1_size_*this->layer2_size_];
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_copy(h_error, this->layer1_size_, this->layer2_size_);
			error_copy = error;
			auto h_deleter = [&](TensorT* ptr) { delete[] ptr; };
			this->h_error_.reset(h_error, h_deleter);
			this->h_error_updated_ = true;
			this->d_error_updated_ = true;
		}; ///< error setter
		void setSharedWeights(const Eigen::Tensor<TensorT, 3>& shared_weights) {
			TensorT* h_shared_weights = new TensorT[this->layer1_size_*this->layer2_size_*this->n_shared_weights_];
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> shared_weights_copy(h_shared_weights, this->layer1_size_, this->layer2_size_, this->n_shared_weights_);
			shared_weights_copy = shared_weights;
			auto h_deleter = [&](TensorT* ptr) { delete[] ptr; };
			this->h_shared_weights_.reset(h_shared_weights, h_deleter);
			this->h_shared_weights_updated_ = true;
			this->d_shared_weights_updated_ = true;
		}; ///< shared_weights setter
		bool syncHAndDError(Eigen::DefaultDevice& device) { return true; }
		bool syncHAndDWeight(Eigen::DefaultDevice& device) { return true; }
		bool syncHAndDSolverParams(Eigen::DefaultDevice& device) { return true; }
		bool syncHAndDSharedWeights(Eigen::DefaultDevice& device) { return true; }
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<WeightTensorData<TensorT, Eigen::DefaultDevice>>(this));
	//	}
	};

#if COMPILE_WITH_CUDA

	template<typename TensorT>
	class WeightTensorDataGpu : public WeightTensorData<TensorT, Eigen::GpuDevice> {
	public:
		void setWeight(const Eigen::Tensor<TensorT, 2>& weight) {
			// allocate cuda and pinned host layer2
			TensorT* d_weight;
			TensorT* h_weight;
			assert(cudaMalloc((void**)(&d_weight), getTensorSize()) == cudaSuccess);
			assert(cudaHostAlloc((void**)(&h_weight), getTensorSize(), cudaHostAllocDefault) == cudaSuccess);
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight_copy(h_weight, this->layer1_size_, this->layer2_size_);
			weight_copy = weight;
			// define the deleters
			auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
			auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
			this->h_weight_.reset(h_weight, h_deleter); 
			this->d_weight_.reset(d_weight, d_deleter);
			this->h_weight_updated_ = true;
			this->d_weight_updated_ = false;
		}; ///< weight setter
		void setSolverParams(const Eigen::Tensor<TensorT, 3>& solver_params) {
			// allocate cuda and pinned host layer2
			TensorT* d_solver_params;
			TensorT* h_solver_params;
			assert(cudaMalloc((void**)(&d_solver_params), getSolverParamsSize()) == cudaSuccess);
			assert(cudaHostAlloc((void**)(&h_solver_params), getSolverParamsSize(), cudaHostAllocDefault) == cudaSuccess);
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> solver_params_copy(h_solver_params, this->layer1_size_, this->layer2_size_, this->n_solver_params_);
			solver_params_copy = solver_params;
			// define the deleters
			auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
			auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
			this->h_solver_params_.reset(h_solver_params, h_deleter);
			this->d_solver_params_.reset(d_solver_params, d_deleter);
			this->h_solver_params_updated_ = true;
			this->d_solver_params_updated_ = false;
		}; ///< solver_params setter
		void setError(const Eigen::Tensor<TensorT, 2>& error) {
			// allocate cuda and pinned host layer2
			TensorT* d_error;
			TensorT* h_error;
			assert(cudaMalloc((void**)(&d_error), getTensorSize()) == cudaSuccess);
			assert(cudaHostAlloc((void**)(&h_error), getTensorSize(), cudaHostAllocDefault) == cudaSuccess);
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_copy(h_error, this->layer1_size_, this->layer2_size_);
			error_copy = error;
			// define the deleters
			auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
			auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
			this->h_error_.reset(h_error, h_deleter);
			this->d_error_.reset(d_error, d_deleter);
			this->h_error_updated_ = true;
			this->d_error_updated_ = false;
		}; ///< error setter
		void setSharedWeights(const Eigen::Tensor<TensorT, 3>& shared_weights) {
			// allocate cuda and pinned host layer2
			TensorT* d_shared_weights;
			TensorT* h_shared_weights;
			assert(cudaMalloc((void**)(&d_shared_weights), getSharedWeightsSize()) == cudaSuccess);
			assert(cudaHostAlloc((void**)(&h_shared_weights), getSharedWeightsSize(), cudaHostAllocDefault) == cudaSuccess);
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> shared_weights_copy(h_shared_weights, this->layer1_size_, this->layer2_size_, this->n_shared_weights_);
			shared_weights_copy = shared_weights;
			// define the deleters
			auto h_deleter = [&](TensorT* ptr) { cudaFreeHost(ptr); };
			auto d_deleter = [&](TensorT* ptr) { cudaFree(ptr); };
			this->h_shared_weights_.reset(h_shared_weights, h_deleter);
			this->d_shared_weights_.reset(d_shared_weights, d_deleter);
			this->h_shared_weights_updated_ = true;
			this->d_shared_weights_updated_ = false;
		}; ///< shared_weights setter
		bool syncHAndDError(Eigen::GpuDevice& device) {
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
				//std::cout << "Both host and device are syncHAndDronized." << std::endl;
				return false;
			}
		}
		bool syncHAndDWeight(Eigen::GpuDevice& device) {
			if (this->h_weight_updated_ && !this->d_weight_updated_) {
				device.memcpyHostToDevice(this->d_weight_.get(), this->h_weight_.get(), getTensorSize());
				this->d_weight_updated_ = true;
				this->h_weight_updated_ = false;
				return true;
			}
			else if (!this->h_weight_updated_ && this->d_weight_updated_) {
				device.memcpyDeviceToHost(this->h_weight_.get(), this->d_weight_.get(), getTensorSize());
				this->h_weight_updated_ = true;
				this->d_weight_updated_ = false;
				return true;
			}
			else {
				//std::cout << "Both host and device are syncHAndDronized." << std::endl;
				return false;
			}
			return true;
		}
		bool syncHAndDSolverParams(Eigen::GpuDevice& device) {
			if (this->h_solver_params_updated_ && !this->d_solver_params_updated_) {
				device.memcpyHostToDevice(this->d_solver_params_.get(), this->h_solver_params_.get(), getSolverParamsSize());
				this->d_solver_params_updated_ = true;
				this->h_solver_params_updated_ = false;
				return true;
			}
			else if (!this->h_solver_params_updated_ && this->d_solver_params_updated_) {
				device.memcpyDeviceToHost(this->h_solver_params_.get(), this->d_solver_params_.get(), getSolverParamsSize());
				this->h_solver_params_updated_ = true;
				this->d_solver_params_updated_ = false;
				return true;
			}
			else {
				//std::cout << "Both host and device are syncHAndDronized." << std::endl;
				return false;
			}
			return true;
		}
		bool syncHAndDSharedWeights(Eigen::GpuDevice& device) {
			if (this->h_shared_weights_updated_ && !this->d_shared_weights_updated_) {
				device.memcpyHostToDevice(this->d_shared_weights_.get(), this->h_shared_weights_.get(), getSharedWeightsSize());
				this->d_shared_weights_updated_ = true;
				this->h_shared_weights_updated_ = false;
				return true;
			}
			else if (!this->h_shared_weights_updated_ && this->d_shared_weights_updated_) {
				device.memcpyDeviceToHost(this->h_shared_weights_.get(), this->d_shared_weights_.get(), getSharedWeightsSize());
				this->h_shared_weights_updated_ = true;
				this->d_shared_weights_updated_ = false;
				return true;
			}
			else {
				//std::cout << "Both host and device are syncHAndDronized." << std::endl;
				return false;
			}
			return true;
		}
	//private:
	//	friend class cereal::access;
	//	template<class Archive>
	//	void serialize(Archive& archive) {
	//		archive(cereal::base_class<WeightTensorData<TensorT, Eigen::GpuDevice>>(this));
	//	}
	};
#endif
}

//CEREAL_REGISTER_TYPE(SmartPeak::WeightTensorDataCpu<float>);
//// TODO: add double, int, etc.
//#if COMPILE_WITH_CUDA
//CEREAL_REGISTER_TYPE(SmartPeak::WeightTensorDataGpu<float>);
//// TODO: add double, int, etc.
//#endif

#endif //SMARTPEAK_WEIGHTTENSORDATA_H