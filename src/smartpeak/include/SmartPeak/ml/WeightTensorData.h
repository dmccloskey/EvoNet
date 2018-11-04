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

namespace SmartPeak
{
  /**
    @brief Network WeightMatrixData

		NOTES:
		- define the batch size and memory sizes
		- define the weight and solver param mapping
		- initialize tensors
  */
	template<typename TensorT>
  class WeightTensorData
  {
public:
    WeightTensorData() = default; ///< Default constructor
    WeightTensorData(const WeightTensorData& other)
		{
			h_weight_ = other.h_weight_;
			h_solver_params_ = other.h_solver_params_;
			h_error_ = other.h_error_;
			d_weight_ = other.d_weight_;
			d_solver_params_ = other.d_solver_params_;
			d_error_ = other.d_error_;
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
			d_weight_ = other.d_weight_;
			d_solver_params_ = other.d_solver_params_;
			d_error_ = other.d_error_;
      return *this;
    }

		void setLayer1Size(const int& layer1_size) { layer1_size_ = layer1_size; }
		void setLayer2Size(const int& layer2_size) { layer2_size_ = layer2_size; }
		void setNSolverParams(const int& n_solver_params) { n_solver_params_ = n_solver_params; }
		int getLayer1Size() const { return layer1_size_; }
		int getLayer2Size() const	{ return layer2_size_; }
		int getNSolverParams() const { return n_solver_params_; }

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

		size_t getTensorSize() { return layer1_size_ * layer2_size_ * sizeof(TensorT); }; ///< Get the size of each tensor in bytes
		size_t getSolverParamsSize() { return layer1_size_ * layer2_size_ * n_solver_params_ * sizeof(TensorT); }; ///< Get the size of each tensor in bytes

		void initWeightTensorData(const int& layer1_size, const int&layer2_size, const std::vector<std::pair<int, int>>& weight_indices, const std::vector<TensorT>& weight_values, const bool& train, std::vector<TensorT>& solver_params);

protected:
		int layer1_size_ = 1; ///< Layer1 size
		int layer2_size_ = 2; ///< Layer2 size
		int n_solver_params_ = 1; ///< The number of solver params

		// [TODO: move to weight]
		std::map<std::string, int> solver_params_indices_; ///< Map from solver params to weight matrix indices
		std::map<std::string, int> weight_indices_;  ///< Map from weights to weight matrix indices

    /**
      @brief weight and error have the following dimensions:
        rows: # of layer1, cols: # of layer2
				while solver_params have the following dimensions:

    */		
		std::shared_ptr<TensorT> h_weight_ = nullptr;
		std::shared_ptr<TensorT> h_solver_params_ = nullptr;
		std::shared_ptr<TensorT> h_error_ = nullptr;
		std::shared_ptr<TensorT> d_weight_ = nullptr;
		std::shared_ptr<TensorT> d_solver_params_ = nullptr;
		std::shared_ptr<TensorT> d_error_ = nullptr;
		// [TODO: add drop probability]
  };

	template<typename TensorT>
	inline void WeightTensorData<TensorT>::initWeightTensorData(const int & layer1_size, const int & layer2_size, const std::vector<std::pair<int, int>>& weight_indices, const std::vector<TensorT>& weight_values, const bool & train, std::vector<TensorT>& solver_params)
	{
		assert(weight_indices.size() == weight_values.size());
		setLayer1Size(layer1_size);
		setLayer2Size(layer2_size);

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
		Eigen::Tensor<TensorT, 3> params(layer1_size, layer2_size, (int)solver_params.size());
		for (int i = 0; i < solver_params.size(); ++i) {
			params.chip(i, 2).setConstant(solver_params[i]);
		}
		setSolverParams(params);
	}

	template<typename TensorT>
	class WeightTensorDataCpu : public WeightTensorData<TensorT> {
	public:
		void setWeight(const Eigen::Tensor<TensorT, 2>& weight) {
			TensorT* h_weight = new TensorT[this->layer1_size_*this->layer2_size_];
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> weight_copy(h_weight, this->layer1_size_, this->layer2_size_);
			weight_copy = weight;
			this->h_weight_.reset(std::move(h_weight));
		}; ///< weight setter
		void setSolverParams(const Eigen::Tensor<TensorT, 3>& solver_params) {
			TensorT* h_solver_params = new TensorT[this->layer1_size_*this->layer2_size_*this->n_solver_params_];
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 3>> solver_params_copy(h_solver_params, this->layer1_size_, this->layer2_size_, this->n_solver_params_);
			solver_params_copy = solver_params;
			this->h_solver_params_.reset(h_solver_params);
		}; ///< solver_params setter
		void setError(const Eigen::Tensor<TensorT, 2>& error) {
			TensorT* h_error = new TensorT[this->layer1_size_*this->layer2_size_];
			// copy the tensor
			Eigen::TensorMap<Eigen::Tensor<TensorT, 2>> error_copy(h_error, this->layer1_size_, this->layer2_size_);
			error_copy = error;
			this->h_error_.reset(h_error);
		}; ///< error setter
	};

#if COMPILE_WITH_CUDA

	template<typename TensorT>
	class WeightTensorDataGpu : public WeightTensorData<TensorT> {
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
		}; ///< error setter
	};
#endif
}

#endif //SMARTPEAK_WEIGHTTENSORDATA_H