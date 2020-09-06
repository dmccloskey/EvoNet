/**TODO:  Add copyright*/

#ifndef EVONET_MODELTRAINERGPU_H
#define EVONET_MODELTRAINERGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>

// .h
#include <EvoNet/ml/ModelTrainer.h>
#include <EvoNet/ml/ModelInterpreterGpu.h>

// .cpp

namespace EvoNet
{

	/**
		@brief Class to train a network model
	*/
	template<typename TensorT>
	class ModelTrainerGpu : public ModelTrainer<TensorT, ModelInterpreterGpu<TensorT>>
	{
	public:
		ModelTrainerGpu() = default; ///< Default constructor
		~ModelTrainerGpu() = default; ///< Default destructor
	};
}

#endif
#endif //EVONET_MODELTRAINERGPU_H