/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELTRAINERGPU_H
#define SMARTPEAK_MODELTRAINERGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>

// .h
#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/ml/ModelInterpreterGpu.h>

// .cpp

namespace SmartPeak
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
#endif //SMARTPEAK_MODELTRAINERGPU_H