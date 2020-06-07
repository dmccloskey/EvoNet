/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELTRAINEREXPERIMENTALGPU_H
#define SMARTPEAK_MODELTRAINEREXPERIMENTALGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>

// .h
#include <SmartPeak/ml/ModelTrainerExperimental.h>
#include <SmartPeak/ml/ModelInterpreterGpu.h>

// .cpp

namespace SmartPeak
{

	/**
		@brief Class to train a network model
	*/
	template<typename TensorT>
	class ModelTrainerExperimentalGpu : public ModelTrainerExperimental<TensorT, ModelInterpreterGpu<TensorT>>
	{
	public:
		ModelTrainerExperimentalGpu() = default; ///< Default constructor
		~ModelTrainerExperimentalGpu() = default; ///< Default destructor
	};
}

#endif
#endif //SMARTPEAK_MODELTRAINEREXPERIMENTALGPU_H