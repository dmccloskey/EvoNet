/**TODO:  Add copyright*/

#ifndef SMARTPEAK_POPULATIONTRAINERGPU_H
#define SMARTPEAK_POPULATIONTRAINERGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>

// .h
#include <SmartPeak/ml/PopulationTrainer.h>
#include <SmartPeak/ml/ModelInterpreterGpu.h>

// .cpp

namespace SmartPeak
{
	/**
		@brief Class to train a vector of models
	*/
	template<typename TensorT>
	class PopulationTrainerGpu : public PopulationTrainer<TensorT, ModelInterpreterGpu<TensorT>>
	{
	};
}
#endif
#endif //SMARTPEAK_POPULATIONTRAINERGPU_H