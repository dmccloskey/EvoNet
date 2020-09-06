/**TODO:  Add copyright*/

#ifndef EVONET_POPULATIONTRAINEREXPERIMENTALGPU_H
#define EVONET_POPULATIONTRAINEREXPERIMENTALGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>

// .h
#include <EvoNet/ml/PopulationTrainerExperimental.h>
#include <EvoNet/ml/ModelInterpreterGpu.h>

// .cpp

namespace EvoNet
{
	/**
		@brief Class to train a vector of models
	*/
	template<typename TensorT>
	class PopulationTrainerExperimentalGpu : public PopulationTrainerExperimental<TensorT, ModelInterpreterGpu<TensorT>>
	{
	};
}
#endif
#endif //EVONET_POPULATIONTRAINEREXPERIMENTALGPU_H