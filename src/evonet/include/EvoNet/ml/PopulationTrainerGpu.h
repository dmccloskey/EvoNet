/**TODO:  Add copyright*/

#ifndef EVONET_POPULATIONTRAINERGPU_H
#define EVONET_POPULATIONTRAINERGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>

// .h
#include <EvoNet/ml/PopulationTrainer.h>
#include <EvoNet/ml/ModelInterpreterGpu.h>

// .cpp

namespace EvoNet
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
#endif //EVONET_POPULATIONTRAINERGPU_H