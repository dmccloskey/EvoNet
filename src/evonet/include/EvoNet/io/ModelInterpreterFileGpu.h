/**TODO:  Add copyright*/

#ifndef EVONET_MODELINTERPRETERFILEGPU_H
#define EVONET_MODELINTERPRETERFILEGPU_H

// .h
#include <EvoNet/io/ModelInterpreterFile.h>

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <EvoNet/ml/ModelInterpreterGpu.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace EvoNet
{
	/**
		@brief ModelInterpreterFileGpu
	*/
	template<typename TensorT>
	class ModelInterpreterFileGpu : public ModelInterpreterFile<TensorT, ModelInterpreterGpu<TensorT>>
	{
	public:
		ModelInterpreterFileGpu() = default; ///< Default constructor
		~ModelInterpreterFileGpu() = default; ///< Default destructor
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ModelInterpreterFile<TensorT, ModelInterpreterGpu<TensorT>>>(this));
		}
	};
}
#endif
#endif //EVONET_MODELINTERPRETERFILEGPU_H