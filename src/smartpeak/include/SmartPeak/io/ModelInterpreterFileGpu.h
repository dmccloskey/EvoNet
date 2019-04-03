/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELINTERPRETERFILEGPU_H
#define SMARTPEAK_MODELINTERPRETERFILEGPU_H

// .h
#include <SmartPeak/io/ModelInterpreterFile.h>

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <SmartPeak/ml/ModelInterpreterGpu.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace SmartPeak
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
#endif //SMARTPEAK_MODELINTERPRETERFILEGPU_H