/**TODO:  Add copyright*/

#ifndef EVONET_MODELTRAINEREXPERIMENTALGPU_H
#define EVONET_MODELTRAINEREXPERIMENTALGPU_H

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>

// .h
#include <EvoNet/ml/ModelTrainerExperimental.h>
#include <EvoNet/ml/ModelInterpreterGpu.h>

// .cpp
#include <EvoNet/io/ModelInterpreterFileGpu.h>
#include <EvoNet/io/ModelFile.h>

namespace EvoNet
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
    
    /// Overrides used in all examples
    void adaptiveTrainerScheduler(const int& n_generations, const int& n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, const std::vector<TensorT>& model_errors) override;
	};
  template<typename TensorT>
  inline void ModelTrainerExperimentalGpu<TensorT>::adaptiveTrainerScheduler(const int& n_generations, const int& n_epochs, Model<TensorT>& model, ModelInterpreterGpu<TensorT>& model_interpreter, const std::vector<TensorT>& model_errors)
  {
    if (n_epochs % 1000 == 0 && n_epochs != 0) {
      // save the model every 1000 epochs
      model_interpreter.getModelResults(model, false, true, false, false);
      ModelFile<TensorT> data;
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileGpu<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    }
  }
}
#endif
#endif //EVONET_MODELTRAINEREXPERIMENTALGPU_H