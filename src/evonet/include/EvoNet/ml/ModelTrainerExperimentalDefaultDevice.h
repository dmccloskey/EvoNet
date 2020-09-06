/**TODO:  Add copyright*/

#ifndef EVONET_MODELTRAINEREXPERIMENTALDEFAULTDEVICE_H
#define EVONET_MODELTRAINEREXPERIMENTALDEFAULTDEVICE_H

// .h
#include <EvoNet/ml/ModelTrainerExperimental.h>
#include <EvoNet/ml/ModelInterpreterDefaultDevice.h>

// .cpp
#include <EvoNet/io/ModelInterpreterFileDefaultDevice.h>
#include <EvoNet/io/ModelFile.h>

namespace EvoNet
{

	/**
		@brief Class to train a network model
	*/
	template<typename TensorT>
	class ModelTrainerExperimentalDefaultDevice : public ModelTrainerExperimental<TensorT, ModelInterpreterDefaultDevice<TensorT>>
	{
	public:
		ModelTrainerExperimentalDefaultDevice() = default; ///< Default constructor
		~ModelTrainerExperimentalDefaultDevice() = default; ///< Default destructor    
                                                        
    /// Overrides used in all examples
    void adaptiveTrainerScheduler(const int& n_generations, const int& n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, const std::vector<TensorT>& model_errors) override;
	};
  template<typename TensorT>
  inline void ModelTrainerExperimentalDefaultDevice<TensorT>::adaptiveTrainerScheduler(const int& n_generations, const int& n_epochs, Model<TensorT>& model, ModelInterpreterDefaultDevice<TensorT>& model_interpreter, const std::vector<TensorT>& model_errors)
  {
    //if (n_epochs == 0) {
    //  ModelInterpreterFileDefaultDevice<TensorT> interpreter_data;
    //  interpreter_data.storeModelInterpreterCsv(model.getName() + "_interpreter.csv", model_interpreter);
    //}
    if (n_epochs % 1000 == 0 && n_epochs != 0) {
      // save the model every 1000 epochs
      model_interpreter.getModelResults(model, false, true, false, false);
      ModelFile<TensorT> data;
      data.storeModelBinary(model.getName() + "_" + std::to_string(n_epochs) + "_model.binary", model);
      ModelInterpreterFileDefaultDevice<TensorT> interpreter_data;
      interpreter_data.storeModelInterpreterBinary(model.getName() + "_" + std::to_string(n_epochs) + "_interpreter.binary", model_interpreter);
    }
  }
}

#endif //EVONET_MODELTRAINEREXPERIMENTALDEFAULTDEVICE_H