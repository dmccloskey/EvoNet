/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELTRAINEREXPERIMENTALDEFAULTDEVICE_H
#define SMARTPEAK_MODELTRAINEREXPERIMENTALDEFAULTDEVICE_H

// .h
#include <SmartPeak/ml/ModelTrainerExperimental.h>
#include <SmartPeak/ml/ModelInterpreterDefaultDevice.h>

// .cpp
#include <SmartPeak/io/ModelInterpreterFileDefaultDevice.h>
#include <SmartPeak/io/ModelFile.h>

namespace SmartPeak
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

#endif //SMARTPEAK_MODELTRAINEREXPERIMENTALDEFAULTDEVICE_H