/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELTRAINEREXPERIMENTALDEFAULTDEVICE_H
#define SMARTPEAK_MODELTRAINEREXPERIMENTALDEFAULTDEVICE_H

// .h
#include <SmartPeak/ml/ModelTrainerExperimental.h>
#include <SmartPeak/ml/ModelInterpreterDefaultDevice.h>

// .cpp

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
	};
}

#endif //SMARTPEAK_MODELTRAINEREXPERIMENTALDEFAULTDEVICE_H