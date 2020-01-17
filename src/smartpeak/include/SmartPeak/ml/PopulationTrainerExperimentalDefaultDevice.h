/**TODO:  Add copyright*/

#ifndef SMARTPEAK_POPULATIONTRAINEREXPERIMENTALDEFAULTDEVICE_H
#define SMARTPEAK_POPULATIONTRAINEREXPERIMENTALDEFAULTDEVICE_H

// .h
#include <SmartPeak/ml/PopulationTrainerExperimental.h>
#include <SmartPeak/ml/ModelInterpreterDefaultDevice.h>

// .cpp

namespace SmartPeak
{
	/**
		@brief Class to train a vector of models
	*/
	template<typename TensorT>
	class PopulationTrainerExperimentalDefaultDevice : public PopulationTrainerExperimental<TensorT, ModelInterpreterDefaultDevice<TensorT>>
	{
	};
}
#endif //SMARTPEAK_POPULATIONTRAINEREXPERIMENTALDEFAULTDEVICE_H