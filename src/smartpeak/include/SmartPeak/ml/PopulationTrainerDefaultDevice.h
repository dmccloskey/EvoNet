/**TODO:  Add copyright*/

#ifndef SMARTPEAK_POPULATIONTRAINERDEFAULTDEVICE_H
#define SMARTPEAK_POPULATIONTRAINERDEFAULTDEVICE_H

// .h
#include <SmartPeak/ml/PopulationTrainer.h>
#include <SmartPeak/ml/ModelInterpreterDefaultDevice.h>

// .cpp

namespace SmartPeak
{
	/**
		@brief Class to train a vector of models
	*/
	template<typename TensorT>
	class PopulationTrainerDefaultDevice : public PopulationTrainer<TensorT, ModelInterpreterDefaultDevice<TensorT>>
	{
	};
}
#endif //SMARTPEAK_POPULATIONTRAINERDEFAULTDEVICE_H