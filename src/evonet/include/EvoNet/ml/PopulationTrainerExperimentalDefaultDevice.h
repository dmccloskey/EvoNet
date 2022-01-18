/**TODO:  Add copyright*/

#ifndef EVONET_POPULATIONTRAINEREXPERIMENTALDEFAULTDEVICE_H
#define EVONET_POPULATIONTRAINEREXPERIMENTALDEFAULTDEVICE_H

// .h
#include <EvoNet/ml/PopulationTrainerExperimental.h>
#include <EvoNet/ml/ModelInterpreterDefaultDevice.h>

// .cpp

namespace EvoNet
{
	/**
		@brief Class to train a vector of models
	*/
	template<typename TensorT>
	class PopulationTrainerExperimentalDefaultDevice : public PopulationTrainerExperimental<TensorT, ModelInterpreterDefaultDevice<TensorT>>
	{
	};
}
#endif //EVONET_POPULATIONTRAINEREXPERIMENTALDEFAULTDEVICE_H