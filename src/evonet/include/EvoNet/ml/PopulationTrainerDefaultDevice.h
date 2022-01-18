/**TODO:  Add copyright*/

#ifndef EVONET_POPULATIONTRAINERDEFAULTDEVICE_H
#define EVONET_POPULATIONTRAINERDEFAULTDEVICE_H

// .h
#include <EvoNet/ml/PopulationTrainer.h>
#include <EvoNet/ml/ModelInterpreterDefaultDevice.h>

// .cpp

namespace EvoNet
{
	/**
		@brief Class to train a vector of models
	*/
	template<typename TensorT>
	class PopulationTrainerDefaultDevice : public PopulationTrainer<TensorT, ModelInterpreterDefaultDevice<TensorT>>
	{
	};
}
#endif //EVONET_POPULATIONTRAINERDEFAULTDEVICE_H