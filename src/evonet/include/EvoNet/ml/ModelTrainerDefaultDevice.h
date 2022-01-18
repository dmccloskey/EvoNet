/**TODO:  Add copyright*/

#ifndef EVONET_MODELTRAINERDEFAULTDEVICE_H
#define EVONET_MODELTRAINERDEFAULTDEVICE_H

// .h
#include <EvoNet/ml/ModelTrainer.h>
#include <EvoNet/ml/ModelInterpreterDefaultDevice.h>

// .cpp

namespace EvoNet
{

	/**
		@brief Class to train a network model
	*/
	template<typename TensorT>
	class ModelTrainerDefaultDevice : public ModelTrainer<TensorT, ModelInterpreterDefaultDevice<TensorT>>
	{
	public:
		ModelTrainerDefaultDevice() = default; ///< Default constructor
		~ModelTrainerDefaultDevice() = default; ///< Default destructor
	};
}

#endif //EVONET_MODELTRAINERDEFAULTDEVICE_H