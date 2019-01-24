/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELTRAINERDEFAULTDEVICE_H
#define SMARTPEAK_MODELTRAINERDEFAULTDEVICE_H

// .h
#include <SmartPeak/ml/ModelTrainer.h>
#include <SmartPeak/ml/ModelInterpreterDefaultDevice.h>

// .cpp

namespace SmartPeak
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

#endif //SMARTPEAK_MODELTRAINERDEFAULTDEVICE_H