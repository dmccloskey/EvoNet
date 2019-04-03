/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELINTERPRETERFILEDEFAULTDEVICE_H
#define SMARTPEAK_MODELINTERPRETERFILEDEFAULTDEVICE_H

// .h
#include <SmartPeak/ml/ModelInterpreterDefaultDevice.h>
#include <SmartPeak/io/ModelInterpreterFile.h>

namespace SmartPeak
{
	/**
		@brief ModelInterpreterFileDefaultDevice
	*/
	template<typename TensorT>
	class ModelInterpreterFileDefaultDevice : public ModelInterpreterFile<TensorT, ModelInterpreterDefaultDevice<TensorT>>
	{
	public:
		ModelInterpreterFileDefaultDevice() = default; ///< Default constructor
		~ModelInterpreterFileDefaultDevice() = default; ///< Default destructor
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ModelInterpreterFile<TensorT, ModelInterpreterDefaultDevice<TensorT>>>(this));
		}
	};
}
#endif //SMARTPEAK_MODELINTERPRETERFILEDEFAULTDEVICE_H