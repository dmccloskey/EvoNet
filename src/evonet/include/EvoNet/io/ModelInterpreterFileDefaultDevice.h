/**TODO:  Add copyright*/

#ifndef EVONET_MODELINTERPRETERFILEDEFAULTDEVICE_H
#define EVONET_MODELINTERPRETERFILEDEFAULTDEVICE_H

// .h
#include <EvoNet/ml/ModelInterpreterDefaultDevice.h>
#include <EvoNet/io/ModelInterpreterFile.h>

namespace EvoNet
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
#endif //EVONET_MODELINTERPRETERFILEDEFAULTDEVICE_H