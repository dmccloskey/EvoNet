/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELINTERPRETERFILE_H
#define SMARTPEAK_MODELINTERPRETERFILE_H

// .h
#include <SmartPeak/ml/ModelInterpreter.h>

// .cpp
//#include <cereal/types/memory.hpp>
//#include <cereal/types/map.hpp>
//#include <cereal/types/tuple.hpp>
//#include <cereal/types/utility.hpp> // std::pair
#include <cereal/types/vector.hpp>
#include <cereal/archives/binary.hpp>

namespace SmartPeak
{
  /**
    @brief ModelInterpreterFile
  */
	template<typename TensorT, typename DeviceT>
  class ModelInterpreterFile
  {
public:
    ModelInterpreterFile() = default; ///< Default constructor
    ~ModelInterpreterFile() = default; ///< Default destructor
 
		/**
			@brief store ModelInterpreter from file
			
			@param filename The name of the model_interpreter file
			@param model_interpreter The model_interpreter to store

			@returns Status True on success, False if not
		*/
		bool storeModelInterpreterBinary(const std::string& filename, const ModelInterpreter<TensorT, DeviceT>& model_interpreter);
 
		/**
			@brief load Model from file

			@param filename The name of the model_interpreter file
			@param model_interpreter The model_interpreter to load data into

			@returns Status True on success, False if not
		*/
		bool loadModelInterpreterBinary(const std::string& filename, ModelInterpreter<TensorT, DeviceT>& model_interpreter);
  };

	template<typename TensorT, typename DeviceT>
	bool ModelInterpreterFile<TensorT, DeviceT>::storeModelInterpreterBinary(const std::string & filename, const  ModelInterpreter<TensorT, DeviceT>& model_interpreter)
	{
		//auto myfile = std::fstream(filename, std::ios::out | std::ios::binary);
		//myfile.write((char*)&model_interpreter, sizeof(model_interpreter));
		//myfile.close();

		std::ofstream ofs(filename, std::ios::binary);  
		//if (ofs.is_open() == false) {// Lines check to make sure the file is not already created
		cereal::BinaryOutputArchive oarchive(ofs); 
		oarchive(model_interpreter); 
		ofs.close();
		//}// Lines check to make sure the file is not already created
		return true;
	}

	template<typename TensorT, typename DeviceT>
	bool ModelInterpreterFile<TensorT, DeviceT>::loadModelInterpreterBinary(const std::string & filename,  ModelInterpreter<TensorT, DeviceT>& model_interpreter)
	{		
		std::ifstream ifs(filename, std::ios::binary); 
		if (ifs.is_open()) {
			cereal::BinaryInputArchive iarchive(ifs);
			iarchive(model_interpreter);
			ifs.close();
		}
		return true;
	}
}

#endif //SMARTPEAK_MODELINTERPRETERFILE_H