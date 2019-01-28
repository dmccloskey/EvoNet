/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELINTERPRETERFILE_H
#define SMARTPEAK_MODELINTERPRETERFILE_H

// .h
#include <SmartPeak/ml/ModelInterpreterDefaultDevice.h>

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <cuda.h>
#include <cuda_runtime.h>
#endif

// .cpp
#include <cereal/archives/binary.hpp>
#include <fstream>

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

	/**
		@brief ModelInterpreterFileDefaultDevice
	*/
	template<typename TensorT>
	class ModelInterpreterFileDefaultDevice : public ModelInterpreterFile<TensorT, Eigen::DefaultDevice>
	{
	public:
		ModelInterpreterFileDefaultDevice() = default; ///< Default constructor
		~ModelInterpreterFileDefaultDevice() = default; ///< Default destructor
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ModelInterpreterFile<TensorT, Eigen::DefaultDevice>>(this));
		}
	};

#if COMPILE_WITH_CUDA
	/**
		@brief ModelInterpreterFileGpu
	*/
	template<typename TensorT>
	class ModelInterpreterFileGpu : public ModelInterpreterFile<TensorT, Eigen::GpuDevice>
	{
	public:
		ModelInterpreterFileGpu() = default; ///< Default constructor
		~ModelInterpreterFileGpu() = default; ///< Default destructor
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ModelInterpreterFile<TensorT, Eigen::GpuDevice>>(this));
		}
	};
#endif
}
#endif //SMARTPEAK_MODELINTERPRETERFILE_H