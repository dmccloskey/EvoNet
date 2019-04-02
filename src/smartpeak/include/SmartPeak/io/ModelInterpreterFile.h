/**TODO:  Add copyright*/

#ifndef SMARTPEAK_MODELINTERPRETERFILE_H
#define SMARTPEAK_MODELINTERPRETERFILE_H

// .h
#include <SmartPeak/ml/ModelInterpreterDefaultDevice.h>

#if COMPILE_WITH_CUDA
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU
#include <SmartPeak/ml/ModelInterpreterGpu.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif

// .cpp
#include <cereal/archives/binary.hpp>
#include <fstream>
#include <SmartPeak/io/CSVWriter.h>

namespace SmartPeak
{
  /**
    @brief ModelInterpreterFile
  */
	template<typename TensorT, typename InterpreterT>
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
		static bool storeModelInterpreterBinary(const std::string& filename, const InterpreterT& model_interpreter);
    static bool storeModelInterpreterCsv(const std::string& filename, const InterpreterT& model_interpreter);
 
		/**
			@brief load Model from file

			@param filename The name of the model_interpreter file
			@param model_interpreter The model_interpreter to load data into

			@returns Status True on success, False if not
		*/
		static bool loadModelInterpreterBinary(const std::string& filename, InterpreterT& model_interpreter);
  };

	template<typename TensorT, typename InterpreterT>
	bool ModelInterpreterFile<TensorT, InterpreterT>::storeModelInterpreterBinary(const std::string & filename, const  InterpreterT& model_interpreter)
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

  template<typename TensorT, typename InterpreterT>
  inline bool ModelInterpreterFile<TensorT, InterpreterT>::storeModelInterpreterCsv(const std::string & filename, const InterpreterT& model_interpreter)
  {
    CSVWriter csvwriter(filename);

    // write the headers to the first line
    const std::vector<std::string> headers = { "Operation", "source_node_name", "source_node_timestep",
      "weight_name", "sink_node_name", "sink_node_timestep" };
    csvwriter.writeDataInRow(headers.begin(), headers.end());

    for (const auto& tensor_ops_step : model_interpreter.getTensorOpsSteps()) {
      for (const auto& tensor_op_map : tensor_ops_step) {
        for (const auto& tensor_op : tensor_op_map.second) {
          auto FP_operations = model_interpreter.getFPOperations();
          std::string sink_node_name = FP_operations[tensor_op].result.sink_node->getName();
          int sink_node_timestep = FP_operations[tensor_op].result.time_step;
          for (const auto& argument : FP_operations[tensor_op].arguments) {
            std::vector<std::string> row;
            row.push_back(tensor_op_map.first);
            row.push_back(argument.source_node->getName());
            row.push_back(std::to_string(argument.time_step));
            row.push_back(argument.weight->getName());
            row.push_back(sink_node_name);
            row.push_back(std::to_string(sink_node_timestep));

            // write to file
            csvwriter.writeDataInRow(row.begin(), row.end());
          }
        }
      }
    }
    return true;
  }

	template<typename TensorT, typename InterpreterT>
	bool ModelInterpreterFile<TensorT, InterpreterT>::loadModelInterpreterBinary(const std::string & filename, InterpreterT& model_interpreter)
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

#if COMPILE_WITH_CUDA
	/**
		@brief ModelInterpreterFileGpu
	*/
	template<typename TensorT>
	class ModelInterpreterFileGpu : public ModelInterpreterFile<TensorT, ModelInterpreterGpu<TensorT>>
	{
	public:
		ModelInterpreterFileGpu() = default; ///< Default constructor
		~ModelInterpreterFileGpu() = default; ///< Default destructor
	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(cereal::base_class<ModelInterpreterFile<TensorT, ModelInterpreterGpu<TensorT>>>(this));
		}
	};
#endif
}
#endif //SMARTPEAK_MODELINTERPRETERFILE_H