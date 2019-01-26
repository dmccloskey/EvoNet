#ifndef SMARTPEAK_MODELRESOURCES_H
#define SMARTPEAK_MODELRESOURCES_H

#include <cereal/access.hpp>  // serialiation of private members

namespace SmartPeak
{

	/**
	@brief Helper class used by the user to define the device resources
	*/
  class ModelDevice
  {
	public:    
    ModelDevice() = default; ///< Default constructor 
		ModelDevice(const int& id, const int& n_engines) :
			id_(id), n_engines_(n_engines) {}; ///< Constructor    
		ModelDevice(const int& id) :
			id_(id) {}; ///< Constructor    
		~ModelDevice() = default; ///< Destructor

		void setID(const int& id) { id_ = id;};
		void setNEngines(const int& n_engines) { n_engines_ = n_engines; };

		int getID() const { return id_; };
		int getNEngines() const { return n_engines_; };

	private:
		friend class cereal::access;
		template<class Archive>
		void serialize(Archive& archive) {
			archive(id_, n_engines_);
		}
		int id_;  ///< ID of the device
		int n_engines_ = -1; ///< the number of threads (CPU) or asynchroous engines (GPU) available on the device
  };

	/**
	@brief List of available devices for training each model

	The ModelTrainer will invoke the correct ModelInterpreter based on the DeviceType
	*/
	typedef std::vector<ModelDevice> ModelResources;

	/**
	@brief List of available resources for training each model in the population.
		It is assumed that each element in the vector will be given a seperate thread
		to control the model training.

	Example 1: 1 GPU per model for a population size of 16 with 4 concurrent training sessions
	ModelResources.size() = 1;
	PopulationResources.size() = 4;

	Example 2: 2 GPU per model for a population size of 16 with 4 concurrent training sessions
	ModelResources.size() = 2;
	PopulationResources.size() = 4;

	Example 3: 1 CPU Pool per model for a population size of 16 with 4 concurrent training sessions
	ModelResources.size() = 1;
	PopulationResources.size() = 4;
	*/
	typedef std::vector<ModelResources> PopulationResources;
}

#endif //SMARTPEAK_MODELRESOURCES_H