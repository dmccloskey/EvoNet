#ifndef SMARTPEAK_MODELRESOURCES_H
#define SMARTPEAK_MODELRESOURCES_H

namespace SmartPeak
{
	enum class DeviceType
	{ // higher is better
		default = 1, 
		cpu = 2, 
		gpu = 3
	};

	/**
	@brief Helper class used by the user to define the device resources
	*/
  class ModelDevice
  {
	public:    
    ModelDevice() = default; ///< Default constructor 
		ModelDevice(const int& id, const DeviceType& type, const int& n_engines) :
			id_(id), type_(type), n_engines_(n_engines) {}; ///< Constructor    
		ModelDevice(const int& id, const DeviceType& type) :
			id_(id), type_(type) {}; ///< Constructor    
		~ModelDevice() = default; ///< Destructor

		void setID(const int& id) { id_ = id;};
		void setType(const DeviceType& type) { type_ = type; };
		void setNEngines(const int& n_engines) { n_engines_ = n_engines; };

		int getID() const { return id_; };
		DeviceType getType() const { return type_; };
		int getNEngines() const { return n_engines_; };

	private:
		int id_;  ///< ID of the device
		DeviceType type_; ///< the type of device
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