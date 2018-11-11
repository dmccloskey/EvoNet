#ifndef SMARTPEAK_DEVICEMANAGER_H
#define SMARTPEAK_DEVICEMANAGER_H

#include <unsupported/Eigen/CXX11/Tensor>

namespace SmartPeak
{
	enum class DeviceType
	{ // higher is better
		default = 1, 
		cpu = 2, 
		gpu = 3
	};

  class DeviceManager
  {
	public:    
    DeviceManager() = default; ///< Default constructor 
		DeviceManager(const int& id, const DeviceType& type, const int& n_engines) :
			id_(id), type_(type), n_engines_(n_engines) {}; ///< Constructor    
		DeviceManager(const int& id, const DeviceType& type) :
			id_(id), type_(type) {}; ///< Constructor    
		~DeviceManager() = default; ///< Destructor

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
}

#endif //SMARTPEAK_DEVICEMANAGER_H