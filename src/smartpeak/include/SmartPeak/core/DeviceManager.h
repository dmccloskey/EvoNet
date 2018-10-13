#ifndef SMARTPEAK_DEVICEMANAGER_H
#define SMARTPEAK_DEVICEMANAGER_H

#include <unsupported/Eigen/CXX11/Tensor>

namespace SmartPeak
{
  class DeviceManager
  {
	public:    
    DeviceManager() = default; ///< Default constructor 
		DeviceManager(const int& id, const int& n_engines) :
			id_(id), n_engines_(n_engines) {}; ///< Constructor    
		~DeviceManager() = default; ///< Destructor

		void setID(const int& id) { id_ = id;};
		void setName(const std::string& name) { name_ = name; };
		void setNEngines(const int& n_engines) { n_engines_ = n_engines; };

		int getID() const { return id_; };
		std::string getName() const { return name_; };
		int getNEngines() const { return n_engines_; };

	private:
		int id_;  ///< ID of the device
		std::string name_; ///< the name of the device
		int n_engines_; ///< the number of threads (CPU) or asynchroous engines (GPU) available on the device
		int steams_used_; ///< the current number of engines in use
  };
}

#endif //SMARTPEAK_DEVICEMANAGER_H