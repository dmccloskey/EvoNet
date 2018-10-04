#ifndef SMARTPEAK_DEVICEMANAGER_H
#define SMARTPEAK_DEVICEMANAGER_H

#include <unsupported/Eigen/CXX11/Tensor>

namespace SmartPeak
{
	enum DevicePriority
	{
		HIGH = 0,
		MEDIUM = 1,
		LOW = 2,
	};

	template <typename DeviceType>
  class DeviceManager
  {
	public:    
    DeviceManager() = default; ///< Default constructor 
		DeviceManager(const int& id, const DevicePriority& device_priority, const int& n_kernals) :
			id_(id), device_priority_(device_priority), n_kernals_(n_kernals) {}; ///< Constructor    
		~DeviceManager() = default; ///< Destructor

		virtual void setDevice(int n_kernals = 0) = 0;
		DeviceType getDevice() { return device_; };

		void setID(const int& id) { id_ = id;};
		void setName(const std::string& name) { name_ = name; };
		void setDevicePriority(const DevicePriority& device_priority) { device_priority_ = device_priority; };
		void setNKernals(const int& n_kernals) { n_kernals_ = n_kernals; };

		int getID() const { return id_; };
		std::string getName() const { return name_; };
		DevicePriority getDevicePriority() const { return device_priority_; };
		int getNKernals() const { return n_kernals_; };

	private:
		int id_;  ///< ID of the device
		std::string name_; ///< the name of the device
		DeviceType device_;  ///< the type of device (e.g., CPU or GPU)
		DevicePriority device_priority_; ///< the priority of the device
		int n_kernals_; ///< the number of threads (CPU) or streams (GPU) available on the device
  };
}

#endif //SMARTPEAK_DEVICEMANAGER_H