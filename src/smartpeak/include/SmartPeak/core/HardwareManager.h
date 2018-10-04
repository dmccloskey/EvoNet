#ifndef SMARTPEAK_HARDWAREMANAGER_H
#define SMARTPEAK_HARDWAREMANAGER_H

#include <SmartPeak/core/DeviceManager>
#include <memory>

namespace SmartPeak
{
	template <typename DeviceType>
  class HardwareManager
  {
	public:    
    HardwareManager() = default; ///< Default constructor    
		~HardwareManager() = default; ///< Destructor

		void addDevice(const std::shared_ptr<DeviceManager<DeviceType>>& device) {
			shared_ptr<DeviceManager<DeviceType>> device_ptr;
			device_ptr.reset(new DeviceManager<DeviceType>(device));
			auto found_id = devices_.emplace(device->getID(), device_ptr);
			if (found_id.second) {
				std::cout << "Device: " << device->getID() << " will be updated." << std::endl;
				devices_.at(device->getID()) = device_ptr;
			}
			auto found_name = devices_by_name_.emplace(device->getName(), device_ptr);
			if (found_name.second) {
				std::cout << "Device: " << device->getName() << " will be updated." << std::endl;
				devices_by_name_.at(device->getName()) = device_ptr;
			}
		};

		DeviceManager getDevice(const int& id) const { return devices_.at(id); };
		DeviceManager getDevice(const std::string& name) const { return devices_by_name_.at(name); };

	private:
		std::map<int, std::shared_ptr<DeviceManager<DeviceType>>> devices_;
		std::map<std::string, std::shared_ptr<DeviceManager<DeviceType>>> devices_by_name_;
  };
}

#endif //SMARTPEAK_HARDWAREMANAGER_H