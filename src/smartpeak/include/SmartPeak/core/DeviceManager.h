#ifndef SMARTPEAK_DEVICEMANAGER_H
#define SMARTPEAK_DEVICEMANAGER_H

#include <unsupported/Eigen/CXX11/Tensor>

namespace SmartPeak
{
	//template <typename DeviceType>
	class KernalManager 
	{
	public:
		KernalManager() = default; ///< Default constructor  
		~KernalManager() = default; ///< Destructor

		virtual void executeForwardPropogationOp() = 0; ///< GPU/CPU specific FP operations
		virtual void executeBackwardPropogationOp() = 0;
		virtual void executeCalcError() = 0;
		virtual void executeUpdateWeights() = 0;

		//DeviceType getDevice() { return device_; };
		int getDeviceID() const { return device_id_; };

	protected:
		int device_id_;
		//DeviceType device_; 
	};

  class DeviceManager
  {
	public:    
    DeviceManager() = default; ///< Default constructor 
		DeviceManager(const int& id, const int& n_streams) :
			id_(id), n_streams_(n_streams) {}; ///< Constructor    
		~DeviceManager() = default; ///< Destructor

		void setID(const int& id) { id_ = id;};
		void setName(const std::string& name) { name_ = name; };
		void setNStreams(const int& n_streams) { n_streams_ = n_streams; };

		int getID() const { return id_; };
		std::string getName() const { return name_; };
		int getNStreams() const { return n_streams_; };

	private:
		int id_;  ///< ID of the device
		std::string name_; ///< the name of the device
		int n_streams_; ///< the number of threads (CPU) or streams (GPU) available on the device
		int steams_used_; ///< the current number of streams in use
  };
}

#endif //SMARTPEAK_DEVICEMANAGER_H