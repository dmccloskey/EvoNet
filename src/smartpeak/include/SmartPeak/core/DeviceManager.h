#ifndef SMARTPEAK_DEVICEMANAGER_H
#define SMARTPEAK_DEVICEMANAGER_H

#include <unsupported/Eigen/CXX11/Tensor>

namespace SmartPeak
{
	//template <typename DeviceType>
	template <typename TensorT>
	class KernalManager 
	{
	public:
		KernalManager() = default; ///< Default constructor  
		explicit KernalManager(int device_id = -1, int n_threads = 1) : device_id_(device_id), n_threads_(n_threads) {};
		~KernalManager() = default; ///< Destructor

		virtual void initKernal() = 0;
		virtual void syncKernal() = 0;
		virtual void destroyKernal() = 0;

		virtual void executeForwardPropogationOp() = 0; ///< GPU/CPU specific FP operations
		virtual void executeBackwardPropogationOp() = 0;
		virtual void executeCalcError() = 0;
		virtual void executeUpdateWeights() = 0;

		//DeviceType getDevice() { return device_; };
		int getDeviceID() const { return device_id_; };
		int getNThreads() const { return n_threads_; };

	protected:
		int device_id_;
		int n_threads_;
		//DeviceType device_; 
	};

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