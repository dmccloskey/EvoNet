/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE KernalManager test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/core/KernalManager.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(KernalManager1)

BOOST_AUTO_TEST_CASE(constructorCpu)
{
	CpuKernal* ptr = nullptr;
	CpuKernal* nullPointer = nullptr;
	ptr = new CpuKernal();
	BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorCpu)
{
	CpuKernal* ptr = nullptr;
	ptr = new CpuKernal();
	delete ptr;
}

BOOST_AUTO_TEST_CASE(usageCpu)
{
	CpuKernal kernal(0, 1);
	kernal.initKernal();

	Eigen::Tensor<float, 1> in1(2);
	Eigen::Tensor<float, 1> in2(2);
	Eigen::Tensor<float, 1> out(2);
	in1.setConstant(10.0f);
	in2.setConstant(10.0f);

	Eigen::ThreadPool threadPool(kernal.getNThreads());
	Eigen::ThreadPoolDevice device(&threadPool, kernal.getNThreads());

	out.device(device) = in1 + in2;

	kernal.syncKernal();
	BOOST_CHECK_CLOSE(out(0), 20.0f, 1e-4);
	BOOST_CHECK_CLOSE(out(1), 20.0f, 1e-4);

	kernal.destroyKernal();
}

#if EVONET_CUDA
BOOST_AUTO_TEST_CASE(constructorGpu)
{
	GpuKernal* ptr = nullptr;
	GpuKernal* nullPointer = nullptr;
	ptr = new GpuKernal();
	BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorGpu)
{
	GpuKernal* ptr = nullptr;
	ptr = new GpuKernal();
	delete ptr;
}
#endif

BOOST_AUTO_TEST_SUITE_END()