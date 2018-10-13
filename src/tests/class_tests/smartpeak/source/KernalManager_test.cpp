/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE KernalManager test suite 
#include <boost/test/included/unit_test.hpp>
#include <SmartPeak/core/KernalManager.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(KernalManager1)

BOOST_AUTO_TEST_CASE(constructorDefault)
{
	DefaultKernal* ptr = nullptr;
	DefaultKernal* nullPointer = nullptr;
	ptr = new DefaultKernal();
	BOOST_CHECK_NE(ptr, nullPointer);
}

BOOST_AUTO_TEST_CASE(destructorDefault)
{
	DefaultKernal* ptr = nullptr;
	ptr = new DefaultKernal();
	delete ptr;
}

BOOST_AUTO_TEST_CASE(usageDefault)
{
	DefaultKernal kernal;

	Eigen::Tensor<float, 1> in1(2);
	Eigen::Tensor<float, 1> in2(2);
	Eigen::Tensor<float, 1> out(2);
	in1.setConstant(10.0f);
	in2.setConstant(10.0f);

	out.device(kernal.getDevice()) = in1 + in2;

	BOOST_CHECK_CLOSE(out(0), 20.0f, 1e-4);
	BOOST_CHECK_CLOSE(out(1), 20.0f, 1e-4);

	kernal.initKernal();
	kernal.syncKernal();
	kernal.destroyKernal();
}

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

	Eigen::Tensor<float, 1> in1(2);
	Eigen::Tensor<float, 1> in2(2);
	Eigen::Tensor<float, 1> out(2);
	in1.setConstant(10.0f);
	in2.setConstant(10.0f);

	out.device(kernal.getDevice()) = in1 + in2;

	BOOST_CHECK_CLOSE(out(0), 20.0f, 1e-4);
	BOOST_CHECK_CLOSE(out(1), 20.0f, 1e-4);

	kernal.initKernal();
	kernal.syncKernal();
	kernal.destroyKernal();
}

#ifndef EVONET_CUDA
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