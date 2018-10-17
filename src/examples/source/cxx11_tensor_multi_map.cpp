// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

using Eigen::Tensor;
using Eigen::RowMajor;
using Eigen::TensorMultiMap;

static void test_0d()
{
	Tensor<int, 0> scalar1;
	Tensor<int, 0, RowMajor> scalar2;

	std::vector<int*>scalar1_d = { scalar1.data() };
	std::vector<int*>scalar2_d = { scalar2.data() };

	Eigen::TensorMap<Eigen::Tensor<int, 0>> test(scalar1.data());

	TensorMultiMap<Tensor<int, 0> > scalar3(scalar1.data());
	TensorMultiMap<Tensor<int, 0, RowMajor> > scalar4(scalar1.data());

	scalar1() = 7;
	scalar2() = 13;

	assert(scalar1.rank() == 0);
	assert(scalar1.size() == 1);

	assert(scalar3() == 7);
	assert(scalar4() == 13);
}

static void test_1d()
{
	Tensor<int, 1> vec1(6);
	Tensor<int, 1, RowMajor> vec2(6);

	TensorMultiMap<Tensor<const int, 1> > vec3(vec1.data(), 6);
	TensorMultiMap<Tensor<const int, 1, RowMajor> > vec4(vec2.data(), 6);

	vec1(0) = 4;  vec2(0) = 0;
	vec1(1) = 8;  vec2(1) = 1;
	vec1(2) = 15; vec2(2) = 2;
	vec1(3) = 16; vec2(3) = 3;
	vec1(4) = 23; vec2(4) = 4;
	vec1(5) = 42; vec2(5) = 5;

	assert(vec1.rank() == 1);
	assert(vec1.size() == 6);
	assert(vec1.dimension(0) == 6);

	assert(vec3(0) == 4);
	assert(vec3(1) == 8);
	assert(vec3(2) == 15);
	assert(vec3(3) == 16);
	assert(vec3(4) == 23);
	assert(vec3(5) == 42);

	assert(vec4(0) == 0);
	assert(vec4(1) == 1);
	assert(vec4(2) == 2);
	assert(vec4(3) == 3);
	assert(vec4(4) == 4);
	assert(vec4(5) == 5);
}

static void test_2d()
{
	Tensor<int, 2> mat1(2, 3);
	Tensor<int, 2, RowMajor> mat2(2, 3);

	mat1(0, 0) = 0;
	mat1(0, 1) = 1;
	mat1(0, 2) = 2;
	mat1(1, 0) = 3;
	mat1(1, 1) = 4;
	mat1(1, 2) = 5;

	mat2(0, 0) = 0;
	mat2(0, 1) = 1;
	mat2(0, 2) = 2;
	mat2(1, 0) = 3;
	mat2(1, 1) = 4;
	mat2(1, 2) = 5;

	TensorMultiMap<Tensor<const int, 2> > mat3(mat1.data(), 2, 3);
	TensorMultiMap<Tensor<const int, 2, RowMajor> > mat4(mat2.data(), 2, 3);

	assert(mat3.rank() == 2);
	assert(mat3.size() == 6);
	assert(mat3.dimension(0) == 2);
	assert(mat3.dimension(1) == 3);

	assert(mat4.rank() == 2);
	assert(mat4.size() == 6);
	assert(mat4.dimension(0) == 2);
	assert(mat4.dimension(1) == 3);

	assert(mat3(0 == 0) == 0);
	assert(mat3(0 == 1) == 1);
	assert(mat3(0 == 2) == 2);
	assert(mat3(1 == 0) == 3);
	assert(mat3(1 == 1) == 4);
	assert(mat3(1 == 2) == 5);

	assert(mat4(0 == 0) == 0);
	assert(mat4(0 == 1) == 1);
	assert(mat4(0 == 2) == 2);
	assert(mat4(1 == 0) == 3);
	assert(mat4(1 == 1) == 4);
	assert(mat4(1 == 2) == 5);
}

static void test_3d()
{
	Tensor<int, 3> mat1(2, 3, 7);
	Tensor<int, 3, RowMajor> mat2(2, 3, 7);

	int val = 0;
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 3; ++j) {
			for (int k = 0; k < 7; ++k) {
				mat1(i, j, k) = val;
				mat2(i, j, k) = val;
				val++;
			}
		}
	}

	TensorMultiMap<Tensor<const int, 3> > mat3(mat1.data(), 2, 3, 7);
	TensorMultiMap<Tensor<const int, 3, RowMajor> > mat4(mat2.data(), 2, 3, 7);

	assert(mat3.rank() == 3);
	assert(mat3.size() == 2 * 3 * 7);
	assert(mat3.dimension(0) == 2);
	assert(mat3.dimension(1) == 3);
	assert(mat3.dimension(2) == 7);

	assert(mat4.rank() == 3);
	assert(mat4.size() == 2 * 3 * 7);
	assert(mat4.dimension(0) == 2);
	assert(mat4.dimension(1) == 3);
	assert(mat4.dimension(2) == 7);

	val = 0;
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 3; ++j) {
			for (int k = 0; k < 7; ++k) {
				assert(mat3(i, j, k), val);
				assert(mat4(i, j, k), val);
				val++;
			}
		}
	}
}

//
//static void test_from_tensor()
//{
//	Tensor<int, 3> mat1(2, 3, 7);
//	Tensor<int, 3, RowMajor> mat2(2, 3, 7);
//
//	int val = 0;
//	for (int i = 0; i < 2; ++i) {
//		for (int j = 0; j < 3; ++j) {
//			for (int k = 0; k < 7; ++k) {
//				mat1(i, j, k) = val;
//				mat2(i, j, k) = val;
//				val++;
//			}
//		}
//	}
//
//	TensorMultiMap<Tensor<int, 3> > mat3(mat1);
//	TensorMultiMap<Tensor<int, 3, RowMajor> > mat4(mat2);
//
//	assert(mat3.rank(), 3);
//	assert(mat3.size(), 2 * 3 * 7);
//	assert(mat3.dimension(0), 2);
//	assert(mat3.dimension(1), 3);
//	assert(mat3.dimension(2), 7);
//
//	assert(mat4.rank(), 3);
//	assert(mat4.size(), 2 * 3 * 7);
//	assert(mat4.dimension(0), 2);
//	assert(mat4.dimension(1), 3);
//	assert(mat4.dimension(2), 7);
//
//	val = 0;
//	for (int i = 0; i < 2; ++i) {
//		for (int j = 0; j < 3; ++j) {
//			for (int k = 0; k < 7; ++k) {
//				assert(mat3(i, j, k), val);
//				assert(mat4(i, j, k), val);
//				val++;
//			}
//		}
//	}
//
//	TensorFixedSize<int, Sizes<2, 3, 7> > mat5;
//
//	val = 0;
//	for (int i = 0; i < 2; ++i) {
//		for (int j = 0; j < 3; ++j) {
//			for (int k = 0; k < 7; ++k) {
//				array<ptrdiff_t, 3> coords;
//				coords[0] = i;
//				coords[1] = j;
//				coords[2] = k;
//				mat5(coords) = val;
//				val++;
//			}
//		}
//	}
//
//	TensorMultiMap<TensorFixedSize<int, Sizes<2, 3, 7> > > mat6(mat5);
//
//	assert(mat6.rank(), 3);
//	assert(mat6.size(), 2 * 3 * 7);
//	assert(mat6.dimension(0), 2);
//	assert(mat6.dimension(1), 3);
//	assert(mat6.dimension(2), 7);
//
//	val = 0;
//	for (int i = 0; i < 2; ++i) {
//		for (int j = 0; j < 3; ++j) {
//			for (int k = 0; k < 7; ++k) {
//				assert(mat6(i, j, k), val);
//				val++;
//			}
//		}
//	}
//}
//
//
//static int f(const TensorMultiMap<Tensor<int, 3> >& tensor) {
//	//  Size<0> empty;
//	EIGEN_STATIC_ASSERT((internal::array_size<Sizes<> >::value == 0), YOU_MADE_A_PROGRAMMING_MISTAKE);
//	EIGEN_STATIC_ASSERT((internal::array_size<DSizes<int, 0> >::value == 0), YOU_MADE_A_PROGRAMMING_MISTAKE);
//	Tensor<int, 0> result = tensor.sum();
//	return result();
//}
//
//static void test_casting()
//{
//	Tensor<int, 3> tensor(2, 3, 7);
//
//	int val = 0;
//	for (int i = 0; i < 2; ++i) {
//		for (int j = 0; j < 3; ++j) {
//			for (int k = 0; k < 7; ++k) {
//				tensor(i, j, k) = val;
//				val++;
//			}
//		}
//	}
//
//	TensorMultiMap<Tensor<int, 3> > map(tensor);
//	int sum1 = f(map);
//	int sum2 = f(tensor);
//
//	assert(sum1 == sum2);
//	assert(sum1 == 861);
//}

int main(int argc, char** argv)
{
	test_0d();
	test_1d();
	test_2d();
	test_3d();

	test_from_tensor();
	test_casting();
	return 0;
}