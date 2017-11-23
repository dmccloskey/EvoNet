// // TEST: should never fail
// #include <cstdio>
// #include <iostream>

// int main()
// {
//     printf("hello");
//     std::cout << "hello";
//     return 0;
// }

#include <OpenMS/FILTERING/TRANSFORMERS/LinearResampler.h>
#include <OpenMS/FILTERING/SMOOTHING/SavitzkyGolayFilter.h>
#include <OpenMS/FORMAT/DTAFile.h>
#include <OpenMS/KERNEL/StandardTypes.h>
#include <iostream>
#include <string>
#include "/home/user/code/OpenMS/include/TransformationModel.h"
#include "/home/user/code/OpenMS/include/TransformationModelLinear.h"

#include <OpenMS/CONCEPT/ClassTest.h>

using namespace OpenMS;
using namespace std;

// code snippet testing
int main(int argc, const char** argv)
{
  // if (argc < 2) return 1;
  // // the path to the data should be given on the command line
  // string tutorial_data_path(argv[1]);
  
//   PeakSpectrum spectrum;

//   DTAFile dta_file;
//   dta_file.load(tutorial_data_path + "/data/Tutorial_SavitzkyGolayFilter.dta", spectrum);

//   LinearResampler lr;
//   Param param_lr;
//   param_lr.setValue("spacing", 0.01);
//   lr.setParameters(param_lr);
//   lr.raster(spectrum);

//   SavitzkyGolayFilter sg;
//   Param param_sg;
//   param_sg.setValue("frame_length", 21);
//   param_sg.setValue("polynomial_order", 3);
//   sg.setParameters(param_sg);
//   sg.filter(spectrum);

TransformationModelLinear* ptr = 0;
TransformationModelLinear* nullPointer = 0;

  // set-up the parameters
  Param param; 
  std::string x_weight_test, y_weight_test;
  x_weight_test = "ln(x)";
  y_weight_test = "ln(y)";
  param.setValue("x_weight", x_weight_test);
  param.setValue("y_weight", y_weight_test);
  param.setValue("x_datum_min", 1e-15);
  param.setValue("x_datum_max", 1e8);
  param.setValue("y_datum_min", 1e-8);
  param.setValue("y_datum_max", 1e15);

  // set-up the data and test
  TransformationModel::DataPoints data1;
  TransformationModel::DataPoints test1;
  data1.clear();
  data1.push_back(make_pair(1, 2));
  data1.push_back(make_pair(2, 4));
  data1.push_back(make_pair(4, 8)); 

  // test evaluate
  TransformationModelLinear lm(data1, param);
  TEST_REAL_SIMILAR(lm.evaluate(2),4);

  // test evaluate using the inverted model
  lm.invert();
  TEST_REAL_SIMILAR(lm.evaluate(4),2);

  return 0;
} //end of main