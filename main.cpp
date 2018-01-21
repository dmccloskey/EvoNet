// // TEST: should never fail
// #include <cstdio>
// #include <iostream>

// int main()
// {
//     printf("hello");
//     std::cout << "hello";
//     return 0;
// }

#include <smartPeak/helloworld.h>

int main(int argc, const char** argv)
{

  smartPeak::helloworld h;
  double r = h.addNumbers();
  std::cout << r << std::endl;

  return 0;
} //end of main