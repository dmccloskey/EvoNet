#ifndef EVONET_HELLOWORLD_H
#define EVONET_HELLOWORLD_H

namespace EvoNet
{

  class Helloworld
  {
public:
    /// Default constructor
    Helloworld();    
    /// Destructor
    ~Helloworld();

    double addNumbers(const double& x, const double& y) const;

  };
}

#endif //EVONET_HELLOWORLD_H