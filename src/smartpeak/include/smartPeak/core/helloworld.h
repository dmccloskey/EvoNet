#ifndef SMARTPEAK_HELLOWORLD_H
#define SMARTPEAK_HELLOWORLD_H

namespace smartPeak
{

  class helloworld
  {
public:
    /// Default constructor
    helloworld();    
    /// Destructor
    ~helloworld();

    double addNumbers(const double& x, const double& y) const;

  };
}

#endif SMARTPEAK_HELLOWORLD_H