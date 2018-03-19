/**TODO:  Add copyright*/

#include <SmartPeak/ml/Weight.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <cmath>
#include <iostream>

namespace SmartPeak
{
  Weight::Weight()
  {        
  }

  Weight::Weight(const int& id):
    id_(id)
  {
  }

  Weight::~Weight()
  {
  }
  
  void Weight::setId(const int& id)
  {
    id_ = id;
  }
  int Weight::getId() const
  {
    return id_;
  }

  void Weight::setWeight(const float& weight)
  {
    weight_ = weight;
  }
  float Weight::getWeight() const
  {
    return weight_;
  }

  void Weight::setWeightInitOp(WeightInitOp& weight_init)
  {
    weight_init_ = &weight_init;
  }
  WeightInitOp* Weight::getWeightInitOp() const
  {
    return weight_init_;
  }

  void Weight::setSolverOp(SolverOp& solver)
  {
    solver_ = &solver;
  }
  SolverOp* Weight::getSolverOp() const
  {
    return solver_;
  }


}