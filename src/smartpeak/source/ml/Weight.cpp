/**TODO:  Add copyright*/

#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/Operation.h>

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

  Weight::Weight(const int& id,
      const SmartPeak::WeightInitMethod& weight_init,
      const SmartPeak::WeightUpdateMethod& weight_update):
    id_(id), weight_init_(weight_init), weight_update_(weight_update)
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
  void Weight::setWeightUpdates(const Eigen::Tensor<float, 1>& weight_updates)
  {
    weight_updates_ = weight_updates;
  }
  Eigen::Tensor<float, 1> Weight::getWeightUpdates() const
  {
    return weight_updates_;
  }

  void Weight::setWeightInitMethod(const SmartPeak::WeightInitMethod& weight_init)
  {
    weight_init_ = weight_init;
  }
  SmartPeak::WeightInitMethod Weight::getWeightInitMethod() const
  {
    return weight_init_;
  }

  void Weight::setWeightUpdateMethod(const SmartPeak::WeightUpdateMethod& weight_update)
  {
    weight_update_ = weight_update;
  }
  SmartPeak::WeightUpdateMethod Weight::getWeightUpdateMethod() const
  {
    return weight_update_;
  }

  void Weight::initWeight(const float& op_input)
  {
    switch (weight_init_)
    {
      case SmartPeak::WeightInitMethod::RandWeightInit:
      {
        RandWeightInitOp operation;
        weight_ = operation(op_input);
        break;
      }
      case SmartPeak::WeightInitMethod::ConstWeightInit:
      {
        ConstWeightInitOp operation;
        weight_ = operation(op_input);
        break;
      }
      default:
      {
        weight_ = 0.0;
        break;
      }
    }
  }
}