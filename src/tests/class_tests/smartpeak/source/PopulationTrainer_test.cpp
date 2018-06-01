/**TODO:  Add copyright*/

#define BOOST_TEST_MODULE PopulationTrainer test suite 
#include <boost/test/unit_test.hpp>
#include <SmartPeak/ml/PopulationTrainer.h>

#include <SmartPeak/ml/Model.h>
#include <SmartPeak/ml/Weight.h>
#include <SmartPeak/ml/Link.h>
#include <SmartPeak/ml/Node.h>

using namespace SmartPeak;
using namespace std;

BOOST_AUTO_TEST_SUITE(populationTrainer)

// BOOST_AUTO_TEST_CASE(constructor) 
// {
//   PopulationTrainer* ptr = nullptr;
//   PopulationTrainer* nullPointer = nullptr;
// 	ptr = new PopulationTrainer();
//   BOOST_CHECK_NE(ptr, nullPointer);
// }

// BOOST_AUTO_TEST_CASE(destructor) 
// {
//   PopulationTrainer* ptr = nullptr;
// 	ptr = new PopulationTrainer();
//   delete ptr;
// }

// BOOST_AUTO_TEST_CASE(DELETEAfterTesting) 
// {
//   PopulationTrainer population_trainer;

//   // define the model replicator for growth mode
//   ModelReplicator model_replicator;
//   model_replicator.setNNodeAdditions(1);
//   model_replicator.setNLinkAdditions(1);
//   model_replicator.setNNodeDeletions(0);
//   model_replicator.setNLinkDeletions(0);

//   // define the model trainer
//   class ModelTrainerTest: public ModelTrainer
//   {
//   public:
//     Model makeModel(){};
//     void trainModel(Model& model,
//       const Eigen::Tensor<float, 4>& input,
//       const Eigen::Tensor<float, 3>& output,
//       const Eigen::Tensor<float, 3>& time_steps,
//       const std::vector<std::string>& input_nodes,
//       const std::vector<std::string>& output_nodes)
//     {
//       // [TODO: define DCG training regime]
//     }
//     void validateModel(Model& model,
//       const Eigen::Tensor<float, 4>& input,
//       const Eigen::Tensor<float, 3>& output,
//       const Eigen::Tensor<float, 3>& time_steps,
//       const std::vector<std::string>& input_nodes,
//       const std::vector<std::string>& output_nodes)
//     {
//       // [TODO: define DCG validation regime]
//     }
//   };
//   ModelTrainerTest model_trainer;
//   model_trainer.setBatchSize(5);
//   model_trainer.setMemorySize(8);
//   model_trainer.setNEpochs(100);

//   // define the initial population of 10 baseline models
//   std::vector<Model> population; 
//   std::shared_ptr<WeightInitOp> weight_init;
//   std::shared_ptr<SolverOp> solver;
//   for (int i=0; i<10; ++i)
//   {
//     // baseline model
//     weight_init.reset(new ConstWeightInitOp(1.0));
//     solver.reset(new SGDOp(0.01, 0.9));
//     Model model = model_replicator.makeBaselineModel(
//       1, 0, 1,
//       NodeActivation::ReLU, NodeActivation::ReLU,
//       weight_init, solver);
    
//     // modify the models
//     model_replicator.modifyModel(model);

//     population.push_back(model);
//   }

//   // train the population
//   for (int i=0; i<population.size(); ++i)
//   {

//   }
// }

// BOOST_AUTO_TEST_CASE(selectModels) 
// {
//   PopulationTrainer population_trainer;

//   // [TODO: add tests]
// }

// BOOST_AUTO_TEST_CASE(copyModels) 
// {
//   PopulationTrainer population_trainer;

//   // [TODO: add tests]
// }

// BOOST_AUTO_TEST_CASE(modifyModels) 
// {
//   PopulationTrainer population_trainer;

//   // [TODO: add tests]
// }

// BOOST_AUTO_TEST_CASE(trainModels) 
// {
//   PopulationTrainer population_trainer;

//   // [TODO: add tests]
// }

// BOOST_AUTO_TEST_CASE(validateModels) 
// {
//   PopulationTrainer population_trainer;

//   // [TODO: add tests]
// }


BOOST_AUTO_TEST_SUITE_END()