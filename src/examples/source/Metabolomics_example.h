/**TODO:  Add copyright*/

#include <SmartPeak/simulator/BiochemicalReaction.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace SmartPeak;

// Extended data classes
template<typename TensorT>
class MetDataSimClassification : public DataSimulator<TensorT>
{
public:
	MetDataSimClassification() = default;
	~MetDataSimClassification() = default;

	void simulateData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_output_nodes = output_data.dimension(2);
		const int n_epochs = input_data.dimension(3);

		// NOTE: used for testing
		//std::string sample_group_name = sample_group_names_[0];
		//std::vector<float> mars;
		//for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
		//	float mar = calculateMAR(metabolomicsData_.at(sample_group_name),
		//		biochemicalReactions_.at(reaction_ids_[nodes_iter]));
		//	mars.push_back(mar);
		//	//std::cout << "OutputNode: "<<nodes_iter<< " = " << mar << std::endl;
		//}

		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
				for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

					// pick a random sample group name
					std::string sample_group_name = selectRandomElement(this->model_.sample_group_names_);

					for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = this->model_.calculateMAR(
							this->model_.metabolomicsData_.at(sample_group_name),
							this->model_.biochemicalReactions_.at(this->model_.reaction_ids_[nodes_iter]));
						//input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = mars[nodes_iter]; // NOTE: used for testing
					}

					// convert the label to a one hot vector
					Eigen::Tensor<TensorT, 1> one_hot_vec = OneHotEncoder<std::string, TensorT>(this->model_.metaData_.at(sample_group_name).condition, this->model_.labels_);
					Eigen::Tensor<TensorT, 1> one_hot_vec_smoothed = one_hot_vec.unaryExpr(LabelSmoother<TensorT>(0.01, 0.01));

					//// MSE + LogLoss
					//for (int nodes_iter = 0; nodes_iter < n_output_nodes/2; ++nodes_iter) {
					//	output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = one_hot_vec(nodes_iter);
					//	output_data(batch_iter, memory_iter, nodes_iter + n_output_nodes/2, epochs_iter) = one_hot_vec(nodes_iter);
					//	//output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = one_hot_vec_smoothed(nodes_iter);
					//}

					// MSE or LogLoss only
					for (int nodes_iter = 0; nodes_iter < n_output_nodes; ++nodes_iter) {
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = one_hot_vec(nodes_iter);
						//output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = one_hot_vec_smoothed(nodes_iter);
					}
				}
			}
		}

		// update the time_steps
		time_steps.setConstant(1.0f);
	}
	void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		simulateData(input_data, output_data, time_steps);
	}
	void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		simulateData(input_data, output_data, time_steps);
	}
	void simulateEvaluationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 3>& time_steps) {};

	BiochemicalReactionModel<TensorT> model_;
};

template<typename TensorT>
class MetDataSimReconstruction : public MetDataSimClassification<TensorT>
{
public:
	void simulateData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		// infer data dimensions based on the input tensors
		const int batch_size = input_data.dimension(0);
		const int memory_size = input_data.dimension(1);
		const int n_input_nodes = input_data.dimension(2);
		const int n_output_nodes = output_data.dimension(2);
		const int n_epochs = input_data.dimension(3);

		for (int batch_iter = 0; batch_iter < batch_size; ++batch_iter) {
			for (int memory_iter = 0; memory_iter < memory_size; ++memory_iter) {
				for (int epochs_iter = 0; epochs_iter < n_epochs; ++epochs_iter) {

					// pick a random sample group name
					//std::string sample_group_name = selectRandomElement(sample_group_names_);
					std::string sample_group_name = this->model_.sample_group_names_[0];

					for (int nodes_iter = 0; nodes_iter < n_input_nodes; ++nodes_iter) {
						const TensorT mar = this->model_.calculateMAR(
							this->model_.metabolomicsData_.at(sample_group_name),
							this->model_.biochemicalReactions_.at(this->model_.reaction_ids_[nodes_iter]));
						input_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = mar;
						output_data(batch_iter, memory_iter, nodes_iter, epochs_iter) = mar;
					}
				}
			}
		}
	}
	void simulateTrainingData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		simulateData(input_data, output_data, time_steps);
	}
	void simulateValidationData(Eigen::Tensor<TensorT, 4>& input_data, Eigen::Tensor<TensorT, 4>& output_data, Eigen::Tensor<TensorT, 3>& time_steps)
	{
		simulateData(input_data, output_data, time_steps);
	}
	
	BiochemicalReactionModel<TensorT> model_;
};

/*
@brief Find significant pair-wise MARS between samples (one pre/post vs. all pre/post)
*/
PWData PWComparison(BiochemicalReactionModel<float>& metabolomics_data, const std::vector<std::string>& sample_names, int n_samples = 10000, float alpha = 0.05, float fc = 1.0) {
	PWData pw_data;
	for (const std::string& mar : metabolomics_data.reaction_ids_) {
		for (size_t sgn1_iter = 0; sgn1_iter < sample_names.size(); ++sgn1_iter) {

			// check if the sample name exists
			if (metabolomics_data.metabolomicsData_.count(sample_names[sgn1_iter]) == 0)
				continue;

			// sample the MAR data
			std::vector<float> samples1;
			for (int sample_iter = 0; sample_iter < n_samples; ++sample_iter) {
				samples1.push_back(
					metabolomics_data.calculateMAR(metabolomics_data.metabolomicsData_.at(sample_names[sgn1_iter]),
						metabolomics_data.biochemicalReactions_.at(mar)));
			}
			for (size_t sgn2_iter = sgn1_iter + 1; sgn2_iter < sample_names.size(); ++sgn2_iter) {

				// check if the sample name exists
				if (metabolomics_data.metabolomicsData_.count(sample_names[sgn2_iter]) == 0)
					continue;

				std::cout << "MAR: " << mar << " Sample1: " << sgn1_iter << " Sample2: " << sgn2_iter << std::endl;

				// initialize the data struct
				PWStats pw_stats;
				pw_stats.feature_name = mar;
				pw_stats.feature_comment = metabolomics_data.biochemicalReactions_.at(mar).equation;
				pw_stats.sample_name_1 = sample_names[sgn1_iter];
				pw_stats.sample_name_2 = sample_names[sgn2_iter];
				pw_stats.n1 = n_samples;
				pw_stats.n2 = n_samples;

				// sample the MAR data
				std::vector<float> samples2;
				for (int sample_iter = 0; sample_iter < n_samples; ++sample_iter) {
					samples2.push_back(
						metabolomics_data.calculateMAR(metabolomics_data.metabolomicsData_.at(sample_names[sgn2_iter]),
							metabolomics_data.biochemicalReactions_.at(mar)));
				}

				// calculate the moments and fold change
				float ave1, adev1, sdev1, var1, skew1, curt1;
				SmartPeak::moment(&samples1[0], n_samples, ave1, adev1, sdev1, var1, skew1, curt1);
				float ave2, adev2, sdev2, var2, skew2, curt2;
				SmartPeak::moment(&samples2[0], n_samples, ave2, adev2, sdev2, var2, skew2, curt2);
				pw_stats.fold_change = std::log2(ave2 / ave1);

				// calculate the 95% CI
				pw_stats.confidence_interval_1 = confidence(samples1, alpha);
				pw_stats.confidence_interval_2 = confidence(samples2, alpha);

				//// calculate the K-S prob
				//float d, prob;
				//kstwo(&samples1[0], n_samples, &samples2[0], n_samples, d, prob);
				//pw_stats.prob = prob;

				//if (prob < 0.05) {
				if ((pw_stats.confidence_interval_1.first > pw_stats.confidence_interval_2.second
					|| pw_stats.confidence_interval_1.second < pw_stats.confidence_interval_2.first)
					&& (pw_stats.fold_change > fc || pw_stats.fold_change < -fc)) {
					pw_stats.is_significant = true;
					std::vector<PWStats> pw_stats_vec = { pw_stats };
					auto found = pw_data.emplace(mar, pw_stats_vec);
					if (!found.second) {
						pw_data.at(mar).push_back(pw_stats);
					}
				}
			}
		}
	}
	return pw_data;
}

/*
@brief Find significant pair-wise MARS between pre/post samples (one pre vs one post)
*/
PWData PWPrePostComparison(BiochemicalReactionModel<float>& metabolomics_data,
	std::vector<std::string>& pre_samples, std::vector<std::string>& post_samples, const int& n_pairs,
	int n_samples = 10000, float alpha = 0.05, float fc = 1.0) {
	PWData pw_data;
	for (const std::string& mar : metabolomics_data.reaction_ids_) {
		for (size_t pairs_iter = 0; pairs_iter < n_pairs; ++pairs_iter) {

			// check if the sample name exists
			if (metabolomics_data.metabolomicsData_.count(pre_samples[pairs_iter]) == 0 ||
				metabolomics_data.metabolomicsData_.count(post_samples[pairs_iter]) == 0)
				continue;

			std::cout << "MAR: " << mar << " Pair: " << pairs_iter << std::endl;

			// initialize the data struct
			PWStats pw_stats;
			pw_stats.feature_name = mar;
			pw_stats.feature_comment = metabolomics_data.biochemicalReactions_.at(mar).equation;
			pw_stats.sample_name_1 = pre_samples[pairs_iter];
			pw_stats.sample_name_2 = post_samples[pairs_iter];
			pw_stats.n1 = n_samples;
			pw_stats.n2 = n_samples;

			// sample the MAR data
			std::vector<float> samples1, samples2;
			for (int sample_iter = 0; sample_iter < n_samples; ++sample_iter) {
				samples1.push_back(
					metabolomics_data.calculateMAR(metabolomics_data.metabolomicsData_.at(pre_samples[pairs_iter]),
						metabolomics_data.biochemicalReactions_.at(mar)));
				samples2.push_back(
					metabolomics_data.calculateMAR(metabolomics_data.metabolomicsData_.at(post_samples[pairs_iter]),
						metabolomics_data.biochemicalReactions_.at(mar)));
			}

			// calculate the moments and fold change
			float ave1, adev1, sdev1, var1, skew1, curt1;
			moment(&samples1[0], n_samples, ave1, adev1, sdev1, var1, skew1, curt1);
			float ave2, adev2, sdev2, var2, skew2, curt2;
			moment(&samples2[0], n_samples, ave2, adev2, sdev2, var2, skew2, curt2);
			pw_stats.fold_change = std::log2(ave2 / ave1);

			// calculate the 95% CI
			pw_stats.confidence_interval_1 = confidence(samples1, alpha);
			pw_stats.confidence_interval_2 = confidence(samples2, alpha);

			//// calculate the K-S prob
			//float d, prob;
			//kstwo(&samples1[0], n_samples, &samples2[0], n_samples, d, prob);
			//pw_stats.prob = prob;

			//if (prob < 0.05) {
			if ((pw_stats.confidence_interval_1.first > pw_stats.confidence_interval_2.second
				|| pw_stats.confidence_interval_1.second < pw_stats.confidence_interval_2.first)
				&& (pw_stats.fold_change > fc || pw_stats.fold_change < -fc)) {
				pw_stats.is_significant = true;
				std::vector<PWStats> pw_stats_vec = { pw_stats };
				auto found = pw_data.emplace(mar, pw_stats_vec);
				if (!found.second) {
					pw_data.at(mar).push_back(pw_stats);
				}
			}
		}
	}
	return pw_data;
}

/*
@brief Find significant pair-wise MARS between pre/post samples (one pre vs one post)
*/
PWData PWPrePostDifference(BiochemicalReactionModel<float>& metabolomics_data,
	std::vector<std::string>& pre_samples, std::vector<std::string>& post_samples, const int& n_pairs,
	int n_samples = 10000, float alpha = 0.05, float fc = 0.43229) {

	PWData pw_data;
	for (const std::string& mar : metabolomics_data.reaction_ids_) {
		for (size_t pairs_iter1 = 0; pairs_iter1 < n_pairs; ++pairs_iter1) {

			std::string sample_name_1 = post_samples[pairs_iter1] + "-" + pre_samples[pairs_iter1];

			// sample the MAR data
			std::vector<float> samples1;
			for (int sample_iter = 0; sample_iter < n_samples; ++sample_iter) {
				float s1 = metabolomics_data.calculateMAR(metabolomics_data.metabolomicsData_.at(pre_samples[pairs_iter1]),
					metabolomics_data.biochemicalReactions_.at(mar));
				float s2 = metabolomics_data.calculateMAR(metabolomics_data.metabolomicsData_.at(post_samples[pairs_iter1]),
					metabolomics_data.biochemicalReactions_.at(mar));
				samples1.push_back(s2 - s1);
			}

			// calculate the moments and fold change
			float ave1, adev1, sdev1, var1, skew1, curt1;
			moment(&samples1[0], n_samples, ave1, adev1, sdev1, var1, skew1, curt1);

			// calculate the 95% CI
			std::pair<float, float> confidence_interval_1 = confidence(samples1, alpha);

			for (size_t pairs_iter2 = pairs_iter1 + 1; pairs_iter2 < n_pairs; ++pairs_iter2) {
				std::cout << "MAR: " << mar << " Pair1: " << pairs_iter1 << " Pair2: " << pairs_iter2 << std::endl;

				std::string sample_name_2 = post_samples[pairs_iter2] + "-" + pre_samples[pairs_iter2];

				// initialize the data struct
				PWStats pw_stats;
				pw_stats.feature_name = mar;
				pw_stats.feature_comment = metabolomics_data.biochemicalReactions_.at(mar).equation;
				pw_stats.sample_name_1 = sample_name_1;
				pw_stats.sample_name_2 = sample_name_2;
				pw_stats.n1 = n_samples;
				pw_stats.n2 = n_samples;

				// sample the MAR data
				std::vector<float> samples2;
				for (int sample_iter = 0; sample_iter < n_samples; ++sample_iter) {
					float s1 = metabolomics_data.calculateMAR(metabolomics_data.metabolomicsData_.at(pre_samples[pairs_iter2]),
						metabolomics_data.biochemicalReactions_.at(mar));
					float s2 = metabolomics_data.calculateMAR(metabolomics_data.metabolomicsData_.at(post_samples[pairs_iter2]),
						metabolomics_data.biochemicalReactions_.at(mar));
					samples2.push_back(s2 - s1);
				}

				// calculate the moments and fold change
				float ave2, adev2, sdev2, var2, skew2, curt2;
				moment(&samples2[0], n_samples, ave2, adev2, sdev2, var2, skew2, curt2);

				// calculate the 95% CI
				std::pair<float, float> confidence_interval_2 = confidence(samples2, alpha);

				// calculate the normalized geometric fold change
				pw_stats.fold_change = std::log(std::exp(ave2) / std::exp(ave1)) / (std::log(std::exp(ave2) + std::exp(ave1)));

				pw_stats.confidence_interval_1 = confidence_interval_1;
				pw_stats.confidence_interval_2 = confidence_interval_2;

				//if (prob < 0.05) {
				if ((pw_stats.confidence_interval_1.first > pw_stats.confidence_interval_2.second
					|| pw_stats.confidence_interval_1.second < pw_stats.confidence_interval_2.first)
					&& (pw_stats.fold_change > fc || pw_stats.fold_change < -fc)) {
					pw_stats.is_significant = true;
					std::vector<PWStats> pw_stats_vec = { pw_stats };
					auto found = pw_data.emplace(mar, pw_stats_vec);
					if (!found.second) {
						pw_data.at(mar).push_back(pw_stats);
					}
				}
			}
		}
	}
	return pw_data;
}

void PWSummary(const PWData& pw_data, PWSampleSummaries& pw_sample_summaries, PWFeatureSummaries& pw_feature_summaries, PWTotalSummary& pw_total_summary) {

	std::map<std::string, PWSampleSummary> pw_sample_summary_map;
	std::map<std::string, PWFeatureSummary> pw_feature_summary_map;
	for (const auto& pw_datum : pw_data) {
		for (const auto& pw_stats : pw_datum.second) {
			if (!pw_stats.is_significant) continue;

			// Samples
			PWSampleSummary pw_sample_summary;
			pw_sample_summary.sample_name_1 = pw_stats.sample_name_1;
			pw_sample_summary.sample_name_2 = pw_stats.sample_name_2;
			pw_sample_summary.n_significant = 1;
			std::string key = pw_stats.sample_name_1 + "_vs_" + pw_stats.sample_name_2;
			auto found_samples = pw_sample_summary_map.emplace(key, pw_sample_summary);
			if (!found_samples.second) {
				pw_sample_summary_map.at(key).n_significant += 1;
			}

			// Features
			PWFeatureSummary pw_feature_summary;
			pw_feature_summary.feature_name = pw_stats.feature_name;
			pw_feature_summary.n_significant = 1;
			auto found_features = pw_feature_summary_map.emplace(pw_stats.feature_name, pw_feature_summary);
			if (!found_features.second) {
				pw_feature_summary_map.at(pw_stats.feature_name).n_significant += 1;
			}

			// Totals
			pw_total_summary.n_significant_total += 1;
			pw_total_summary.significant_features.insert(pw_stats.feature_name);
			pw_total_summary.significant_pairs.insert(key);
		}
	}
	// Samples
	for (const auto& map : pw_sample_summary_map)
		pw_sample_summaries.push_back(map.second);
	std::sort(pw_sample_summaries.begin(), pw_sample_summaries.end(),
		[](const PWSampleSummary& a, const PWSampleSummary& b)
	{
		return a.sample_name_2 < b.sample_name_2;
	});
	std::sort(pw_sample_summaries.begin(), pw_sample_summaries.end(),
		[](const PWSampleSummary& a, const PWSampleSummary& b)
	{
		return a.sample_name_1 < b.sample_name_1;
	});

	// Features
	for (const auto& map : pw_feature_summary_map)
		pw_feature_summaries.push_back(map.second);
	std::sort(pw_feature_summaries.begin(), pw_feature_summaries.end(),
		[](const PWFeatureSummary& a, const PWFeatureSummary& b)
	{
		return a.feature_name < b.feature_name;
	});

	// Totals
	pw_total_summary.n_significant_features = (int)pw_total_summary.significant_features.size();
	pw_total_summary.n_significant_pairs = (int)pw_total_summary.significant_pairs.size();
}
bool WritePWData(const std::string& filename, const PWData& pw_data) {

	// Export the results to file
	CSVWriter csvwriter(filename);
	std::vector<std::string> headers = { "Feature", "FeatureComment", "Sample1", "Sample2", "LB1", "LB2", "UB1", "UB2", "Log2(FC)" };
	csvwriter.writeDataInRow(headers.begin(), headers.end());
	for (const auto& pw_datum : pw_data) {
		for (const auto& pw_stats : pw_datum.second) {
			std::vector<std::string> line;
			line.push_back(pw_stats.feature_name);
			line.push_back(pw_stats.feature_comment);
			line.push_back(pw_stats.sample_name_1);
			line.push_back(pw_stats.sample_name_2);
			line.push_back(std::to_string(pw_stats.confidence_interval_1.first));
			line.push_back(std::to_string(pw_stats.confidence_interval_2.first));
			line.push_back(std::to_string(pw_stats.confidence_interval_1.second));
			line.push_back(std::to_string(pw_stats.confidence_interval_2.second));
			line.push_back(std::to_string(pw_stats.fold_change));
			csvwriter.writeDataInRow(line.begin(), line.end());
		}
	}
	return true;
}
bool ReadPWData(const std::string& filename, PWData& pw_data) {
	io::CSVReader<8> data_in(filename);
	data_in.read_header(io::ignore_extra_column,
		"Feature", "Sample1", "Sample2", "LB1", "LB2", "UB1", "UB2", "Log2(FC)");
	std::string feature_str, sample_1_str, sample_2_str, lb1_str, lb2_str, ub1_str, ub2_str, log2fc_str;

	while (data_in.read_row(feature_str, sample_1_str, sample_2_str, lb1_str, lb2_str, ub1_str, ub2_str, log2fc_str))
	{
		// parse the .csv file
		PWStats pw_stats;
		pw_stats.feature_name = feature_str;
		pw_stats.sample_name_1 = sample_1_str;
		pw_stats.sample_name_2 = sample_2_str;
		pw_stats.confidence_interval_1 = std::make_pair(std::stof(lb1_str), std::stof(ub1_str));
		pw_stats.confidence_interval_2 = std::make_pair(std::stof(lb2_str), std::stof(ub2_str));
		pw_stats.fold_change = std::stof(log2fc_str);
		pw_stats.is_significant = true;

		std::vector<PWStats> pw_stats_vec = { pw_stats };
		auto found = pw_data.emplace(feature_str, pw_stats_vec);
		if (!found.second) {
			pw_data.at(feature_str).push_back(pw_stats);
		}
	}
	return true;
}
bool WritePWSampleSummaries(const std::string& filename, const PWSampleSummaries& pw_sample_summaries) {

	// Export the results to file
	CSVWriter csvwriter(filename);
	std::vector<std::string> headers = { "Sample1", "Sample2", "Sig_pairs" };
	csvwriter.writeDataInRow(headers.begin(), headers.end());
	for (const auto& pw_sample_summary : pw_sample_summaries) {
		std::vector<std::string> line;
		line.push_back(pw_sample_summary.sample_name_1);
		line.push_back(pw_sample_summary.sample_name_2);
		line.push_back(std::to_string(pw_sample_summary.n_significant));
		csvwriter.writeDataInRow(line.begin(), line.end());
	}
	return true;
}
bool WritePWFeatureSummaries(const std::string& filename, const PWFeatureSummaries& pw_feature_summaries) {

	// Export the results to file
	CSVWriter csvwriter(filename);
	std::vector<std::string> headers = { "Feature", "Sig_features" };
	csvwriter.writeDataInRow(headers.begin(), headers.end());
	for (const auto& pw_feature_summary : pw_feature_summaries) {
		std::vector<std::string> line;
		line.push_back(pw_feature_summary.feature_name);
		line.push_back(std::to_string(pw_feature_summary.n_significant));
		csvwriter.writeDataInRow(line.begin(), line.end());
	}
	return true;
}