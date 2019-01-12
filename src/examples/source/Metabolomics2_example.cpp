/**TODO:  Add copyright*/

#include "Metabolomics_example.h"

using namespace SmartPeak;

/*
@brief Example using intracellular E. coli metabolomics data
	taken from re-grown glycerol stock solutions on Glucose M9 at mid-exponential phase
	from adaptive laboratory evolution (ALE) experiments following gene knockout (KO)
*/

// Scripts to run
void main_statistics_timecourseSummary(
	bool run_timeCourse_Ref = false, bool run_timeCourse_Gnd = false, bool run_timeCourse_SdhCB = false, bool run_timeCourse_Pgi = false, bool run_timeCourse_PtsHIcrr = false,
	bool run_timeCourse_TpiA = false)
{
	// define the data simulator
	MetDataSimClassification<float> metabolomics_data;

	// data dirs
	std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
	//std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
	//std::string data_dir = "/home/user/Data/";

	std::string 
		timeCourse_Ref_filename, timeCourse_Gnd_filename, timeCourse_SdhCB_filename, timeCourse_Pgi_filename, timeCourse_PtsHIcrr_filename,
		timeCourse_TpiA_filename, 
		timeCourseSampleSummary_Ref_filename, timeCourseSampleSummary_Gnd_filename, timeCourseSampleSummary_SdhCB_filename, timeCourseSampleSummary_Pgi_filename, timeCourseSampleSummary_PtsHIcrr_filename,
		timeCourseSampleSummary_TpiA_filename, 
		timeCourseFeatureSummary_Ref_filename, timeCourseFeatureSummary_Gnd_filename, timeCourseFeatureSummary_SdhCB_filename, timeCourseFeatureSummary_Pgi_filename, timeCourseFeatureSummary_PtsHIcrr_filename,
		timeCourseFeatureSummary_TpiA_filename;

	// filenames
	timeCourse_Ref_filename = data_dir + "EColi_timeCourse_Ref.csv";
	timeCourse_Gnd_filename = data_dir + "EColi_timeCourse_Gnd.csv";
	timeCourse_SdhCB_filename = data_dir + "EColi_timeCourse_SdhCB.csv";
	timeCourse_Pgi_filename = data_dir + "EColi_timeCourse_Pgi.csv";
	timeCourse_PtsHIcrr_filename = data_dir + "EColi_timeCourse_PtsHIcrr.csv";
	timeCourse_TpiA_filename = data_dir + "EColi_timeCourse_TpiA.csv";
	timeCourseSampleSummary_Ref_filename = data_dir + "EColi_timeCourseSampleSummary_Ref.csv";
	timeCourseSampleSummary_Gnd_filename = data_dir + "EColi_timeCourseSampleSummary_Gnd.csv";
	timeCourseSampleSummary_SdhCB_filename = data_dir + "EColi_timeCourseSampleSummary_SdhCB.csv";
	timeCourseSampleSummary_Pgi_filename = data_dir + "EColi_timeCourseSampleSummary_Pgi.csv";
	timeCourseSampleSummary_PtsHIcrr_filename = data_dir + "EColi_timeCourseSampleSummary_PtsHIcrr.csv";
	timeCourseSampleSummary_TpiA_filename = data_dir + "EColi_timeCourseSampleSummary_TpiA.csv";
	timeCourseFeatureSummary_Ref_filename = data_dir + "EColi_timeCourseFeatureSummary_Ref.csv";
	timeCourseFeatureSummary_Gnd_filename = data_dir + "EColi_timeCourseFeatureSummary_Gnd.csv";
	timeCourseFeatureSummary_SdhCB_filename = data_dir + "EColi_timeCourseFeatureSummary_SdhCB.csv";
	timeCourseFeatureSummary_Pgi_filename = data_dir + "EColi_timeCourseFeatureSummary_Pgi.csv";
	timeCourseFeatureSummary_PtsHIcrr_filename = data_dir + "EColi_timeCourseFeatureSummary_PtsHIcrr.csv";
	timeCourseFeatureSummary_TpiA_filename = data_dir + "EColi_timeCourseFeatureSummary_TpiA.csv";

	if (run_timeCourse_Ref) {
		// Read in the data
		PWData timeCourseRef;
		ReadPWData(timeCourse_Ref_filename, timeCourseRef);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseRef, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_Ref_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_Ref_filename, pw_feature_summaries);
	}

	if (run_timeCourse_Gnd) {
		// Read in the data
		PWData timeCourseGnd;
		ReadPWData(timeCourse_Gnd_filename, timeCourseGnd);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseGnd, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_Gnd_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_Gnd_filename, pw_feature_summaries);
	}

	if (run_timeCourse_SdhCB) {
		// Read in the data
		PWData timeCourseSdhCB;
		ReadPWData(timeCourse_SdhCB_filename, timeCourseSdhCB);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseSdhCB, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_SdhCB_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_SdhCB_filename, pw_feature_summaries);
	}

	if (run_timeCourse_Pgi) {
		// Read in the data
		PWData timeCoursePgi;
		ReadPWData(timeCourse_Pgi_filename, timeCoursePgi);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCoursePgi, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_Pgi_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_Pgi_filename, pw_feature_summaries);
	}

	if (run_timeCourse_PtsHIcrr) {
		// Read in the data
		PWData timeCoursePtsHIcrr;
		ReadPWData(timeCourse_PtsHIcrr_filename, timeCoursePtsHIcrr);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCoursePtsHIcrr, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_PtsHIcrr_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_PtsHIcrr_filename, pw_feature_summaries);
	}

	if (run_timeCourse_TpiA) {
		// Read in the data
		PWData timeCourseTpiA;
		ReadPWData(timeCourse_TpiA_filename, timeCourseTpiA);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseTpiA, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_TpiA_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_TpiA_filename, pw_feature_summaries);
	}
}
void main_statistics_timecourse(
	bool run_timeCourse_Ref = false, bool run_timeCourse_Gnd = false, bool run_timeCourse_SdhCB = false, bool run_timeCourse_Pgi = false, bool run_timeCourse_PtsHIcrr = false,
	bool run_timeCourse_TpiA = false)
{
	// define the data simulator
	MetDataSimClassification<float> metabolomics_data;

	// data dirs
	//std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
	std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
	//std::string data_dir = "/home/user/Data/";

	std::string biochem_rxns_filename, metabo_data_filename, meta_data_filename,
		timeCourse_Ref_filename, timeCourse_Gnd_filename, timeCourse_SdhCB_filename, timeCourse_Pgi_filename, timeCourse_PtsHIcrr_filename,
		timeCourse_TpiA_filename;
	std::vector<std::string> pre_samples,
		timeCourse_Ref_samples, timeCourse_Gnd_samples, timeCourse_SdhCB_samples, timeCourse_Pgi_samples, timeCourse_PtsHIcrr_samples,
		timeCourse_TpiA_samples;
	// filenames
	biochem_rxns_filename = data_dir + "iJO1366.csv";
	metabo_data_filename = data_dir + "MetabolomicsData_EColi.csv";
	meta_data_filename = data_dir + "MetaData_prePost_EColi.csv";
	timeCourse_Ref_filename = data_dir + "EColi_timeCourse_Ref.csv";
	timeCourse_Gnd_filename = data_dir + "EColi_timeCourse_Gnd.csv";
	timeCourse_SdhCB_filename = data_dir + "EColi_timeCourse_SdhCB.csv";
	timeCourse_Pgi_filename = data_dir + "EColi_timeCourse_Pgi.csv";
	timeCourse_PtsHIcrr_filename = data_dir + "EColi_timeCourse_PtsHIcrr.csv";
	timeCourse_TpiA_filename = data_dir + "EColi_timeCourse_TpiA.csv";
	timeCourse_Ref_samples = { "Evo04-10", "Evo04-11", "Evo04-12", "Evo04-4", "Evo04-5", "Evo04-6", "Evo04Evo01EP-1", "Evo04Evo01EP-2", "Evo04Evo01EP-3", "Evo04Evo01EP-4", "Evo04Evo01EP-5", "Evo04Evo01EP-6", "Evo04Evo02EP-1", "Evo04Evo02EP-2", "Evo04Evo02EP-3", "Evo04Evo02EP-4", "Evo04Evo02EP-5", "Evo04Evo02EP-6" };
	timeCourse_Gnd_samples = { "Evo04-10", "Evo04-11", "Evo04-12", "Evo04-4", "Evo04-5", "Evo04-6", "Evo04gnd-1", "Evo04gnd-2", "Evo04gnd-3", "Evo04gnd-7", "Evo04gnd-8", "Evo04gnd-9", "Evo04gndEvo01EP-1", "Evo04gndEvo01EP-2", "Evo04gndEvo01EP-3", "Evo04gndEvo01EP-4", "Evo04gndEvo01EP-5", "Evo04gndEvo01EP-6", "Evo04gndEvo02EP-1", "Evo04gndEvo02EP-2", "Evo04gndEvo02EP-3", "Evo04gndEvo02EP-4", "Evo04gndEvo02EP-5", "Evo04gndEvo02EP-6", "Evo04gndEvo03EP-1", "Evo04gndEvo03EP-10", "Evo04gndEvo03EP-2", "Evo04gndEvo03EP-3", "Evo04gndEvo03EP-4", "Evo04gndEvo03EP-5", "Evo04gndEvo03EP-6", "Evo04gndEvo03EP-7", "Evo04gndEvo03EP-8", "Evo04gndEvo03EP-9" };
	timeCourse_SdhCB_samples = { "Evo04-10", "Evo04-11", "Evo04-12", "Evo04-4", "Evo04-5", "Evo04-6", "Evo04sdhCB-1", "Evo04sdhCB-2", "Evo04sdhCB-3", "Evo04sdhCB-4", "Evo04sdhCB-5", "Evo04sdhCB-6", "Evo04sdhCBEvo01EP-1", "Evo04sdhCBEvo01EP-2", "Evo04sdhCBEvo01EP-3", "Evo04sdhCBEvo01EP-4", "Evo04sdhCBEvo01EP-5", "Evo04sdhCBEvo01EP-6", "Evo04sdhCBEvo02EP-1", "Evo04sdhCBEvo02EP-2", "Evo04sdhCBEvo02EP-3", "Evo04sdhCBEvo02EP-4", "Evo04sdhCBEvo02EP-5", "Evo04sdhCBEvo02EP-6", "Evo04sdhCBEvo03EP-1", "Evo04sdhCBEvo03EP-2", "Evo04sdhCBEvo03EP-3", "Evo04sdhCBEvo03EP-4", "Evo04sdhCBEvo03EP-5", "Evo04sdhCBEvo03EP-6" };
	timeCourse_Pgi_samples = { "Evo04-10", "Evo04-11", "Evo04-12", "Evo04-4", "Evo04-5", "Evo04-6", "Evo04pgi-1", "Evo04pgi-2", "Evo04pgi-3", "Evo04pgi-4", "Evo04pgi-5", "Evo04pgi-6", "Evo04pgiEvo01EP-1", "Evo04pgiEvo01EP-2", "Evo04pgiEvo01EP-3", "Evo04pgiEvo01EP-4", "Evo04pgiEvo01EP-5", "Evo04pgiEvo01EP-6", "Evo04pgiEvo01J01-1", "Evo04pgiEvo01J01-2", "Evo04pgiEvo01J01-3", "Evo04pgiEvo01J01-4", "Evo04pgiEvo01J01-5", "Evo04pgiEvo01J01-6", "Evo04pgiEvo01J02-1", "Evo04pgiEvo01J02-2", "Evo04pgiEvo01J02-3", "Evo04pgiEvo01J02-4", "Evo04pgiEvo01J02-5", "Evo04pgiEvo01J02-6", "Evo04pgiEvo02EP-1", "Evo04pgiEvo02EP-2", "Evo04pgiEvo02EP-3", "Evo04pgiEvo02EP-4", "Evo04pgiEvo02EP-5", "Evo04pgiEvo02EP-6", "Evo04pgiEvo02J01-1", "Evo04pgiEvo02J01-2", "Evo04pgiEvo02J01-3", "Evo04pgiEvo02J01-4", "Evo04pgiEvo02J01-5", "Evo04pgiEvo02J01-6", "Evo04pgiEvo02J02-1", "Evo04pgiEvo02J02-2", "Evo04pgiEvo02J02-3", "Evo04pgiEvo02J02-4", "Evo04pgiEvo02J02-5", "Evo04pgiEvo02J02-6", "Evo04pgiEvo02J03-1", "Evo04pgiEvo02J03-2", "Evo04pgiEvo02J03-3", "Evo04pgiEvo02J03-4", "Evo04pgiEvo02J03-5", "Evo04pgiEvo02J03-6", "Evo04pgiEvo03EP-1", "Evo04pgiEvo03EP-2", "Evo04pgiEvo03EP-3", "Evo04pgiEvo03EP-4", "Evo04pgiEvo03EP-5", "Evo04pgiEvo03EP-6", "Evo04pgiEvo03J01-1", "Evo04pgiEvo03J01-2", "Evo04pgiEvo03J01-3", "Evo04pgiEvo03J01-4", "Evo04pgiEvo03J01-5", "Evo04pgiEvo03J01-6", "Evo04pgiEvo03J02-1", "Evo04pgiEvo03J02-2", "Evo04pgiEvo03J02-3", "Evo04pgiEvo03J02-4", "Evo04pgiEvo03J02-5", "Evo04pgiEvo03J02-6", "Evo04pgiEvo03J03-1", "Evo04pgiEvo03J03-2", "Evo04pgiEvo03J03-3", "Evo04pgiEvo03J03-4", "Evo04pgiEvo03J03-5", "Evo04pgiEvo03J03-6", "Evo04pgiEvo04EP-1", "Evo04pgiEvo04EP-2", "Evo04pgiEvo04EP-3", "Evo04pgiEvo04EP-4", "Evo04pgiEvo04EP-5", "Evo04pgiEvo04EP-6", "Evo04pgiEvo04J01-1", "Evo04pgiEvo04J01-2", "Evo04pgiEvo04J01-3", "Evo04pgiEvo04J01-4", "Evo04pgiEvo04J01-5", "Evo04pgiEvo04J01-6", "Evo04pgiEvo04J02-1", "Evo04pgiEvo04J02-2", "Evo04pgiEvo04J02-3", "Evo04pgiEvo04J02-4", "Evo04pgiEvo04J02-5", "Evo04pgiEvo04J02-6", "Evo04pgiEvo04J03-1", "Evo04pgiEvo04J03-2", "Evo04pgiEvo04J03-3", "Evo04pgiEvo04J03-4", "Evo04pgiEvo04J03-5", "Evo04pgiEvo04J03-6", "Evo04pgiEvo05EP-1", "Evo04pgiEvo05EP-2", "Evo04pgiEvo05EP-3", "Evo04pgiEvo05EP-4", "Evo04pgiEvo05EP-5", "Evo04pgiEvo05EP-6", "Evo04pgiEvo05J01-1", "Evo04pgiEvo05J01-2", "Evo04pgiEvo05J01-3", "Evo04pgiEvo05J01-4", "Evo04pgiEvo05J01-5", "Evo04pgiEvo05J01-6", "Evo04pgiEvo05J02-1", "Evo04pgiEvo05J02-2", "Evo04pgiEvo05J02-3", "Evo04pgiEvo05J02-4", "Evo04pgiEvo05J02-5", "Evo04pgiEvo05J02-6", "Evo04pgiEvo05J03-1", "Evo04pgiEvo05J03-2", "Evo04pgiEvo05J03-3", "Evo04pgiEvo05J03-4", "Evo04pgiEvo05J03-5", "Evo04pgiEvo05J03-6", "Evo04pgiEvo06EP-1", "Evo04pgiEvo06EP-2", "Evo04pgiEvo06EP-3", "Evo04pgiEvo06EP-4", "Evo04pgiEvo06EP-5", "Evo04pgiEvo06EP-6", "Evo04pgiEvo06J01-1", "Evo04pgiEvo06J01-2", "Evo04pgiEvo06J01-3", "Evo04pgiEvo06J01-4", "Evo04pgiEvo06J01-5", "Evo04pgiEvo06J01-6", "Evo04pgiEvo06J02-1", "Evo04pgiEvo06J02-2", "Evo04pgiEvo06J02-3", "Evo04pgiEvo06J02-4", "Evo04pgiEvo06J02-5", "Evo04pgiEvo06J02-6", "Evo04pgiEvo06J03-1", "Evo04pgiEvo06J03-2", "Evo04pgiEvo06J03-3", "Evo04pgiEvo06J03-4", "Evo04pgiEvo06J03-5", "Evo04pgiEvo06J03-6", "Evo04pgiEvo07EP-1", "Evo04pgiEvo07EP-2", "Evo04pgiEvo07EP-3", "Evo04pgiEvo07EP-4", "Evo04pgiEvo07EP-5", "Evo04pgiEvo07EP-6", "Evo04pgiEvo07J01-1", "Evo04pgiEvo07J01-2", "Evo04pgiEvo07J01-3", "Evo04pgiEvo07J01-4", "Evo04pgiEvo07J01-5", "Evo04pgiEvo07J01-6", "Evo04pgiEvo07J02-1", "Evo04pgiEvo07J02-2", "Evo04pgiEvo07J02-3", "Evo04pgiEvo07J02-4", "Evo04pgiEvo07J02-5", "Evo04pgiEvo07J02-6", "Evo04pgiEvo07J03-1", "Evo04pgiEvo07J03-2", "Evo04pgiEvo07J03-3", "Evo04pgiEvo07J03-4", "Evo04pgiEvo07J03-5", "Evo04pgiEvo07J03-6", "Evo04pgiEvo08EP-1", "Evo04pgiEvo08EP-2", "Evo04pgiEvo08EP-3", "Evo04pgiEvo08EP-4", "Evo04pgiEvo08EP-5", "Evo04pgiEvo08EP-6", "Evo04pgiEvo08J01-1", "Evo04pgiEvo08J01-2", "Evo04pgiEvo08J01-3", "Evo04pgiEvo08J01-4", "Evo04pgiEvo08J01-5", "Evo04pgiEvo08J01-6", "Evo04pgiEvo08J02-1", "Evo04pgiEvo08J02-2", "Evo04pgiEvo08J02-3", "Evo04pgiEvo08J02-4", "Evo04pgiEvo08J02-5", "Evo04pgiEvo08J02-6", "Evo04pgiEvo08J03-1", "Evo04pgiEvo08J03-2", "Evo04pgiEvo08J03-3", "Evo04pgiEvo08J03-4", "Evo04pgiEvo08J03-5", "Evo04pgiEvo08J03-6" };
	timeCourse_PtsHIcrr_samples = { "Evo04-10", "Evo04-11", "Evo04-12", "Evo04-4", "Evo04-5", "Evo04-6", "Evo04ptsHIcrr-1", "Evo04ptsHIcrr-2", "Evo04ptsHIcrr-3", "Evo04ptsHIcrr-4", "Evo04ptsHIcrr-5", "Evo04ptsHIcrr-6", "Evo04ptsHIcrrEvo01EP-1", "Evo04ptsHIcrrEvo01EP-2", "Evo04ptsHIcrrEvo01EP-3", "Evo04ptsHIcrrEvo01EP-4", "Evo04ptsHIcrrEvo01EP-5", "Evo04ptsHIcrrEvo01EP-6", "Evo04ptsHIcrrEvo01J01-1", "Evo04ptsHIcrrEvo01J01-2", "Evo04ptsHIcrrEvo01J01-3", "Evo04ptsHIcrrEvo01J01-4", "Evo04ptsHIcrrEvo01J01-5", "Evo04ptsHIcrrEvo01J01-6", "Evo04ptsHIcrrEvo01J03-1", "Evo04ptsHIcrrEvo01J03-2", "Evo04ptsHIcrrEvo01J03-3", "Evo04ptsHIcrrEvo01J03-4", "Evo04ptsHIcrrEvo01J03-5", "Evo04ptsHIcrrEvo01J03-6", "Evo04ptsHIcrrEvo02EP-1", "Evo04ptsHIcrrEvo02EP-2", "Evo04ptsHIcrrEvo02EP-3", "Evo04ptsHIcrrEvo02EP-4", "Evo04ptsHIcrrEvo02EP-5", "Evo04ptsHIcrrEvo02EP-6", "Evo04ptsHIcrrEvo02J01-1", "Evo04ptsHIcrrEvo02J01-2", "Evo04ptsHIcrrEvo02J01-3", "Evo04ptsHIcrrEvo02J01-4", "Evo04ptsHIcrrEvo02J01-5", "Evo04ptsHIcrrEvo02J01-6", "Evo04ptsHIcrrEvo02J03-1", "Evo04ptsHIcrrEvo02J03-2", "Evo04ptsHIcrrEvo02J03-3", "Evo04ptsHIcrrEvo02J03-4", "Evo04ptsHIcrrEvo02J03-5", "Evo04ptsHIcrrEvo02J03-6", "Evo04ptsHIcrrEvo03EP-1", "Evo04ptsHIcrrEvo03EP-2", "Evo04ptsHIcrrEvo03EP-3", "Evo04ptsHIcrrEvo03EP-4", "Evo04ptsHIcrrEvo03EP-5", "Evo04ptsHIcrrEvo03EP-6", "Evo04ptsHIcrrEvo03J01-1", "Evo04ptsHIcrrEvo03J01-2", "Evo04ptsHIcrrEvo03J01-3", "Evo04ptsHIcrrEvo03J01-4", "Evo04ptsHIcrrEvo03J01-5", "Evo04ptsHIcrrEvo03J01-6", "Evo04ptsHIcrrEvo03J03-1", "Evo04ptsHIcrrEvo03J03-2", "Evo04ptsHIcrrEvo03J03-3", "Evo04ptsHIcrrEvo03J03-4", "Evo04ptsHIcrrEvo03J03-5", "Evo04ptsHIcrrEvo03J03-6", "Evo04ptsHIcrrEvo03J04-1", "Evo04ptsHIcrrEvo03J04-2", "Evo04ptsHIcrrEvo03J04-3", "Evo04ptsHIcrrEvo03J04-4", "Evo04ptsHIcrrEvo03J04-5", "Evo04ptsHIcrrEvo03J04-6", "Evo04ptsHIcrrEvo04EP-1", "Evo04ptsHIcrrEvo04EP-2", "Evo04ptsHIcrrEvo04EP-3", "Evo04ptsHIcrrEvo04EP-4", "Evo04ptsHIcrrEvo04EP-5", "Evo04ptsHIcrrEvo04EP-6", "Evo04ptsHIcrrEvo04J01-1", "Evo04ptsHIcrrEvo04J01-2", "Evo04ptsHIcrrEvo04J01-3", "Evo04ptsHIcrrEvo04J01-4", "Evo04ptsHIcrrEvo04J01-5", "Evo04ptsHIcrrEvo04J01-6", "Evo04ptsHIcrrEvo04J03-1", "Evo04ptsHIcrrEvo04J03-2", "Evo04ptsHIcrrEvo04J03-3", "Evo04ptsHIcrrEvo04J03-4", "Evo04ptsHIcrrEvo04J03-5", "Evo04ptsHIcrrEvo04J03-6", "Evo04ptsHIcrrEvo04J04-1", "Evo04ptsHIcrrEvo04J04-2", "Evo04ptsHIcrrEvo04J04-3", "Evo04ptsHIcrrEvo04J04-4", "Evo04ptsHIcrrEvo04J04-5", "Evo04ptsHIcrrEvo04J04-6" };
	timeCourse_TpiA_samples = { "Evo04-10", "Evo04-11", "Evo04-12", "Evo04-4", "Evo04-5", "Evo04-6", "Evo04tpiA-1", "Evo04tpiA-2", "Evo04tpiA-3", "Evo04tpiA-4", "Evo04tpiA-5", "Evo04tpiA-6", "Evo04tpiAEvo01EP-1", "Evo04tpiAEvo01EP-2", "Evo04tpiAEvo01EP-3", "Evo04tpiAEvo01EP-4", "Evo04tpiAEvo01EP-5", "Evo04tpiAEvo01EP-6", "Evo04tpiAEvo01J01-1", "Evo04tpiAEvo01J01-2", "Evo04tpiAEvo01J01-3", "Evo04tpiAEvo01J01-4", "Evo04tpiAEvo01J01-5", "Evo04tpiAEvo01J01-6", "Evo04tpiAEvo01J03-1", "Evo04tpiAEvo01J03-2", "Evo04tpiAEvo01J03-3", "Evo04tpiAEvo01J03-4", "Evo04tpiAEvo01J03-5", "Evo04tpiAEvo01J03-6", "Evo04tpiAEvo02EP-1", "Evo04tpiAEvo02EP-2", "Evo04tpiAEvo02EP-3", "Evo04tpiAEvo02EP-4", "Evo04tpiAEvo02EP-5", "Evo04tpiAEvo02EP-6", "Evo04tpiAEvo02J01-1", "Evo04tpiAEvo02J01-2", "Evo04tpiAEvo02J01-3", "Evo04tpiAEvo02J01-4", "Evo04tpiAEvo02J01-5", "Evo04tpiAEvo02J01-6", "Evo04tpiAEvo02J03-1", "Evo04tpiAEvo02J03-2", "Evo04tpiAEvo02J03-3", "Evo04tpiAEvo02J03-4", "Evo04tpiAEvo02J03-5", "Evo04tpiAEvo02J03-6", "Evo04tpiAEvo03EP-1", "Evo04tpiAEvo03EP-2", "Evo04tpiAEvo03EP-3", "Evo04tpiAEvo03EP-4", "Evo04tpiAEvo03EP-5", "Evo04tpiAEvo03EP-6", "Evo04tpiAEvo03J01-1", "Evo04tpiAEvo03J01-2", "Evo04tpiAEvo03J01-3", "Evo04tpiAEvo03J01-4", "Evo04tpiAEvo03J01-5", "Evo04tpiAEvo03J01-6", "Evo04tpiAEvo03J03-1", "Evo04tpiAEvo03J03-2", "Evo04tpiAEvo03J03-3", "Evo04tpiAEvo03J03-4", "Evo04tpiAEvo03J03-5", "Evo04tpiAEvo03J03-6", "Evo04tpiAEvo04EP-1", "Evo04tpiAEvo04EP-2", "Evo04tpiAEvo04EP-3", "Evo04tpiAEvo04EP-4", "Evo04tpiAEvo04EP-5", "Evo04tpiAEvo04EP-6", "Evo04tpiAEvo04J01-1", "Evo04tpiAEvo04J01-2", "Evo04tpiAEvo04J01-3", "Evo04tpiAEvo04J01-4", "Evo04tpiAEvo04J01-5", "Evo04tpiAEvo04J01-6", "Evo04tpiAEvo04J03-1", "Evo04tpiAEvo04J03-2", "Evo04tpiAEvo04J03-3", "Evo04tpiAEvo04J03-4", "Evo04tpiAEvo04J03-5", "Evo04tpiAEvo04J03-6" };

	// read in the data
	metabolomics_data.readBiochemicalReactions(biochem_rxns_filename);
	metabolomics_data.readMetabolomicsData(metabo_data_filename);
	metabolomics_data.readMetaData(meta_data_filename);
	metabolomics_data.findComponentGroupNames();
	metabolomics_data.findMARs();
	metabolomics_data.findMARs(true, false);
	metabolomics_data.findMARs(false, true);
	metabolomics_data.findLabels();

	if (run_timeCourse_Ref) {
		// Find significant pair-wise MARS between each sample (one vs one)
		PWData timeCourseRef = PWComparison(metabolomics_data, timeCourse_Ref_samples, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_Ref_filename, timeCourseRef);
	}

	if (run_timeCourse_Gnd) {
		// Find significant pair-wise MARS between each sample (one vs one)
		PWData timeCourseGnd = PWComparison(metabolomics_data, timeCourse_Gnd_samples, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_Gnd_filename, timeCourseGnd);
	}

	if (run_timeCourse_SdhCB) {
		// Find significant pair-wise MARS between each sample (one vs one)
		PWData timeCourseSdhCB = PWComparison(metabolomics_data, timeCourse_SdhCB_samples, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_SdhCB_filename, timeCourseSdhCB);
	}

	if (run_timeCourse_Pgi) {
		// Find significant pair-wise MARS between each sample (one vs one)
		PWData timeCoursePgi = PWComparison(metabolomics_data, timeCourse_Pgi_samples, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_Pgi_filename, timeCoursePgi);
	}

	if (run_timeCourse_PtsHIcrr) {
		// Find significant pair-wise MARS between each sample (one vs one)
		PWData timeCoursePtsHIcrr = PWComparison(metabolomics_data, timeCourse_PtsHIcrr_samples, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_PtsHIcrr_filename, timeCoursePtsHIcrr);
	}

	if (run_timeCourse_TpiA) {
		// Find significant pair-wise MARS between each sample (one vs one)
		PWData timeCourseTpiA = PWComparison(metabolomics_data, timeCourse_TpiA_samples, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_TpiA_filename, timeCourseTpiA);
	}
}

void main_classification(bool make_model = true)
{

	// define the population trainer parameters
	PopulationTrainerExt<float> population_trainer;
	population_trainer.setNGenerations(1);
	population_trainer.setNTop(3);
	population_trainer.setNRandom(3);
	population_trainer.setNReplicatesPerModel(3);

	// define the multithreading parameters
	const int n_hard_threads = std::thread::hardware_concurrency();
	//const int n_threads = n_hard_threads / 2; // the number of threads
	//char threads_cout[512];
	//sprintf(threads_cout, "Threads for population training: %d, Threads for model training/validation: %d\n",
	//	n_hard_threads, 2);
	//std::cout << threads_cout;
	const int n_threads = 1;

	// define the data simulator
	MetDataSimClassification<float> metabolomics_data;
	std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
	//std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
	//std::string data_dir = "/home/user/Data/";
	std::string model_name = "0_Metabolomics";

	std::string biochem_rxns_filename, metabo_data_filename, meta_data_filename;
	// EColi filenames
	biochem_rxns_filename = data_dir + "iAB_EColi_283.csv";
	metabo_data_filename = data_dir + "MetabolomicsData_EColi.csv";
	meta_data_filename = data_dir + "MetaData_EColi.csv";
	metabolomics_data.readBiochemicalReactions(biochem_rxns_filename);
	metabolomics_data.readMetabolomicsData(metabo_data_filename);
	metabolomics_data.readMetaData(meta_data_filename);
	metabolomics_data.findComponentGroupNames();
	metabolomics_data.findMARs();
	metabolomics_data.findMARs(true, false);
	metabolomics_data.findMARs(false, true);
	metabolomics_data.removeRedundantMARs();
	metabolomics_data.findLabels();

	// define the model input/output nodes
	const int n_input_nodes = metabolomics_data.reaction_ids_.size();
	const int n_output_nodes = metabolomics_data.labels_.size();
	std::vector<std::string> input_nodes;
	std::vector<std::string> output_nodes, output_nodes_softmax;
	for (int i = 0; i < n_input_nodes; ++i) {
		char name_char[512];
		sprintf(name_char, "Input_%012d", i);
		std::string name(name_char);
		input_nodes.push_back(name);
	}
	for (int i = 0; i < n_output_nodes; ++i) {
		char name_char[512];
		sprintf(name_char, "Output_%012d", i);
		std::string name(name_char);
		output_nodes.push_back(name);
	} 
	for (int i = 0; i < n_output_nodes; ++i) {
		char name_char[512];
		sprintf(name_char, "SoftMax-Out_%012d", i);
		std::string name(name_char);
		output_nodes_softmax.push_back(name);
	}

	// define the model trainers and resources for the trainers
	std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
	for (size_t i = 0; i < n_threads; ++i) {
		ModelResources model_resources = { ModelDevice(0, 1) };
		ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
		model_interpreters.push_back(model_interpreter);
	}
	ModelTrainerExt<float> model_trainer;
	model_trainer.setBatchSize(64);
	model_trainer.setMemorySize(1);
	model_trainer.setNEpochsTraining(1000);
	model_trainer.setNEpochsValidation(0);
	model_trainer.setVerbosityLevel(1);
	model_trainer.setLogging(true, false, false);
	//model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()), std::shared_ptr<LossFunctionOp<float>>(new NegativeLogLikelihoodOp<float>(2)) 
	//});
	//model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()), std::shared_ptr<LossFunctionGradOp<float>>(new NegativeLogLikelihoodGradOp<float>(2)) 
	//});
	model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new CrossEntropyWithLogitsOp<float>()) });
	model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new CrossEntropyWithLogitsGradOp<float>()) });
	model_trainer.setOutputNodes({ output_nodes });
	//model_trainer.setOutputNodes({ output_nodes, output_nodes_softmax
	//});

	// define the model logger
	ModelLogger<float> model_logger(true, true, false, false, false, false, false, false);

	// initialize the model replicator
	ModelReplicatorExt<float> model_replicator;
	model_replicator.setNodeActivations({ std::make_pair(std::shared_ptr<ActivationOp<float>>(new ReLUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ReLUGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new LinearOp<float>()), std::shared_ptr<ActivationOp<float>>(new LinearGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new ELUOp<float>()), std::shared_ptr<ActivationOp<float>>(new ELUGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new SigmoidOp<float>()), std::shared_ptr<ActivationOp<float>>(new SigmoidGradOp<float>())),
		std::make_pair(std::shared_ptr<ActivationOp<float>>(new TanHOp<float>()), std::shared_ptr<ActivationOp<float>>(new TanHGradOp<float>())) });
	model_replicator.setNodeIntegrations({ std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new ProdOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new ProdErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new ProdWeightGradOp<float>())),
		std::make_tuple(std::shared_ptr<IntegrationOp<float>>(new SumOp<float>()), std::shared_ptr<IntegrationErrorOp<float>>(new SumErrorOp<float>()), std::shared_ptr<IntegrationWeightGradOp<float>>(new SumWeightGradOp<float>())) });

	// define the initial population
	std::cout << "Initializing the population..." << std::endl;
	std::vector<Model<float>> population;
	if (make_model) {
		Model<float> model;
		//model_trainer.makeModelFCClass(model, n_input_nodes, n_output_nodes);
		//model_trainer.makeMultiHeadDotProdAttention(model, input_nodes.size(), output_nodes.size(), { 2, 2 }, { 2, 2 }, { 2, 2 }, false, false, false);
		model_trainer.makeMultiHeadDotProdAttention(model, input_nodes.size(), output_nodes.size(), { 12, 6 }, { 48, 24 }, { 96, 48 }, false, false, false); //GPU
		population = { model };
	}
	else {
		ModelFile<float> model_file;
		Model<float> model;
		model_file.loadModelCsv(data_dir + model_name + "_Nodes.csv", data_dir + model_name + "_Links.csv", data_dir + model_name + "_Weights.csv", model);
		population = { model };
	}

	// Evolve the population
	std::vector<std::vector<std::tuple<int, std::string, float>>> models_validation_errors_per_generation = population_trainer.evolveModels(
		population, model_trainer, model_interpreters, model_replicator, metabolomics_data, model_logger, input_nodes);

	PopulationTrainerFile<float> population_trainer_file;
	population_trainer_file.storeModels(population, "Metabolomics");
	population_trainer_file.storeModelValidations("MetabolomicsValidationErrors.csv", models_validation_errors_per_generation.back());
}

// Main
int main(int argc, char** argv)
{
	main_statistics_timecourse(
		true, true, true, true, true,
		true);
	main_statistics_timecourseSummary(
		true, true, true, true, true,
		true);
	main_classification(true);
	return 0;
}