/**TODO:  Add copyright*/

#include "Metabolomics_example.h"

using namespace SmartPeak;

/*
@brief Example using intracellular E. coli metabolomics data
  taken from re-grown glycerol stock solutions on Glucose M9 at mid-exponential phase
  from adaptive laboratory evolution (ALE) experiments following gene knockout (KO)
*/

/// Script to run the time-course Summary
void main_statistics_timecourseSummary(const std::string& data_dir,
  bool run_timeCourse_Ref = false, bool run_timeCourse_Gnd = false, bool run_timeCourse_SdhCB = false, bool run_timeCourse_Pgi = false, bool run_timeCourse_PtsHIcrr = false,
  bool run_timeCourse_TpiA = false)
{
  // define the data simulator
  BiochemicalReactionModel<float> metabolomics_data;

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

/// Script to run the time-course MARs analysis
void main_statistics_timecourse(const std::string& data_dir,
  bool run_timeCourse_Ref = false, bool run_timeCourse_Gnd = false, bool run_timeCourse_SdhCB = false, bool run_timeCourse_Pgi = false, bool run_timeCourse_PtsHIcrr = false,
  bool run_timeCourse_TpiA = false)
{
  // define the data simulator
  BiochemicalReactionModel<float> metabolomics_data;

  std::string biochem_rxns_filename, metabo_data_filename, meta_data_filename,
    timeCourse_Ref_filename, timeCourse_Gnd_filename, timeCourse_SdhCB_filename, timeCourse_Pgi_filename, timeCourse_PtsHIcrr_filename,
    timeCourse_TpiA_filename;
  std::vector<std::string> pre_samples,
    timeCourse_Ref_samples, timeCourse_Gnd_samples, timeCourse_SdhCB_samples, timeCourse_Pgi_samples, timeCourse_PtsHIcrr_samples,
    timeCourse_TpiA_samples;
  // filenames
  biochem_rxns_filename = data_dir + "iJO1366.csv";
  metabo_data_filename = data_dir + "ALEsKOs01_Metabolomics.csv";
  meta_data_filename = data_dir + "ALEsKOs01_MetaData.csv";
  timeCourse_Ref_filename = data_dir + "EColi_timeCourse_Ref.csv";
  timeCourse_Gnd_filename = data_dir + "EColi_timeCourse_Gnd.csv";
  timeCourse_SdhCB_filename = data_dir + "EColi_timeCourse_SdhCB.csv";
  timeCourse_Pgi_filename = data_dir + "EColi_timeCourse_Pgi.csv";
  timeCourse_PtsHIcrr_filename = data_dir + "EColi_timeCourse_PtsHIcrr.csv";
  timeCourse_TpiA_filename = data_dir + "EColi_timeCourse_TpiA.csv";
  timeCourse_Ref_samples = { "Evo04", "Evo04Evo01EP", "Evo04Evo02EP" };
  timeCourse_Gnd_samples = { "Evo04", "Evo04gnd", "Evo04gndEvo01EP", "Evo04gndEvo02EP", "Evo04gndEvo03EP" };
  timeCourse_SdhCB_samples = { "Evo04", "Evo04sdhCB", "Evo04sdhCBEvo01EP", "Evo04sdhCBEvo02EP", "Evo04sdhCBEvo03EP", "Evo04sdhCBEvo03EP-2", "Evo04sdhCBEvo03EP-3", "Evo04sdhCBEvo03EP-4", "Evo04sdhCBEvo03EP-5", "Evo04sdhCBEvo03EP-6" };
  timeCourse_Pgi_samples = { "Evo04", "Evo04pgi", "Evo04pgiEvo01EP", "Evo04pgiEvo01J01", "Evo04pgiEvo01J02", "Evo04pgiEvo02EP", "Evo04pgiEvo02J01", "Evo04pgiEvo02J02", "Evo04pgiEvo02J03", "Evo04pgiEvo03EP", "Evo04pgiEvo03J01", "Evo04pgiEvo03J02", "Evo04pgiEvo03J03", "Evo04pgiEvo04EP", "Evo04pgiEvo04J01", "Evo04pgiEvo04J02", "Evo04pgiEvo04J03", "Evo04pgiEvo05EP", "Evo04pgiEvo05J01", "Evo04pgiEvo05J02", "Evo04pgiEvo05J03", "Evo04pgiEvo06EP", "Evo04pgiEvo06J01", "Evo04pgiEvo06J02", "Evo04pgiEvo06J03", "Evo04pgiEvo07EP", "Evo04pgiEvo07J01", "Evo04pgiEvo07J02", "Evo04pgiEvo07J03", "Evo04pgiEvo08EP", "Evo04pgiEvo08J01", "Evo04pgiEvo08J02", "Evo04pgiEvo08J03" };
  timeCourse_PtsHIcrr_samples = { "Evo04", "Evo04ptsHIcrr", "Evo04ptsHIcrrEvo01EP", "Evo04ptsHIcrrEvo01J01", "Evo04ptsHIcrrEvo01J03", "Evo04ptsHIcrrEvo02EP", "Evo04ptsHIcrrEvo02J01", "Evo04ptsHIcrrEvo02J03", "Evo04ptsHIcrrEvo03EP", "Evo04ptsHIcrrEvo03J01", "Evo04ptsHIcrrEvo03J03", "Evo04ptsHIcrrEvo03J04", "Evo04ptsHIcrrEvo04EP", "Evo04ptsHIcrrEvo04J01", "Evo04ptsHIcrrEvo04J03", "Evo04ptsHIcrrEvo04J04" };
  timeCourse_TpiA_samples = { "Evo04", "Evo04tpiA", "Evo04tpiAEvo01EP", "Evo04tpiAEvo01J01", "Evo04tpiAEvo01J03", "Evo04tpiAEvo02EP", "Evo04tpiAEvo02J01", "Evo04tpiAEvo02J03", "Evo04tpiAEvo03EP", "Evo04tpiAEvo03J01", "Evo04tpiAEvo03J03", "Evo04tpiAEvo04EP", "Evo04tpiAEvo04J01", "Evo04tpiAEvo04J03" };

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

// Main
int main(int argc, char** argv)
{
  // Set the data directories
  //const std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  const std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_KALE/";
  //const std::string data_dir = "/home/user/Data/";

  main_statistics_timecourse(data_dir, 
  	true, true, true, true, true,
  	true);
  main_statistics_timecourseSummary(data_dir, 
  	true, true, true, true, true,
  	true);

  return 0;
}