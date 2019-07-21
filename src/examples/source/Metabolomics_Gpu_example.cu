/**TODO:  Add copyright*/

#include "Metabolomics_example.h"

using namespace SmartPeak;

/*
@brief Example using blood metabolomics data from three different blood fractions including
	Platelet (PLT), Red blood cells (RBC), and Plasma (P) fractions from two experiments including
	an in vivo pre/post drug response and ex vivo drug response
*/

// Scripts to run
void main_statistics_timecourseSummary(std::string blood_fraction = "PLT",
	bool run_timeCourse_S01D01 = false, bool run_timeCourse_S01D02 = false, bool run_timeCourse_S01D03 = false, bool run_timeCourse_S01D04 = false, bool run_timeCourse_S01D05 = false,
	bool run_timeCourse_S02D01 = false, bool run_timeCourse_S02D02 = false, bool run_timeCourse_S02D03 = false, bool run_timeCourse_S02D04 = false, bool run_timeCourse_S02D05 = false,
	bool run_timeCourse_S01D01vsS01D02 = false, bool run_timeCourse_S01D01vsS01D03 = false, bool run_timeCourse_S01D01vsS01D04 = false, bool run_timeCourse_S01D01vsS01D05 = false,
	bool run_timeCourse_S02D01vsS02D02 = false, bool run_timeCourse_S02D01vsS02D03 = false, bool run_timeCourse_S02D01vsS02D04 = false, bool run_timeCourse_S02D01vsS02D05 = false)
{
	// define the data simulator
	BiochemicalReactionModel<float> metabolomics_data;

	// data dirs
	//std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	//std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	std::string data_dir = "/home/user/Data/";

	std::string
		timeCourse_S01D01_filename, timeCourse_S01D02_filename, timeCourse_S01D03_filename, timeCourse_S01D04_filename, timeCourse_S01D05_filename,
		timeCourse_S02D01_filename, timeCourse_S02D02_filename, timeCourse_S02D03_filename, timeCourse_S02D04_filename, timeCourse_S02D05_filename,
		timeCourse_S01D01vsS01D02_filename, timeCourse_S01D01vsS01D03_filename, timeCourse_S01D01vsS01D04_filename, timeCourse_S01D01vsS01D05_filename,
		timeCourse_S02D01vsS02D02_filename, timeCourse_S02D01vsS02D03_filename, timeCourse_S02D01vsS02D04_filename, timeCourse_S02D01vsS02D05_filename,
		timeCourseSampleSummary_S01D01_filename, timeCourseSampleSummary_S01D02_filename, timeCourseSampleSummary_S01D03_filename, timeCourseSampleSummary_S01D04_filename, timeCourseSampleSummary_S01D05_filename,
		timeCourseSampleSummary_S02D01_filename, timeCourseSampleSummary_S02D02_filename, timeCourseSampleSummary_S02D03_filename, timeCourseSampleSummary_S02D04_filename, timeCourseSampleSummary_S02D05_filename,
		timeCourseSampleSummary_S01D01vsS01D02_filename, timeCourseSampleSummary_S01D01vsS01D03_filename, timeCourseSampleSummary_S01D01vsS01D04_filename, timeCourseSampleSummary_S01D01vsS01D05_filename,
		timeCourseSampleSummary_S02D01vsS02D02_filename, timeCourseSampleSummary_S02D01vsS02D03_filename, timeCourseSampleSummary_S02D01vsS02D04_filename, timeCourseSampleSummary_S02D01vsS02D05_filename,
		timeCourseFeatureSummary_S01D01_filename, timeCourseFeatureSummary_S01D02_filename, timeCourseFeatureSummary_S01D03_filename, timeCourseFeatureSummary_S01D04_filename, timeCourseFeatureSummary_S01D05_filename,
		timeCourseFeatureSummary_S02D01_filename, timeCourseFeatureSummary_S02D02_filename, timeCourseFeatureSummary_S02D03_filename, timeCourseFeatureSummary_S02D04_filename, timeCourseFeatureSummary_S02D05_filename,
		timeCourseFeatureSummary_S01D01vsS01D02_filename, timeCourseFeatureSummary_S01D01vsS01D03_filename, timeCourseFeatureSummary_S01D01vsS01D04_filename, timeCourseFeatureSummary_S01D01vsS01D05_filename,
		timeCourseFeatureSummary_S02D01vsS02D02_filename, timeCourseFeatureSummary_S02D01vsS02D03_filename, timeCourseFeatureSummary_S02D01vsS02D04_filename, timeCourseFeatureSummary_S02D01vsS02D05_filename;
	if (blood_fraction == "RBC") {
		// RBC filenames
		timeCourse_S01D01_filename = data_dir + "RBC_timeCourse_S01D01.csv";
		timeCourse_S01D02_filename = data_dir + "RBC_timeCourse_S01D02.csv";
		timeCourse_S01D03_filename = data_dir + "RBC_timeCourse_S01D03.csv";
		timeCourse_S01D04_filename = data_dir + "RBC_timeCourse_S01D04.csv";
		timeCourse_S01D05_filename = data_dir + "RBC_timeCourse_S01D05.csv";
		timeCourse_S02D01_filename = data_dir + "RBC_timeCourse_S02D01.csv";
		timeCourse_S02D02_filename = data_dir + "RBC_timeCourse_S02D02.csv";
		timeCourse_S02D03_filename = data_dir + "RBC_timeCourse_S02D03.csv";
		timeCourse_S02D04_filename = data_dir + "RBC_timeCourse_S02D04.csv";
		timeCourse_S02D05_filename = data_dir + "RBC_timeCourse_S02D05.csv";
		timeCourse_S01D01vsS01D02_filename = data_dir + "RBC_timeCourse_S01D01vsS01D02.csv";
		timeCourse_S01D01vsS01D03_filename = data_dir + "RBC_timeCourse_S01D01vsS01D03.csv";
		timeCourse_S01D01vsS01D04_filename = data_dir + "RBC_timeCourse_S01D01vsS01D04.csv";
		timeCourse_S01D01vsS01D05_filename = data_dir + "RBC_timeCourse_S01D01vsS01D05.csv";
		timeCourse_S02D01vsS02D02_filename = data_dir + "RBC_timeCourse_S02D01vsS02D02.csv";
		timeCourse_S02D01vsS02D03_filename = data_dir + "RBC_timeCourse_S02D01vsS02D03.csv";
		timeCourse_S02D01vsS02D04_filename = data_dir + "RBC_timeCourse_S02D01vsS02D04.csv";
		timeCourse_S02D01vsS02D05_filename = data_dir + "RBC_timeCourse_S02D01vsS02D05.csv";
		timeCourseSampleSummary_S01D01_filename = data_dir + "RBC_timeCourseSampleSummary_S01D01.csv";
		timeCourseSampleSummary_S01D02_filename = data_dir + "RBC_timeCourseSampleSummary_S01D02.csv";
		timeCourseSampleSummary_S01D03_filename = data_dir + "RBC_timeCourseSampleSummary_S01D03.csv";
		timeCourseSampleSummary_S01D04_filename = data_dir + "RBC_timeCourseSampleSummary_S01D04.csv";
		timeCourseSampleSummary_S01D05_filename = data_dir + "RBC_timeCourseSampleSummary_S01D05.csv";
		timeCourseSampleSummary_S02D01_filename = data_dir + "RBC_timeCourseSampleSummary_S02D01.csv";
		timeCourseSampleSummary_S02D02_filename = data_dir + "RBC_timeCourseSampleSummary_S02D02.csv";
		timeCourseSampleSummary_S02D03_filename = data_dir + "RBC_timeCourseSampleSummary_S02D03.csv";
		timeCourseSampleSummary_S02D04_filename = data_dir + "RBC_timeCourseSampleSummary_S02D04.csv";
		timeCourseSampleSummary_S02D05_filename = data_dir + "RBC_timeCourseSampleSummary_S02D05.csv";
		timeCourseSampleSummary_S01D01vsS01D02_filename = data_dir + "RBC_timeCourseSampleSummary_S01D01vsS01D02.csv";
		timeCourseSampleSummary_S01D01vsS01D03_filename = data_dir + "RBC_timeCourseSampleSummary_S01D01vsS01D03.csv";
		timeCourseSampleSummary_S01D01vsS01D04_filename = data_dir + "RBC_timeCourseSampleSummary_S01D01vsS01D04.csv";
		timeCourseSampleSummary_S01D01vsS01D05_filename = data_dir + "RBC_timeCourseSampleSummary_S01D01vsS01D05.csv";
		timeCourseSampleSummary_S02D01vsS02D02_filename = data_dir + "RBC_timeCourseSampleSummary_S02D01vsS02D02.csv";
		timeCourseSampleSummary_S02D01vsS02D03_filename = data_dir + "RBC_timeCourseSampleSummary_S02D01vsS02D03.csv";
		timeCourseSampleSummary_S02D01vsS02D04_filename = data_dir + "RBC_timeCourseSampleSummary_S02D01vsS02D04.csv";
		timeCourseSampleSummary_S02D01vsS02D05_filename = data_dir + "RBC_timeCourseSampleSummary_S02D01vsS02D05.csv";
		timeCourseFeatureSummary_S01D01_filename = data_dir + "RBC_timeCourseFeatureSummary_S01D01.csv";
		timeCourseFeatureSummary_S01D02_filename = data_dir + "RBC_timeCourseFeatureSummary_S01D02.csv";
		timeCourseFeatureSummary_S01D03_filename = data_dir + "RBC_timeCourseFeatureSummary_S01D03.csv";
		timeCourseFeatureSummary_S01D04_filename = data_dir + "RBC_timeCourseFeatureSummary_S01D04.csv";
		timeCourseFeatureSummary_S01D05_filename = data_dir + "RBC_timeCourseFeatureSummary_S01D05.csv";
		timeCourseFeatureSummary_S02D01_filename = data_dir + "RBC_timeCourseFeatureSummary_S02D01.csv";
		timeCourseFeatureSummary_S02D02_filename = data_dir + "RBC_timeCourseFeatureSummary_S02D02.csv";
		timeCourseFeatureSummary_S02D03_filename = data_dir + "RBC_timeCourseFeatureSummary_S02D03.csv";
		timeCourseFeatureSummary_S02D04_filename = data_dir + "RBC_timeCourseFeatureSummary_S02D04.csv";
		timeCourseFeatureSummary_S02D05_filename = data_dir + "RBC_timeCourseFeatureSummary_S02D05.csv";
		timeCourseFeatureSummary_S01D01vsS01D02_filename = data_dir + "RBC_timeCourseFeatureSummary_S01D01vsS01D02.csv";
		timeCourseFeatureSummary_S01D01vsS01D03_filename = data_dir + "RBC_timeCourseFeatureSummary_S01D01vsS01D03.csv";
		timeCourseFeatureSummary_S01D01vsS01D04_filename = data_dir + "RBC_timeCourseFeatureSummary_S01D01vsS01D04.csv";
		timeCourseFeatureSummary_S01D01vsS01D05_filename = data_dir + "RBC_timeCourseFeatureSummary_S01D01vsS01D05.csv";
		timeCourseFeatureSummary_S02D01vsS02D02_filename = data_dir + "RBC_timeCourseFeatureSummary_S02D01vsS02D02.csv";
		timeCourseFeatureSummary_S02D01vsS02D03_filename = data_dir + "RBC_timeCourseFeatureSummary_S02D01vsS02D03.csv";
		timeCourseFeatureSummary_S02D01vsS02D04_filename = data_dir + "RBC_timeCourseFeatureSummary_S02D01vsS02D04.csv";
		timeCourseFeatureSummary_S02D01vsS02D05_filename = data_dir + "RBC_timeCourseFeatureSummary_S02D01vsS02D05.csv";
	}
	else if (blood_fraction == "PLT") {
		// PLT filenames
		timeCourse_S01D01_filename = data_dir + "PLT_timeCourse_S01D01.csv";
		timeCourse_S01D02_filename = data_dir + "PLT_timeCourse_S01D02.csv";
		timeCourse_S01D03_filename = data_dir + "PLT_timeCourse_S01D03.csv";
		timeCourse_S01D04_filename = data_dir + "PLT_timeCourse_S01D04.csv";
		timeCourse_S01D05_filename = data_dir + "PLT_timeCourse_S01D05.csv";
		timeCourse_S02D01_filename = data_dir + "PLT_timeCourse_S02D01.csv";
		timeCourse_S02D02_filename = data_dir + "PLT_timeCourse_S02D02.csv";
		timeCourse_S02D03_filename = data_dir + "PLT_timeCourse_S02D03.csv";
		timeCourse_S02D04_filename = data_dir + "PLT_timeCourse_S02D04.csv";
		timeCourse_S02D05_filename = data_dir + "PLT_timeCourse_S02D05.csv";
		timeCourse_S01D01vsS01D02_filename = data_dir + "PLT_timeCourse_S01D01vsS01D02.csv";
		timeCourse_S01D01vsS01D03_filename = data_dir + "PLT_timeCourse_S01D01vsS01D03.csv";
		timeCourse_S01D01vsS01D04_filename = data_dir + "PLT_timeCourse_S01D01vsS01D04.csv";
		timeCourse_S01D01vsS01D05_filename = data_dir + "PLT_timeCourse_S01D01vsS01D05.csv";
		timeCourse_S02D01vsS02D02_filename = data_dir + "PLT_timeCourse_S02D01vsS02D02.csv";
		timeCourse_S02D01vsS02D03_filename = data_dir + "PLT_timeCourse_S02D01vsS02D03.csv";
		timeCourse_S02D01vsS02D04_filename = data_dir + "PLT_timeCourse_S02D01vsS02D04.csv";
		timeCourse_S02D01vsS02D05_filename = data_dir + "PLT_timeCourse_S02D01vsS02D05.csv";
		timeCourseSampleSummary_S01D01_filename = data_dir + "PLT_timeCourseSampleSummary_S01D01.csv";
		timeCourseSampleSummary_S01D02_filename = data_dir + "PLT_timeCourseSampleSummary_S01D02.csv";
		timeCourseSampleSummary_S01D03_filename = data_dir + "PLT_timeCourseSampleSummary_S01D03.csv";
		timeCourseSampleSummary_S01D04_filename = data_dir + "PLT_timeCourseSampleSummary_S01D04.csv";
		timeCourseSampleSummary_S01D05_filename = data_dir + "PLT_timeCourseSampleSummary_S01D05.csv";
		timeCourseSampleSummary_S02D01_filename = data_dir + "PLT_timeCourseSampleSummary_S02D01.csv";
		timeCourseSampleSummary_S02D02_filename = data_dir + "PLT_timeCourseSampleSummary_S02D02.csv";
		timeCourseSampleSummary_S02D03_filename = data_dir + "PLT_timeCourseSampleSummary_S02D03.csv";
		timeCourseSampleSummary_S02D04_filename = data_dir + "PLT_timeCourseSampleSummary_S02D04.csv";
		timeCourseSampleSummary_S02D05_filename = data_dir + "PLT_timeCourseSampleSummary_S02D05.csv";
		timeCourseSampleSummary_S01D01vsS01D02_filename = data_dir + "PLT_timeCourseSampleSummary_S01D01vsS01D02.csv";
		timeCourseSampleSummary_S01D01vsS01D03_filename = data_dir + "PLT_timeCourseSampleSummary_S01D01vsS01D03.csv";
		timeCourseSampleSummary_S01D01vsS01D04_filename = data_dir + "PLT_timeCourseSampleSummary_S01D01vsS01D04.csv";
		timeCourseSampleSummary_S01D01vsS01D05_filename = data_dir + "PLT_timeCourseSampleSummary_S01D01vsS01D05.csv";
		timeCourseSampleSummary_S02D01vsS02D02_filename = data_dir + "PLT_timeCourseSampleSummary_S02D01vsS02D02.csv";
		timeCourseSampleSummary_S02D01vsS02D03_filename = data_dir + "PLT_timeCourseSampleSummary_S02D01vsS02D03.csv";
		timeCourseSampleSummary_S02D01vsS02D04_filename = data_dir + "PLT_timeCourseSampleSummary_S02D01vsS02D04.csv";
		timeCourseSampleSummary_S02D01vsS02D05_filename = data_dir + "PLT_timeCourseSampleSummary_S02D01vsS02D05.csv";
		timeCourseFeatureSummary_S01D01_filename = data_dir + "PLT_timeCourseFeatureSummary_S01D01.csv";
		timeCourseFeatureSummary_S01D02_filename = data_dir + "PLT_timeCourseFeatureSummary_S01D02.csv";
		timeCourseFeatureSummary_S01D03_filename = data_dir + "PLT_timeCourseFeatureSummary_S01D03.csv";
		timeCourseFeatureSummary_S01D04_filename = data_dir + "PLT_timeCourseFeatureSummary_S01D04.csv";
		timeCourseFeatureSummary_S01D05_filename = data_dir + "PLT_timeCourseFeatureSummary_S01D05.csv";
		timeCourseFeatureSummary_S02D01_filename = data_dir + "PLT_timeCourseFeatureSummary_S02D01.csv";
		timeCourseFeatureSummary_S02D02_filename = data_dir + "PLT_timeCourseFeatureSummary_S02D02.csv";
		timeCourseFeatureSummary_S02D03_filename = data_dir + "PLT_timeCourseFeatureSummary_S02D03.csv";
		timeCourseFeatureSummary_S02D04_filename = data_dir + "PLT_timeCourseFeatureSummary_S02D04.csv";
		timeCourseFeatureSummary_S02D05_filename = data_dir + "PLT_timeCourseFeatureSummary_S02D05.csv";
		timeCourseFeatureSummary_S01D01vsS01D02_filename = data_dir + "PLT_timeCourseFeatureSummary_S01D01vsS01D02.csv";
		timeCourseFeatureSummary_S01D01vsS01D03_filename = data_dir + "PLT_timeCourseFeatureSummary_S01D01vsS01D03.csv";
		timeCourseFeatureSummary_S01D01vsS01D04_filename = data_dir + "PLT_timeCourseFeatureSummary_S01D01vsS01D04.csv";
		timeCourseFeatureSummary_S01D01vsS01D05_filename = data_dir + "PLT_timeCourseFeatureSummary_S01D01vsS01D05.csv";
		timeCourseFeatureSummary_S02D01vsS02D02_filename = data_dir + "PLT_timeCourseFeatureSummary_S02D01vsS02D02.csv";
		timeCourseFeatureSummary_S02D01vsS02D03_filename = data_dir + "PLT_timeCourseFeatureSummary_S02D01vsS02D03.csv";
		timeCourseFeatureSummary_S02D01vsS02D04_filename = data_dir + "PLT_timeCourseFeatureSummary_S02D01vsS02D04.csv";
		timeCourseFeatureSummary_S02D01vsS02D05_filename = data_dir + "PLT_timeCourseFeatureSummary_S02D01vsS02D05.csv";
	}
	else if (blood_fraction == "P") {
		// P filenames
		timeCourse_S01D01_filename = data_dir + "P_timeCourse_S01D01.csv";
		timeCourse_S01D02_filename = data_dir + "P_timeCourse_S01D02.csv";
		timeCourse_S01D03_filename = data_dir + "P_timeCourse_S01D03.csv";
		timeCourse_S01D04_filename = data_dir + "P_timeCourse_S01D04.csv";
		timeCourse_S01D05_filename = data_dir + "P_timeCourse_S01D05.csv";
		timeCourse_S02D01_filename = data_dir + "P_timeCourse_S02D01.csv";
		timeCourse_S02D02_filename = data_dir + "P_timeCourse_S02D02.csv";
		timeCourse_S02D03_filename = data_dir + "P_timeCourse_S02D03.csv";
		timeCourse_S02D04_filename = data_dir + "P_timeCourse_S02D04.csv";
		timeCourse_S02D05_filename = data_dir + "P_timeCourse_S02D05.csv";
		timeCourse_S01D01vsS01D02_filename = data_dir + "P_timeCourse_S01D01vsS01D02.csv";
		timeCourse_S01D01vsS01D03_filename = data_dir + "P_timeCourse_S01D01vsS01D03.csv";
		timeCourse_S01D01vsS01D04_filename = data_dir + "P_timeCourse_S01D01vsS01D04.csv";
		timeCourse_S01D01vsS01D05_filename = data_dir + "P_timeCourse_S01D01vsS01D05.csv";
		timeCourse_S02D01vsS02D02_filename = data_dir + "P_timeCourse_S02D01vsS02D02.csv";
		timeCourse_S02D01vsS02D03_filename = data_dir + "P_timeCourse_S02D01vsS02D03.csv";
		timeCourse_S02D01vsS02D04_filename = data_dir + "P_timeCourse_S02D01vsS02D04.csv";
		timeCourse_S02D01vsS02D05_filename = data_dir + "P_timeCourse_S02D01vsS02D05.csv";
		timeCourseSampleSummary_S01D01_filename = data_dir + "P_timeCourseSampleSummary_S01D01.csv";
		timeCourseSampleSummary_S01D02_filename = data_dir + "P_timeCourseSampleSummary_S01D02.csv";
		timeCourseSampleSummary_S01D03_filename = data_dir + "P_timeCourseSampleSummary_S01D03.csv";
		timeCourseSampleSummary_S01D04_filename = data_dir + "P_timeCourseSampleSummary_S01D04.csv";
		timeCourseSampleSummary_S01D05_filename = data_dir + "P_timeCourseSampleSummary_S01D05.csv";
		timeCourseSampleSummary_S02D01_filename = data_dir + "P_timeCourseSampleSummary_S02D01.csv";
		timeCourseSampleSummary_S02D02_filename = data_dir + "P_timeCourseSampleSummary_S02D02.csv";
		timeCourseSampleSummary_S02D03_filename = data_dir + "P_timeCourseSampleSummary_S02D03.csv";
		timeCourseSampleSummary_S02D04_filename = data_dir + "P_timeCourseSampleSummary_S02D04.csv";
		timeCourseSampleSummary_S02D05_filename = data_dir + "P_timeCourseSampleSummary_S02D05.csv";
		timeCourseSampleSummary_S01D01vsS01D02_filename = data_dir + "P_timeCourseSampleSummary_S01D01vsS01D02.csv";
		timeCourseSampleSummary_S01D01vsS01D03_filename = data_dir + "P_timeCourseSampleSummary_S01D01vsS01D03.csv";
		timeCourseSampleSummary_S01D01vsS01D04_filename = data_dir + "P_timeCourseSampleSummary_S01D01vsS01D04.csv";
		timeCourseSampleSummary_S01D01vsS01D05_filename = data_dir + "P_timeCourseSampleSummary_S01D01vsS01D05.csv";
		timeCourseSampleSummary_S02D01vsS02D02_filename = data_dir + "P_timeCourseSampleSummary_S02D01vsS02D02.csv";
		timeCourseSampleSummary_S02D01vsS02D03_filename = data_dir + "P_timeCourseSampleSummary_S02D01vsS02D03.csv";
		timeCourseSampleSummary_S02D01vsS02D04_filename = data_dir + "P_timeCourseSampleSummary_S02D01vsS02D04.csv";
		timeCourseSampleSummary_S02D01vsS02D05_filename = data_dir + "P_timeCourseSampleSummary_S02D01vsS02D05.csv";
		timeCourseFeatureSummary_S01D01_filename = data_dir + "P_timeCourseFeatureSummary_S01D01.csv";
		timeCourseFeatureSummary_S01D02_filename = data_dir + "P_timeCourseFeatureSummary_S01D02.csv";
		timeCourseFeatureSummary_S01D03_filename = data_dir + "P_timeCourseFeatureSummary_S01D03.csv";
		timeCourseFeatureSummary_S01D04_filename = data_dir + "P_timeCourseFeatureSummary_S01D04.csv";
		timeCourseFeatureSummary_S01D05_filename = data_dir + "P_timeCourseFeatureSummary_S01D05.csv";
		timeCourseFeatureSummary_S02D01_filename = data_dir + "P_timeCourseFeatureSummary_S02D01.csv";
		timeCourseFeatureSummary_S02D02_filename = data_dir + "P_timeCourseFeatureSummary_S02D02.csv";
		timeCourseFeatureSummary_S02D03_filename = data_dir + "P_timeCourseFeatureSummary_S02D03.csv";
		timeCourseFeatureSummary_S02D04_filename = data_dir + "P_timeCourseFeatureSummary_S02D04.csv";
		timeCourseFeatureSummary_S02D05_filename = data_dir + "P_timeCourseFeatureSummary_S02D05.csv";
		timeCourseFeatureSummary_S01D01vsS01D02_filename = data_dir + "P_timeCourseFeatureSummary_S01D01vsS01D02.csv";
		timeCourseFeatureSummary_S01D01vsS01D03_filename = data_dir + "P_timeCourseFeatureSummary_S01D01vsS01D03.csv";
		timeCourseFeatureSummary_S01D01vsS01D04_filename = data_dir + "P_timeCourseFeatureSummary_S01D01vsS01D04.csv";
		timeCourseFeatureSummary_S01D01vsS01D05_filename = data_dir + "P_timeCourseFeatureSummary_S01D01vsS01D05.csv";
		timeCourseFeatureSummary_S02D01vsS02D02_filename = data_dir + "P_timeCourseFeatureSummary_S02D01vsS02D02.csv";
		timeCourseFeatureSummary_S02D01vsS02D03_filename = data_dir + "P_timeCourseFeatureSummary_S02D01vsS02D03.csv";
		timeCourseFeatureSummary_S02D01vsS02D04_filename = data_dir + "P_timeCourseFeatureSummary_S02D01vsS02D04.csv";
		timeCourseFeatureSummary_S02D01vsS02D05_filename = data_dir + "P_timeCourseFeatureSummary_S02D01vsS02D05.csv";
	}

	if (run_timeCourse_S01D01) {
		// Read in the data
		PWData timeCourseS01D01;
		ReadPWData(timeCourse_S01D01_filename, timeCourseS01D01);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseS01D01, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_S01D01_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_S01D01_filename, pw_feature_summaries);
	}

	if (run_timeCourse_S01D02) {
		// Read in the data
		PWData timeCourseS01D02;
		ReadPWData(timeCourse_S01D02_filename, timeCourseS01D02);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseS01D02, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_S01D02_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_S01D02_filename, pw_feature_summaries);
	}

	if (run_timeCourse_S01D03) {
		// Read in the data
		PWData timeCourseS01D03;
		ReadPWData(timeCourse_S01D03_filename, timeCourseS01D03);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseS01D03, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_S01D03_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_S01D03_filename, pw_feature_summaries);
	}

	if (run_timeCourse_S01D04) {
		// Read in the data
		PWData timeCourseS01D04;
		ReadPWData(timeCourse_S01D04_filename, timeCourseS01D04);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseS01D04, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_S01D04_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_S01D04_filename, pw_feature_summaries);
	}

	if (run_timeCourse_S01D05) {
		// Read in the data
		PWData timeCourseS01D05;
		ReadPWData(timeCourse_S01D05_filename, timeCourseS01D05);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseS01D05, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_S01D05_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_S01D05_filename, pw_feature_summaries);
	}

	if (run_timeCourse_S02D01) {
		// Read in the data
		PWData timeCourseS02D01;
		ReadPWData(timeCourse_S02D01_filename, timeCourseS02D01);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseS02D01, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_S02D01_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_S02D01_filename, pw_feature_summaries);
	}

	if (run_timeCourse_S02D02) {
		// Read in the data
		PWData timeCourseS02D02;
		ReadPWData(timeCourse_S02D02_filename, timeCourseS02D02);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseS02D02, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_S02D02_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_S02D02_filename, pw_feature_summaries);
	}

	if (run_timeCourse_S02D03) {
		// Read in the data
		PWData timeCourseS02D03;
		ReadPWData(timeCourse_S02D03_filename, timeCourseS02D03);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseS02D03, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_S02D03_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_S02D03_filename, pw_feature_summaries);
	}

	if (run_timeCourse_S02D04) {
		// Read in the data
		PWData timeCourseS02D04;
		ReadPWData(timeCourse_S02D04_filename, timeCourseS02D04);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseS02D04, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_S02D04_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_S02D04_filename, pw_feature_summaries);
	}

	if (run_timeCourse_S02D05) {
		// Read in the data
		PWData timeCourseS02D05;
		ReadPWData(timeCourse_S02D05_filename, timeCourseS02D05);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseS02D05, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_S02D05_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_S02D05_filename, pw_feature_summaries);
	}

	if (run_timeCourse_S01D01vsS01D02) {
		// Read in the data
		PWData timeCourseS01D01vsS01D02;
		ReadPWData(timeCourse_S01D01vsS01D02_filename, timeCourseS01D01vsS01D02);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseS01D01vsS01D02, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_S01D01vsS01D02_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_S01D01vsS01D02_filename, pw_feature_summaries);
	}

	if (run_timeCourse_S01D01vsS01D03) {
		// Read in the data
		PWData timeCourseS01D01vsS01D03;
		ReadPWData(timeCourse_S01D01vsS01D03_filename, timeCourseS01D01vsS01D03);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseS01D01vsS01D03, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_S01D01vsS01D03_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_S01D01vsS01D03_filename, pw_feature_summaries);
	}

	if (run_timeCourse_S01D01vsS01D04) {
		// Read in the data
		PWData timeCourseS01D01vsS01D04;
		ReadPWData(timeCourse_S01D01vsS01D04_filename, timeCourseS01D01vsS01D04);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseS01D01vsS01D04, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_S01D01vsS01D04_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_S01D01vsS01D04_filename, pw_feature_summaries);
	}

	if (run_timeCourse_S01D01vsS01D05) {
		// Read in the data
		PWData timeCourseS01D01vsS01D05;
		ReadPWData(timeCourse_S01D01vsS01D05_filename, timeCourseS01D01vsS01D05);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseS01D01vsS01D05, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_S01D01vsS01D05_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_S01D01vsS01D05_filename, pw_feature_summaries);
	}

	if (run_timeCourse_S02D01vsS02D02) {
		// Read in the data
		PWData timeCourseS02D01vsS02D02;
		ReadPWData(timeCourse_S02D01vsS02D02_filename, timeCourseS02D01vsS02D02);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseS02D01vsS02D02, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_S02D01vsS02D02_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_S02D01vsS02D02_filename, pw_feature_summaries);
	}

	if (run_timeCourse_S02D01vsS02D03) {
		// Read in the data
		PWData timeCourseS02D01vsS02D03;
		ReadPWData(timeCourse_S02D01vsS02D03_filename, timeCourseS02D01vsS02D03);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseS02D01vsS02D03, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_S02D01vsS02D03_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_S02D01vsS02D03_filename, pw_feature_summaries);
	}

	if (run_timeCourse_S02D01vsS02D04) {
		// Read in the data
		PWData timeCourseS02D01vsS02D04;
		ReadPWData(timeCourse_S02D01vsS02D04_filename, timeCourseS02D01vsS02D04);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseS02D01vsS02D04, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_S02D01vsS02D04_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_S02D01vsS02D04_filename, pw_feature_summaries);
	}

	if (run_timeCourse_S02D01vsS02D05) {
		// Read in the data
		PWData timeCourseS02D01vsS02D05;
		ReadPWData(timeCourse_S02D01vsS02D05_filename, timeCourseS02D01vsS02D05);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(timeCourseS02D01vsS02D05, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(timeCourseSampleSummary_S02D01vsS02D05_filename, pw_sample_summaries);
		WritePWFeatureSummaries(timeCourseFeatureSummary_S02D01vsS02D05_filename, pw_feature_summaries);
	}
}
void main_statistics_timecourse(std::string blood_fraction = "PLT",
	bool run_timeCourse_S01D01 = false, bool run_timeCourse_S01D02 = false, bool run_timeCourse_S01D03 = false, bool run_timeCourse_S01D04 = false, bool run_timeCourse_S01D05 = false,
	bool run_timeCourse_S02D01 = false, bool run_timeCourse_S02D02 = false, bool run_timeCourse_S02D03 = false, bool run_timeCourse_S02D04 = false, bool run_timeCourse_S02D05 = false,
	bool run_timeCourse_S01D01vsS01D02 = false, bool run_timeCourse_S01D01vsS01D03 = false, bool run_timeCourse_S01D01vsS01D04 = false, bool run_timeCourse_S01D01vsS01D05 = false,
	bool run_timeCourse_S02D01vsS02D02 = false, bool run_timeCourse_S02D01vsS02D03 = false, bool run_timeCourse_S02D01vsS02D04 = false, bool run_timeCourse_S02D01vsS02D05 = false)
{
	// define the data simulator
	BiochemicalReactionModel<float> metabolomics_data;

	// data dirs
	//std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	//std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	std::string data_dir = "/home/user/Data/";

	std::string biochem_rxns_filename, metabo_data_filename, meta_data_filename,
		timeCourse_S01D01_filename, timeCourse_S01D02_filename, timeCourse_S01D03_filename, timeCourse_S01D04_filename, timeCourse_S01D05_filename,
		timeCourse_S02D01_filename, timeCourse_S02D02_filename, timeCourse_S02D03_filename, timeCourse_S02D04_filename, timeCourse_S02D05_filename,
		timeCourse_S01D01vsS01D02_filename, timeCourse_S01D01vsS01D03_filename, timeCourse_S01D01vsS01D04_filename, timeCourse_S01D01vsS01D05_filename,
		timeCourse_S02D01vsS02D02_filename, timeCourse_S02D01vsS02D03_filename, timeCourse_S02D01vsS02D04_filename, timeCourse_S02D01vsS02D05_filename;
	std::vector<std::string> pre_samples,
		timeCourse_S01D01_samples, timeCourse_S01D02_samples, timeCourse_S01D03_samples, timeCourse_S01D04_samples, timeCourse_S01D05_samples,
		timeCourse_S02D01_samples, timeCourse_S02D02_samples, timeCourse_S02D03_samples, timeCourse_S02D04_samples, timeCourse_S02D05_samples;
	if (blood_fraction == "RBC") {
		// RBC filenames
		biochem_rxns_filename = data_dir + "iAB_RBC_283.csv";
		metabo_data_filename = data_dir + "MetabolomicsData_RBC.csv";
		meta_data_filename = data_dir + "MetaData_prePost_RBC.csv";
		timeCourse_S01D01_filename = data_dir + "RBC_timeCourse_S01D01.csv";
		timeCourse_S01D02_filename = data_dir + "RBC_timeCourse_S01D02.csv";
		timeCourse_S01D03_filename = data_dir + "RBC_timeCourse_S01D03.csv";
		timeCourse_S01D04_filename = data_dir + "RBC_timeCourse_S01D04.csv";
		timeCourse_S01D05_filename = data_dir + "RBC_timeCourse_S01D05.csv";
		timeCourse_S02D01_filename = data_dir + "RBC_timeCourse_S02D01.csv";
		timeCourse_S02D02_filename = data_dir + "RBC_timeCourse_S02D02.csv";
		timeCourse_S02D03_filename = data_dir + "RBC_timeCourse_S02D03.csv";
		timeCourse_S02D04_filename = data_dir + "RBC_timeCourse_S02D04.csv";
		timeCourse_S02D05_filename = data_dir + "RBC_timeCourse_S02D05.csv";
		timeCourse_S01D01vsS01D02_filename = data_dir + "RBC_timeCourse_S01D01vsS01D02.csv";
		timeCourse_S01D01vsS01D03_filename = data_dir + "RBC_timeCourse_S01D01vsS01D03.csv";
		timeCourse_S01D01vsS01D04_filename = data_dir + "RBC_timeCourse_S01D01vsS01D04.csv";
		timeCourse_S01D01vsS01D05_filename = data_dir + "RBC_timeCourse_S01D01vsS01D05.csv";
		timeCourse_S02D01vsS02D02_filename = data_dir + "RBC_timeCourse_S02D01vsS02D02.csv";
		timeCourse_S02D01vsS02D03_filename = data_dir + "RBC_timeCourse_S02D01vsS02D03.csv";
		timeCourse_S02D01vsS02D04_filename = data_dir + "RBC_timeCourse_S02D01vsS02D04.csv";
		timeCourse_S02D01vsS02D05_filename = data_dir + "RBC_timeCourse_S02D01vsS02D05.csv";
		pre_samples = { "RBC_36","RBC_142","RBC_140","RBC_34","RBC_154","RBC_143","RBC_30","RBC_31","RBC_33","RBC_35","RBC_141" };
		timeCourse_S01D01_samples = { "S01_D01_RBC_25C_0hr","S01_D01_RBC_25C_2hr","S01_D01_RBC_25C_6.5hr","S01_D01_RBC_25C_22hr","S01_D01_RBC_37C_22hr" };
		timeCourse_S01D02_samples = { "S01_D02_RBC_25C_0hr","S01_D02_RBC_25C_2hr","S01_D02_RBC_25C_6.5hr","S01_D02_RBC_25C_22hr","S01_D02_RBC_37C_22hr" };
		timeCourse_S01D03_samples = { "S01_D03_RBC_25C_0hr","S01_D03_RBC_25C_2hr","S01_D03_RBC_25C_6.5hr","S01_D03_RBC_25C_22hr","S01_D03_RBC_37C_22hr" };
		timeCourse_S01D04_samples = { "S01_D04_RBC_25C_0hr","S01_D04_RBC_25C_2hr","S01_D04_RBC_25C_6.5hr","S01_D04_RBC_25C_22hr","S01_D04_RBC_37C_22hr" };
		timeCourse_S01D05_samples = { "S01_D05_RBC_25C_0hr","S01_D05_RBC_25C_2hr","S01_D05_RBC_25C_6.5hr","S01_D05_RBC_25C_22hr","S01_D05_RBC_37C_22hr" };
		timeCourse_S02D01_samples = { "S02_D01_RBC_25C_0hr","S02_D01_RBC_25C_2hr","S02_D01_RBC_25C_6.5hr","S02_D01_RBC_25C_22hr","S02_D01_RBC_37C_22hr" };
		timeCourse_S02D02_samples = { "S02_D02_RBC_25C_0hr","S02_D02_RBC_25C_2hr","S02_D02_RBC_25C_6.5hr","S02_D02_RBC_25C_22hr","S02_D02_RBC_37C_22hr" };
		timeCourse_S02D03_samples = { "S02_D03_RBC_25C_0hr","S02_D03_RBC_25C_2hr","S02_D03_RBC_25C_6.5hr","S02_D03_RBC_25C_22hr","S02_D03_RBC_37C_22hr" };
		timeCourse_S02D04_samples = { "S02_D04_RBC_25C_0hr","S02_D04_RBC_25C_2hr","S02_D04_RBC_25C_6.5hr","S02_D04_RBC_25C_22hr","S02_D04_RBC_37C_22hr" };
		timeCourse_S02D05_samples = { "S02_D05_RBC_25C_0hr","S02_D05_RBC_25C_2hr","S02_D05_RBC_25C_6.5hr","S02_D05_RBC_25C_22hr","S02_D05_RBC_37C_22hr" };
	}
	else if (blood_fraction == "PLT") {
		// PLT filenames
		biochem_rxns_filename = data_dir + "iAT_PLT_636.csv";
		metabo_data_filename = data_dir + "MetabolomicsData_PLT.csv";
		meta_data_filename = data_dir + "MetaData_prePost_PLT.csv";
		timeCourse_S01D01_filename = data_dir + "PLT_timeCourse_S01D01.csv";
		timeCourse_S01D02_filename = data_dir + "PLT_timeCourse_S01D02.csv";
		timeCourse_S01D03_filename = data_dir + "PLT_timeCourse_S01D03.csv";
		timeCourse_S01D04_filename = data_dir + "PLT_timeCourse_S01D04.csv";
		timeCourse_S01D05_filename = data_dir + "PLT_timeCourse_S01D05.csv";
		timeCourse_S02D01_filename = data_dir + "PLT_timeCourse_S02D01.csv";
		timeCourse_S02D02_filename = data_dir + "PLT_timeCourse_S02D02.csv";
		timeCourse_S02D03_filename = data_dir + "PLT_timeCourse_S02D03.csv";
		timeCourse_S02D04_filename = data_dir + "PLT_timeCourse_S02D04.csv";
		timeCourse_S02D05_filename = data_dir + "PLT_timeCourse_S02D05.csv";
		timeCourse_S01D01vsS01D02_filename = data_dir + "PLT_timeCourse_S01D01vsS01D02.csv";
		timeCourse_S01D01vsS01D03_filename = data_dir + "PLT_timeCourse_S01D01vsS01D03.csv";
		timeCourse_S01D01vsS01D04_filename = data_dir + "PLT_timeCourse_S01D01vsS01D04.csv";
		timeCourse_S01D01vsS01D05_filename = data_dir + "PLT_timeCourse_S01D01vsS01D05.csv";
		timeCourse_S02D01vsS02D02_filename = data_dir + "PLT_timeCourse_S02D01vsS02D02.csv";
		timeCourse_S02D01vsS02D03_filename = data_dir + "PLT_timeCourse_S02D01vsS02D03.csv";
		timeCourse_S02D01vsS02D04_filename = data_dir + "PLT_timeCourse_S02D01vsS02D04.csv";
		timeCourse_S02D01vsS02D05_filename = data_dir + "PLT_timeCourse_S02D01vsS02D05.csv";
		pre_samples = { "PLT_36","PLT_142","PLT_140","PLT_34","PLT_154","PLT_143","PLT_30","PLT_31","PLT_33","PLT_35","PLT_141" };
		timeCourse_S01D01_samples = { "S01_D01_PLT_25C_0hr","S01_D01_PLT_25C_2hr","S01_D01_PLT_25C_6.5hr","S01_D01_PLT_25C_22hr","S01_D01_PLT_37C_22hr" };
		timeCourse_S01D02_samples = { "S01_D02_PLT_25C_0hr","S01_D02_PLT_25C_2hr","S01_D02_PLT_25C_6.5hr","S01_D02_PLT_25C_22hr","S01_D02_PLT_37C_22hr" };
		timeCourse_S01D03_samples = { "S01_D03_PLT_25C_0hr","S01_D03_PLT_25C_2hr","S01_D03_PLT_25C_6.5hr","S01_D03_PLT_25C_22hr","S01_D03_PLT_37C_22hr" };
		timeCourse_S01D04_samples = { "S01_D04_PLT_25C_0hr","S01_D04_PLT_25C_2hr","S01_D04_PLT_25C_6.5hr","S01_D04_PLT_25C_22hr","S01_D04_PLT_37C_22hr" };
		timeCourse_S01D05_samples = { "S01_D05_PLT_25C_0hr","S01_D05_PLT_25C_2hr","S01_D05_PLT_25C_6.5hr","S01_D05_PLT_25C_22hr","S01_D05_PLT_37C_22hr" };
		timeCourse_S02D01_samples = { "S02_D01_PLT_25C_0hr","S02_D01_PLT_25C_2hr","S02_D01_PLT_25C_6.5hr","S02_D01_PLT_25C_22hr","S02_D01_PLT_37C_22hr" };
		timeCourse_S02D02_samples = { "S02_D02_PLT_25C_0hr","S02_D02_PLT_25C_2hr","S02_D02_PLT_25C_6.5hr","S02_D02_PLT_25C_22hr","S02_D02_PLT_37C_22hr" };
		timeCourse_S02D03_samples = { "S02_D03_PLT_25C_0hr","S02_D03_PLT_25C_2hr","S02_D03_PLT_25C_6.5hr","S02_D03_PLT_25C_22hr","S02_D03_PLT_37C_22hr" };
		timeCourse_S02D04_samples = { "S02_D04_PLT_25C_0hr","S02_D04_PLT_25C_2hr","S02_D04_PLT_25C_6.5hr","S02_D04_PLT_25C_22hr","S02_D04_PLT_37C_22hr" };
		timeCourse_S02D05_samples = { "S02_D05_PLT_25C_0hr","S02_D05_PLT_25C_2hr","S02_D05_PLT_25C_6.5hr","S02_D05_PLT_25C_22hr","S02_D05_PLT_37C_22hr" };
	}
	else if (blood_fraction == "P") {
		// P filenames
		biochem_rxns_filename = data_dir + "iAT_PLT_636.csv";
		metabo_data_filename = data_dir + "MetabolomicsData_P.csv";
		meta_data_filename = data_dir + "MetaData_prePost_P.csv";
		timeCourse_S01D01_filename = data_dir + "P_timeCourse_S01D01.csv";
		timeCourse_S01D02_filename = data_dir + "P_timeCourse_S01D02.csv";
		timeCourse_S01D03_filename = data_dir + "P_timeCourse_S01D03.csv";
		timeCourse_S01D04_filename = data_dir + "P_timeCourse_S01D04.csv";
		timeCourse_S01D05_filename = data_dir + "P_timeCourse_S01D05.csv";
		timeCourse_S02D01_filename = data_dir + "P_timeCourse_S02D01.csv";
		timeCourse_S02D02_filename = data_dir + "P_timeCourse_S02D02.csv";
		timeCourse_S02D03_filename = data_dir + "P_timeCourse_S02D03.csv";
		timeCourse_S02D04_filename = data_dir + "P_timeCourse_S02D04.csv";
		timeCourse_S02D05_filename = data_dir + "P_timeCourse_S02D05.csv";
		timeCourse_S01D01vsS01D02_filename = data_dir + "P_timeCourse_S01D01vsS01D02.csv";
		timeCourse_S01D01vsS01D03_filename = data_dir + "P_timeCourse_S01D01vsS01D03.csv";
		timeCourse_S01D01vsS01D04_filename = data_dir + "P_timeCourse_S01D01vsS01D04.csv";
		timeCourse_S01D01vsS01D05_filename = data_dir + "P_timeCourse_S01D01vsS01D05.csv";
		timeCourse_S02D01vsS02D02_filename = data_dir + "P_timeCourse_S02D01vsS02D02.csv";
		timeCourse_S02D01vsS02D03_filename = data_dir + "P_timeCourse_S02D01vsS02D03.csv";
		timeCourse_S02D01vsS02D04_filename = data_dir + "P_timeCourse_S02D01vsS02D04.csv";
		timeCourse_S02D01vsS02D05_filename = data_dir + "P_timeCourse_S02D01vsS02D05.csv";
		pre_samples = { "P_36","P_142","P_140","P_34","P_154","P_143","P_30","P_31","P_33","P_35","P_141" };
		timeCourse_S01D01_samples = { "S01_D01_P_25C_0hr","S01_D01_P_25C_2hr","S01_D01_P_25C_6.5hr","S01_D01_P_25C_22hr","S01_D01_P_37C_22hr" };
		timeCourse_S01D02_samples = { "S01_D02_P_25C_0hr","S01_D02_P_25C_2hr","S01_D02_P_25C_6.5hr","S01_D02_P_25C_22hr","S01_D02_P_37C_22hr" };
		timeCourse_S01D03_samples = { "S01_D03_P_25C_0hr","S01_D03_P_25C_2hr","S01_D03_P_25C_6.5hr","S01_D03_P_25C_22hr","S01_D03_P_37C_22hr" };
		timeCourse_S01D04_samples = { "S01_D04_P_25C_0hr","S01_D04_P_25C_2hr","S01_D04_P_25C_6.5hr","S01_D04_P_25C_22hr","S01_D04_P_37C_22hr" };
		timeCourse_S01D05_samples = { "S01_D05_P_25C_0hr","S01_D05_P_25C_2hr","S01_D05_P_25C_6.5hr","S01_D05_P_25C_22hr","S01_D05_P_37C_22hr" };
		timeCourse_S02D01_samples = { "S02_D01_P_25C_0hr","S02_D01_P_25C_2hr","S02_D01_P_25C_6.5hr","S02_D01_P_25C_22hr","S02_D01_P_37C_22hr" };
		timeCourse_S02D02_samples = { "S02_D02_P_25C_0hr","S02_D02_P_25C_2hr","S02_D02_P_25C_6.5hr","S02_D02_P_25C_22hr","S02_D02_P_37C_22hr" };
		timeCourse_S02D03_samples = { "S02_D03_P_25C_0hr","S02_D03_P_25C_2hr","S02_D03_P_25C_6.5hr","S02_D03_P_25C_22hr","S02_D03_P_37C_22hr" };
		timeCourse_S02D04_samples = { "S02_D04_P_25C_0hr","S02_D04_P_25C_2hr","S02_D04_P_25C_6.5hr","S02_D04_P_25C_22hr","S02_D04_P_37C_22hr" };
		timeCourse_S02D05_samples = { "S02_D05_P_25C_0hr","S02_D05_P_25C_2hr","S02_D05_P_25C_6.5hr","S02_D05_P_25C_22hr","S02_D05_P_37C_22hr" };
	}

	// read in the data
	metabolomics_data.readBiochemicalReactions(biochem_rxns_filename);
	metabolomics_data.readMetabolomicsData(metabo_data_filename);
	metabolomics_data.readMetaData(meta_data_filename);
	metabolomics_data.findComponentGroupNames();
	metabolomics_data.findMARs();
	metabolomics_data.findLabels();

	if (run_timeCourse_S01D01) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData timeCourseS01D01 = PWComparison(metabolomics_data, timeCourse_S01D01_samples, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_S01D01_filename, timeCourseS01D01);
	}

	if (run_timeCourse_S01D02) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData timeCourseS01D02 = PWComparison(metabolomics_data, timeCourse_S01D02_samples, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_S01D02_filename, timeCourseS01D02);
	}

	if (run_timeCourse_S01D03) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData timeCourseS01D03 = PWComparison(metabolomics_data, timeCourse_S01D03_samples, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_S01D03_filename, timeCourseS01D03);
	}

	if (run_timeCourse_S01D04) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData timeCourseS01D04 = PWComparison(metabolomics_data, timeCourse_S01D04_samples, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_S01D04_filename, timeCourseS01D04);
	}

	if (run_timeCourse_S01D05) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData timeCourseS01D05 = PWComparison(metabolomics_data, timeCourse_S01D05_samples, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_S01D05_filename, timeCourseS01D05);
	}

	if (run_timeCourse_S02D01) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData timeCourseS02D01 = PWComparison(metabolomics_data, timeCourse_S02D01_samples, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_S02D01_filename, timeCourseS02D01);
	}

	if (run_timeCourse_S02D02) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData timeCourseS02D02 = PWComparison(metabolomics_data, timeCourse_S02D02_samples, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_S02D02_filename, timeCourseS02D02);
	}

	if (run_timeCourse_S02D03) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData timeCourseS02D03 = PWComparison(metabolomics_data, timeCourse_S02D03_samples, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_S02D03_filename, timeCourseS02D03);
	}

	if (run_timeCourse_S02D04) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData timeCourseS02D04 = PWComparison(metabolomics_data, timeCourse_S02D04_samples, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_S02D04_filename, timeCourseS02D04);
	}

	if (run_timeCourse_S02D05) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData timeCourseS02D05 = PWComparison(metabolomics_data, timeCourse_S02D05_samples, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_S02D05_filename, timeCourseS02D05);
	}

	if (run_timeCourse_S01D01vsS01D02) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData timeCourseS01D01vsS01D02 = PWPrePostComparison(metabolomics_data, timeCourse_S01D01_samples, timeCourse_S01D02_samples, 4,
			10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_S01D01vsS01D02_filename, timeCourseS01D01vsS01D02);
	}

	if (run_timeCourse_S01D01vsS01D03) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData timeCourseS01D01vsS01D03 = PWPrePostComparison(metabolomics_data, timeCourse_S01D01_samples, timeCourse_S01D03_samples, 4,
			10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_S01D01vsS01D03_filename, timeCourseS01D01vsS01D03);
	}

	if (run_timeCourse_S01D01vsS01D04) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData timeCourseS01D01vsS01D04 = PWPrePostComparison(metabolomics_data, timeCourse_S01D01_samples, timeCourse_S01D04_samples, 4,
			10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_S01D01vsS01D04_filename, timeCourseS01D01vsS01D04);
	}

	if (run_timeCourse_S01D01vsS01D05) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData timeCourseS01D01vsS01D05 = PWPrePostComparison(metabolomics_data, timeCourse_S01D01_samples, timeCourse_S01D05_samples, 4,
			10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_S01D01vsS01D05_filename, timeCourseS01D01vsS01D05);
	}

	if (run_timeCourse_S02D01vsS02D02) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData timeCourseS02D01vsS02D02 = PWPrePostComparison(metabolomics_data, timeCourse_S02D01_samples, timeCourse_S02D02_samples, 4,
			10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_S02D01vsS02D02_filename, timeCourseS02D01vsS02D02);
	}

	if (run_timeCourse_S02D01vsS02D03) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData timeCourseS02D01vsS02D03 = PWPrePostComparison(metabolomics_data, timeCourse_S02D01_samples, timeCourse_S02D03_samples, 4,
			10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_S02D01vsS02D03_filename, timeCourseS02D01vsS02D03);
	}

	if (run_timeCourse_S02D01vsS02D04) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData timeCourseS02D01vsS02D04 = PWPrePostComparison(metabolomics_data, timeCourse_S02D01_samples, timeCourse_S02D04_samples, 4,
			10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_S02D01vsS02D04_filename, timeCourseS02D01vsS02D04);
	}

	if (run_timeCourse_S02D01vsS02D05) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData timeCourseS02D01vsS02D05 = PWPrePostComparison(metabolomics_data, timeCourse_S02D01_samples, timeCourse_S02D05_samples, 4,
			10000, 0.05, 1.0);

		// Export to file
		WritePWData(timeCourse_S02D01vsS02D05_filename, timeCourseS02D01vsS02D05);
	}
}

void main_statistics_controlsSummary(std::string blood_fraction = "PLT", bool run_controls = false)
{
	// define the data simulator
	BiochemicalReactionModel<float> metabolomics_data;

	// data dirs
	//std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	//std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	std::string data_dir = "/home/user/Data/";

	std::string
		controls_filename, controlsSampleSummary_filename, controlsFeatureSummary_filename;
	if (blood_fraction == "RBC") {
		// RBC filenames
		controls_filename = data_dir + "RBC_controls.csv";
		controlsSampleSummary_filename = data_dir + "RBC_controlsSampleSummary.csv";
		controlsFeatureSummary_filename = data_dir + "RBC_controlsFeatureSummary.csv";
	}
	else if (blood_fraction == "PLT") {
		// PLT filenames
		controls_filename = data_dir + "PLT_controls.csv";
		controlsSampleSummary_filename = data_dir + "PLT_controlsSampleSummary.csv";
		controlsFeatureSummary_filename = data_dir + "PLT_controlsFeatureSummary.csv";
	}
	else if (blood_fraction == "P") {
		// P filenames
		controls_filename = data_dir + "P_controls.csv";
		controlsSampleSummary_filename = data_dir + "P_controlsSampleSummary.csv";
		controlsFeatureSummary_filename = data_dir + "P_controlsFeatureSummary.csv";
	}

	if (run_controls) {
		// Read in the data
		PWData controls;
		ReadPWData(controls_filename, controls);

		// Summarize the data
		PWSampleSummaries pw_sample_summaries;
		PWFeatureSummaries pw_feature_summaries;
		PWTotalSummary pw_total_summary;
		PWSummary(controls, pw_sample_summaries, pw_feature_summaries, pw_total_summary);

		// Export to file
		WritePWSampleSummaries(controlsSampleSummary_filename, pw_sample_summaries);
		WritePWFeatureSummaries(controlsFeatureSummary_filename, pw_feature_summaries);
	}
}
void main_statistics_controls(std::string blood_fraction = "PLT", bool run_controls = false)
{
	// define the data simulator
	BiochemicalReactionModel<float> metabolomics_data;

	// data dirs
	//std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	//std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	std::string data_dir = "/home/user/Data/";

	std::string biochem_rxns_filename, metabo_data_filename, meta_data_filename,
		controls_filename;
	std::vector<std::string> invivo_samples, invitro_samples;
	if (blood_fraction == "RBC") {
		// RBC filenames
		biochem_rxns_filename = data_dir + "iAB_RBC_283.csv";
		metabo_data_filename = data_dir + "MetabolomicsData_RBC.csv";
		meta_data_filename = data_dir + "MetaData_prePost_RBC.csv";
		controls_filename = data_dir + "RBC_controls.csv";
		invivo_samples = { "RBC_36","RBC_140" };
		invitro_samples = { "S02_D01_RBC_25C_0hr","S01_D01_RBC_25C_0hr" };
	}
	else if (blood_fraction == "PLT") {
		// PLT filenames
		biochem_rxns_filename = data_dir + "iAT_PLT_636.csv";
		metabo_data_filename = data_dir + "MetabolomicsData_PLT.csv";
		meta_data_filename = data_dir + "MetaData_prePost_PLT.csv";
		controls_filename = data_dir + "PLT_controls.csv";
		invivo_samples = { "PLT_36","PLT_140" };
		invitro_samples = { "S02_D01_PLT_25C_0hr","S01_D01_PLT_25C_0hr" };
	}
	else if (blood_fraction == "P") {
		// P filenames
		biochem_rxns_filename = data_dir + "iAT_PLT_636.csv";
		metabo_data_filename = data_dir + "MetabolomicsData_P.csv";
		meta_data_filename = data_dir + "MetaData_prePost_P.csv";
		controls_filename = data_dir + "P_controls.csv";
		invivo_samples = { "P_36","P_140" };
		invitro_samples = { "S02_D01_P_25C_0hr","S01_D01_P_25C_0hr" };
	}

	// read in the data
	metabolomics_data.readBiochemicalReactions(biochem_rxns_filename);
	metabolomics_data.readMetabolomicsData(metabo_data_filename);
	metabolomics_data.readMetaData(meta_data_filename);
	metabolomics_data.findComponentGroupNames();
	metabolomics_data.findMARs();
	metabolomics_data.findLabels();

	if (run_controls) {
		// Find significant pair-wise MARS between pre/post samples (one pre vs one post)
		PWData controls = PWPrePostComparison(metabolomics_data, invivo_samples, invitro_samples, 2, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(controls_filename, controls);
	}
}
void main_statistics_preVsPost(std::string blood_fraction = "PLT", bool run_oneVSone = true, bool run_preVSpost = true, bool run_postMinPre = false)
{
	// define the data simulator
	BiochemicalReactionModel<float> metabolomics_data;

	// data dirs
	//std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	//std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	std::string data_dir = "/home/user/Data/";

	std::string biochem_rxns_filename, metabo_data_filename, meta_data_filename,
		oneVSonePre_filename, oneVSonePost_filename, preVSpost_filename, postMinPre_filename;
	std::vector<std::string> pre_samples, post_samples;
	if (blood_fraction == "RBC") {
		// RBC filenames
		biochem_rxns_filename = data_dir + "iAB_RBC_283.csv";
		metabo_data_filename = data_dir + "MetabolomicsData_RBC.csv";
		meta_data_filename = data_dir + "MetaData_prePost_RBC.csv";
		oneVSonePre_filename = data_dir + "RBC_oneVSonePre.csv";
		oneVSonePost_filename = data_dir + "RBC_oneVSonePost.csv";
		preVSpost_filename = data_dir + "RBC_preVSpost.csv";
		postMinPre_filename = data_dir + "RBC_postMinPre.csv";
		pre_samples = { "RBC_36","RBC_142","RBC_140","RBC_34","RBC_154","RBC_143","RBC_30","RBC_31","RBC_33","RBC_35","RBC_141" };
		post_samples = { "RBC_43","RBC_152","RBC_150","RBC_38","RBC_155","RBC_153","RBC_37","RBC_39","RBC_42","RBC_40","RBC_151" };
	}
	else if (blood_fraction == "PLT") {
		// PLT filenames
		biochem_rxns_filename = data_dir + "iAT_PLT_636.csv";
		metabo_data_filename = data_dir + "MetabolomicsData_PLT.csv";
		meta_data_filename = data_dir + "MetaData_prePost_PLT.csv";
		oneVSonePre_filename = data_dir + "PLT_oneVSonePre.csv";
		oneVSonePost_filename = data_dir + "PLT_oneVSonePost.csv";
		preVSpost_filename = data_dir + "PLT_preVSpost.csv";
		postMinPre_filename = data_dir + "PLT_postMinPre.csv";
		pre_samples = { "PLT_36","PLT_142","PLT_140","PLT_34","PLT_154","PLT_143","PLT_30","PLT_31","PLT_33","PLT_35","PLT_141" };
		post_samples = { "PLT_43","PLT_152","PLT_150","PLT_38","PLT_155","PLT_153","PLT_37","PLT_39","PLT_42","PLT_40","PLT_151" };
	}
	else if (blood_fraction == "P") {
		// P filenames
		biochem_rxns_filename = data_dir + "iAT_PLT_636.csv";
		metabo_data_filename = data_dir + "MetabolomicsData_P.csv";
		meta_data_filename = data_dir + "MetaData_prePost_P.csv";
		oneVSonePre_filename = data_dir + "P_oneVSonePre.csv";
		oneVSonePost_filename = data_dir + "P_oneVSonePost.csv";
		preVSpost_filename = data_dir + "P_preVSpost.csv";
		postMinPre_filename = data_dir + "P_postMinPre.csv";
		pre_samples = { "P_36","P_142","P_140","P_34","P_154","P_143","P_30","P_31","P_33","P_35","P_141" };
		post_samples = { "P_43","P_152","P_150","P_38","P_155","P_153","P_37","P_39","P_42","P_40","P_151" };
	}

	// read in the data
	metabolomics_data.readBiochemicalReactions(biochem_rxns_filename);
	metabolomics_data.readMetabolomicsData(metabo_data_filename);
	metabolomics_data.readMetaData(meta_data_filename);
	metabolomics_data.findComponentGroupNames();
	metabolomics_data.findMARs();
	metabolomics_data.findMARs(true, false);
	metabolomics_data.findMARs(false, true);
	metabolomics_data.removeRedundantMARs();
	metabolomics_data.findLabels();

	if (run_oneVSone) {
		// Find significant pair-wise MARS between each sample (one vs one Pre-ASA)
		PWData oneVSonePre = PWComparison(metabolomics_data, pre_samples, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(oneVSonePre_filename, oneVSonePre);

		// Find significant pair-wise MARS between each sample (one vs one Post-ASA)
		PWData oneVSonePost = PWComparison(metabolomics_data, post_samples, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(oneVSonePost_filename, oneVSonePost);
	}

	if (run_preVSpost) {
		// Find significant pair-wise MARS between pre/post samples (one pre vs one post)
		PWData preVSpost = PWPrePostComparison(metabolomics_data, pre_samples, post_samples, 11, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(preVSpost_filename, preVSpost);
	}

	if (run_postMinPre) {
		// Find significant pair-wise MARS between post-pre samples (post-pre vs post-pre) for each individual
		PWData postMinPre = PWPrePostDifference(metabolomics_data, pre_samples, post_samples, 11, 10000, 0.05, 1.0);

		// Export to file
		WritePWData(postMinPre_filename, postMinPre);
	}
}

// Main
int main(int argc, char** argv)
{
	main_statistics_controls("PLT", true);
	main_statistics_controls("RBC", true);
	main_statistics_controls("P", true);
	main_statistics_controlsSummary("PLT", true);
	main_statistics_controlsSummary("RBC", true);
	main_statistics_controlsSummary("P", true);
	main_statistics_timecourse("PLT",
		true, true, true, true, true,
		true, true, true, true, true,
		true, true, true, true,
		true, true, true, true);
	main_statistics_timecourse("P",
		true, true, true, true, true,
		true, true, true, true, true,
		true, true, true, true,
		true, true, true, true);
	main_statistics_timecourse("RBC",
		true, true, true, true, true,
		true, true, true, true, true,
		true, true, true, true,
		true, true, true, true);
	main_statistics_timecourseSummary("PLT",
		true, true, true, true, true,
		true, true, true, true, true,
		true, true, true, true,
		true, true, true, true);
	main_statistics_timecourseSummary("P",
		true, true, true, true, true,
		true, true, true, true, true,
		true, true, true, true,
		true, true, true, true);
	main_statistics_timecourseSummary("RBC",
		true, true, true, true, true,
		true, true, true, true, true,
		true, true, true, true,
		true, true, true, true);
	main_statistics_preVsPost("PLT", true, true, false);
	main_statistics_preVsPost("RBC", true, true, false);
	main_statistics_preVsPost("P", true, true, false);


	return 0;
}