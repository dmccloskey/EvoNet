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
	MetDataSimClassification<float> metabolomics_data;

	// data dirs
	//std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	//std::string data_dir = "/home/user/Data/";

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
	MetDataSimClassification<float> metabolomics_data;

	// data dirs
	//std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	//std::string data_dir = "/home/user/Data/";

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
	MetDataSimClassification<float> metabolomics_data;

	// data dirs
	//std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	//std::string data_dir = "/home/user/Data/";

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
	MetDataSimClassification<float> metabolomics_data;

	// data dirs
	//std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	//std::string data_dir = "/home/user/Data/";

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
	MetDataSimClassification<float> metabolomics_data;

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
		 pre_samples = {"RBC_36","RBC_142","RBC_140","RBC_34","RBC_154","RBC_143","RBC_30","RBC_31","RBC_33","RBC_35","RBC_141"};
		 post_samples = {"RBC_43","RBC_152","RBC_150","RBC_38","RBC_155","RBC_153","RBC_37","RBC_39","RBC_42","RBC_40","RBC_151"};
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
void main_classification(std::string blood_fraction = "PLT", bool make_model = true)
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
	std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	//std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	//std::string data_dir = "/home/user/Data/";
	std::string model_name = "0_Metabolomics";

	std::string biochem_rxns_filename, metabo_data_filename, meta_data_filename;
	if (blood_fraction == "RBC") {
		// RBC filenames
		biochem_rxns_filename = data_dir + "iAB_RBC_283.csv";
		metabo_data_filename = data_dir + "MetabolomicsData_RBC.csv";
		meta_data_filename = data_dir + "MetaData_prePost_RBC.csv";
	}
	else if (blood_fraction == "PLT") {
		// PLT filenames
		biochem_rxns_filename = data_dir + "iAT_PLT_636.csv";
		metabo_data_filename = data_dir + "MetabolomicsData_PLT.csv";
		meta_data_filename = data_dir + "MetaData_prePost_PLT.csv";
	}
	else if (blood_fraction == "P") {
		// P filenames
		biochem_rxns_filename = data_dir + "iAB_RBC_283.csv";
		metabo_data_filename = data_dir + "MetabolomicsData_P.csv";
		meta_data_filename = data_dir + "MetaData_prePost_P.csv";
	}
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
void main_reconstruction()
{
	// define the multithreading parameters
	const int n_hard_threads = std::thread::hardware_concurrency();
	//const int n_threads = n_hard_threads / 2; // the number of threads
	//char threads_cout[512];
	//sprintf(threads_cout, "Threads for population training: %d, Threads for model training/validation: %d\n",
	//	n_hard_threads, 2);
	//std::cout << threads_cout;
	const int n_threads = 1;

	// define the population trainer parameters
	PopulationTrainerExt<float> population_trainer;
	population_trainer.setNGenerations(1);
	population_trainer.setNTop(3);
	population_trainer.setNRandom(3);
	population_trainer.setNReplicatesPerModel(3);

	// define the data simulator
	MetDataSimReconstruction<float> metabolomics_data;
	//std::string data_dir = "C:/Users/dmccloskey/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	//std::string data_dir = "C:/Users/domccl/Dropbox (UCSD SBRG)/Metabolomics_RBC_Platelet/";
	std::string data_dir = "/home/user/Data/";
	std::string biochem_rxns_filename = data_dir + "iAT_PLT_636.csv";
	std::string metabo_data_filename = data_dir + "MetabolomicsData_PLT.csv";
	std::string meta_data_filename = data_dir + "MetaData_prePost_PLT.csv";
	metabolomics_data.readBiochemicalReactions(biochem_rxns_filename);
	metabolomics_data.readMetabolomicsData(metabo_data_filename);
	metabolomics_data.readMetaData(meta_data_filename);
	metabolomics_data.findComponentGroupNames();
	metabolomics_data.findMARs();
	metabolomics_data.findLabels();

	// define the model input/output nodes
	const int n_input_nodes = metabolomics_data.reaction_ids_.size();
	const int n_output_nodes = metabolomics_data.reaction_ids_.size();
	std::vector<std::string> input_nodes, encoder_nodes, output_nodes;
	for (int i = 0; i < n_input_nodes; ++i)
		input_nodes.push_back("Input_" + std::to_string(i));
	for (int i = 0; i < n_output_nodes; ++i)
		output_nodes.push_back("Output_" + std::to_string(i));

	// define the model trainers and resources for the trainers
	std::vector<ModelInterpreterDefaultDevice<float>> model_interpreters;
	for (size_t i = 0; i < n_threads; ++i) {
		ModelResources model_resources = { ModelDevice(0, 1) };
		ModelInterpreterDefaultDevice<float> model_interpreter(model_resources);
		model_interpreters.push_back(model_interpreter);
	}
	ModelTrainerExt<float> model_trainer;
	model_trainer.setBatchSize(8);
	model_trainer.setMemorySize(1);
	model_trainer.setNEpochsTraining(1001);
	model_trainer.setNEpochsValidation(10);
	model_trainer.setVerbosityLevel(1);
	model_trainer.setLogging(false, false);
	model_trainer.setLossFunctions({ std::shared_ptr<LossFunctionOp<float>>(new MSEOp<float>()) });
	model_trainer.setLossFunctionGrads({ std::shared_ptr<LossFunctionGradOp<float>>(new MSEGradOp<float>()) });
	model_trainer.setOutputNodes({ output_nodes });

	// define the model logger
	ModelLogger<float> model_logger;

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
	const int population_size = 1;
	for (int i = 0; i<population_size; ++i)
	{
		// baseline model
		std::shared_ptr<WeightInitOp<float>> weight_init;
		std::shared_ptr<SolverOp<float>> solver;
		weight_init.reset(new RandWeightInitOp<float>(n_input_nodes));
		solver.reset(new AdamOp<float>(0.01, 0.9, 0.999, 1e-8));
		std::shared_ptr<LossFunctionOp<float>> loss_function(new MSEOp<float>());
		std::shared_ptr<LossFunctionGradOp<float>> loss_function_grad(new MSEGradOp<float>());
		Model<float> model;
		// TODO: define model
		model.setId(i);
		population.push_back(model);
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
	main_statistics_controls("PLT", true);
	//main_statistics_controls("RBC", true);
	//main_statistics_controls("P", true);
	main_statistics_controlsSummary("PLT", true);
	//main_statistics_controlsSummary("RBC", true);
	//main_statistics_controlsSummary("P", true);
	main_statistics_timecourse("PLT",
		true, true, true, true, true,
		true, true, true, true, true,
		true, true, true, true,
		true, true, true, true);
	//main_statistics_timecourse("P",
	//	true, true, true, true, true,
	//	true, true, true, true, true,
	//	true, true, true, true,
	//	true, true, true, true);
	//main_statistics_timecourse("RBC",
	//	true, true, true, true, true,
	//	true, true, true, true, true,
	//	true, true, true, true,
	//	true, true, true, true);
	main_statistics_timecourseSummary("PLT", 
		true, true, true, true, true,
		true, true, true, true, true,
		true, true, true, true,
		true, true, true, true);
	//main_statistics_timecourseSummary("P",
	//	true, true, true, true, true,
	//	true, true, true, true, true,
	//	true, true, true, true,
	//	true, true, true, true);
	//main_statistics_timecourseSummary("RBC",
	//	true, true, true, true, true,
	//	true, true, true, true, true,
	//	true, true, true, true,
	//	true, true, true, true);
	main_statistics_preVsPost("PLT", true, true, false);
	//main_statistics_preVsPost("RBC", true, true, false);
	//main_statistics_preVsPost("P", true, true, false);
	main_classification("PLT", true);
	main_reconstruction();
	return 0;
}