	// --------------------------------------------------------------------------
//                   OpenMS -- Open-Source Mass Spectrometry
// --------------------------------------------------------------------------
// Copyright The OpenMS Team -- Eberhard Karls University Tuebingen,
// ETH Zurich, and Freie Universitaet Berlin 2002-2017.
//
// This software is released under a three-clause BSD license:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of any author or any participating institution
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
// For a full list of authors, refer to the file AUTHORS.
// --------------------------------------------------------------------------
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL ANY OF THE AUTHORS OR THE CONTRIBUTING
// INSTITUTIONS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// --------------------------------------------------------------------------
// $Maintainer: Douglas McCloskey $
// $Authors: Douglas McCloskey $
// --------------------------------------------------------------------------

#include <OpenMS/FORMAT/FileHandler.h>
#include <OpenMS/APPLICATIONS/TOPPBase.h>
#include <OpenMS/ANALYSIS/TARGETED/TargetedExperiment.h>

// files
#include <OpenMS/FORMAT/TraMLFile.h>
#include <OpenMS/FORMAT/MzMLFile.h>

using namespace OpenMS;
using namespace std;

//-------------------------------------------------------------
// Doxygen docu
	//-------------------------------------------------------------

/**
    @page UTILS_Dummy Dummy

    @brief Map survey to aquisition.
<CENTER>
    <table>
        <tr>
            <td ALIGN = "center" BGCOLOR="#EBEBEB"> pot. predecessor tools </td>
            <td VALIGN="middle" ROWSPAN=2> \f$ \longrightarrow \f$ Digestor \f$ \longrightarrow \f$</td>
            <td ALIGN = "center" BGCOLOR="#EBEBEB"> pot. successor tools </td>
        </tr>
        <tr>
            <td VALIGN="middle" ALIGN = "center" ROWSPAN=1> none (FASTA input) </td>
            <td VALIGN="middle" ALIGN = "center" ROWSPAN=1> @ref TOPP_IDFilter (peptide blacklist)</td>
        </tr>
    </table>
</CENTER>


    <B>The command line parameters of this tool are:</B>
    @verbinclude UTILS_Dummy.cli
    <B>INI file documentation of this tool:</B>
    @htmlinclude UTILS_Dummy.html
*/

// We do not want this class to show up in the docu:
/// @cond TOPPCLASSES

class TOPPDummy :
  public TOPPBase
{
public:
  TOPPDummy() :
    TOPPBase("Dummy", "Tool to map survey scans to aquisition.", false)
  {

  }

protected:
  void registerOptionsAndFlags_()
  {
    // register parameter in and set mzML as valid format
    registerInputFile_("in", "<file>", "", "input file");
    setValidFormats_("in", ListUtils::create<String>("mzML"));
    registerInputFile_("transitions", "<file>", "", "input file containing the transitions");
    setValidFormats_("transitions", ListUtils::create<String>("traML"));
    registerOutputFile_("out", "<file>", "", "Output file.");
    setValidFormats_("out", ListUtils::create<String>("mzML"));
    registerDoubleOption_("rt_tolerance", "<number>", 30.0, "Retention time tolerance in seconds.", false);
  }

  ExitCodes main_(int, const char**)
  {
    //-------------------------------------------------------------
    // parsing parameters
    //-------------------------------------------------------------
    String inputfile_name = getStringOption_("in");
    String transition_name = getStringOption_("transitions");
    String outputfile_name = getStringOption_("out");
    double rt_tolerance = getDoubleOption_("rt_tolerance");

    //-------------------------------------------------------------
    // reading input
    //-------------------------------------------------------------
    MSExperiment ms_exp;
    MzMLFile().load(inputfile_name, ms_exp);

    TargetedExperiment transition_exp;
    TraMLFile().load(transition_name, transition_exp);

    //-------------------------------------------------------------
    // annotation
    //-------------------------------------------------------------
    for (Size i = 0; i != ms_exp.getNrSpectra(); ++i)
    {
      PeakSpectrum & current_spectrum = ms_exp.getSpectrum(i);
      cout << "RT: " << current_spectrum.getRT() << endl;
      if (!current_spectrum.getPrecursors().empty())
      {
        cout << "MZ: " << current_spectrum.getPrecursors()[0].getMZ() << endl;        
      } 
    }

    for (Size i = 0; i != transitions_exp; ++i)
    {
      const ReactionMonitoringTransition & t = transitions_exp.getTransitions()[i];
      // TODO: read RT and precursor mass 
    }

    //-------------------------------------------------------------
    // writing output
    //-------------------------------------------------------------
    MzMLFile().store(outputfile_name, ms_exp);

    return EXECUTION_OK;
  }

};


	int main(int argc, const char** argv)
{
  TOPPDummy tool;
  return tool.main(argc, argv);
}

/// @endcond
