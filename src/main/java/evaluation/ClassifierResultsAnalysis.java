/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
package evaluation;

import ResultsProcessing.MatlabController;
import evaluation.storage.ClassifierResults;
import evaluation.storage.ClassifierResultsCollection;
import fileIO.OutFile;
import jxl.Workbook;
import jxl.WorkbookSettings;
import jxl.write.WritableCellFormat;
import jxl.write.WritableFont;
import jxl.write.WritableSheet;
import jxl.write.WritableWorkbook;
import statistics.tests.OneSampleTests;
import statistics.tests.TwoSampleTests;
import utilities.GenericTools;
import utilities.InstanceTools;
import utilities.StatisticalUtilities;
import utilities.generic_storage.Pair;
import weka.clusterers.XMeans;
import weka.core.Instances;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.util.function.Function;



/**
 *
 * This is a monster of a class, with some bad code and not enough documentation. It's improving over time however.
 If there are any questions about it, best bet would be to email me (see below).

 This class is given a much better front end/'api' in MultipleClassifierEvaluation.java. Users should almost always use
 that class for their comparative summaries of different classifiers. 

 The two functions from this class in particular a user would actually use in their code might be: 
 performFullEvaluation(...) and performTestAccEvalOnly(...), the former of which is the the 
 function wrapped by MultipleClassifierEvaluation

 Basically, this is a collection of static functions to analyse/handle COMPLETED (i.e no folds missing out 
 of those expected of the specified classifierXdatasetXfoldXsplit set) sets of results in ClassifierResults format

 For some reason, the excel workbook writer library i found/used makes xls files (instead of xlsx) and doesn't 
 support recent excel default fonts. Just open it and saveas xlsx if you want to

 Future work when wanted/needed would be to handle incomplete results (e.g random folds missing), more matlab figures over time, 
 and a MASSIVE refactor to remove the crap code
 *
 * @author James Large james.large@uea.ac.uk
 */
public class ClassifierResultsAnalysis {

    //actual parameters
    public static String expRootDirectory;
    public static boolean buildMatlabDiagrams = false;
    public static boolean testResultsOnly = false;



    public static PerformanceMetric trainTimeMetric = PerformanceMetric.buildTime;
    public static PerformanceMetric benchmarkedTrainTimeMetric = PerformanceMetric.buildTimeBenchmarked;

//        PerformanceMetric testTimeMetric = PerformanceMetric.totalTestTime;
//        PerformanceMetric benchmarkedTestTimeMetric = PerformanceMetric.totalTestTimeBenchmarked;
    public static PerformanceMetric testTimeMetric = PerformanceMetric.avgTestPredTime;
    public static PerformanceMetric benchmarkedTestTimeMetric = PerformanceMetric.avgTestPredTimeBenchmarked;

    //    public static PerformanceMetric estimateTimeMetric = PerformanceMetric.additionalTimeForEstimate;
    //    public static PerformanceMetric benchmarkedEstimateTimeMetric = PerformanceMetric.additionalTimeForEstimateBenchmarked;
    public static PerformanceMetric estimateTimeMetric = PerformanceMetric.fromScratchEstimateTime;
    public static PerformanceMetric benchmarkedEstimateTimeMetric = PerformanceMetric.fromScratchEstimateTimeBenchmarked;

    public static PerformanceMetric memoryMaxMetric = PerformanceMetric.memory;

    public static List<PerformanceMetric> allComputationalMetrics = Arrays.asList(trainTimeMetric, benchmarkedTrainTimeMetric, testTimeMetric,
            benchmarkedTestTimeMetric, memoryMaxMetric, estimateTimeMetric, benchmarkedEstimateTimeMetric);


    //final id's and path suffixes
    protected static final String matlabFilePath = "src/main/matlab/";
    protected static final String pairwiseScatterDiaPath = "dias_PairwiseScatter/";
    protected static final String cdDiaFolderName = "dias_CriticalDifference/";
    protected static final String pairwiseCDDiaDirName = "pairwise/";
    protected static final String friedmanCDDiaDirName = "friedman/";

    //todo being used for both raw and benchmarked 2019/10/21
    //eval_timings... editing folder name before own ana, then resetting. fix.
    protected static String computationalDiaFolderName = "dias_Timings";
    protected static String computationalDiaFolderName_raw = "dias_ComputationalResourcesRAW";
    protected static String computationalDiaFolderName_benchmark = "dias_ComputationalResourcesBENCHMARKED";

    public static final double FRIEDMANCDDIA_PVAL = 0.05;
    public static final String testLabel = "TEST";
    public static final String trainLabel = "TRAIN";
    public static final String trainTestDiffLabel = "TRAINTESTDIFFS";
    public static final String estimateLabel = "ESTIMATE";
    public static final String clusterGroupingIdentifier = "PostHocXmeansClustering";


    public static class ClassifierEvaluation  {
        public String classifierName;
        public ClassifierResults[][] testResults; //[dataset][fold]
        public ClassifierResults[][] trainResults; //[dataset][fold]

        public ClassifierEvaluation(String name, ClassifierResults[][] testResults, ClassifierResults[][] trainResults) {
            this.classifierName = name;
            this.testResults = testResults;
            this.trainResults = trainResults;
        }
    }


    /**
     * THIS IS THE METHOD YOU'D ACTUALLY USE, the public 'actually do stuff' method   
     *
     * @param outPath a single directory, called expName, will be made in this location, containing the analysis
     * @param expname this will be the name of the parent folder that is made and will appear on a number of files
     * @param metrics a list of PerformanceMetrics that effectively are able to summarise a ClassifierResults 
     *          object into a single double, the prediction set's score. e.g. accuracy for these predictions
     *          These metrics will also have indications of how comparisons of this metric should be calculated and represented
     * @param results a ClassifierResultsCollection containing test (and optionally train) results of 1/more classifiers on 1/more datasets over 1/more resamples.
     *          Classifier and dataset names are retrieved from this object
     * @param dsetGroupings Optional, a map { grouping name, groupings } of maps { group name, datasets in groups } that describe different subsets of 
     *          the data within which to repeat the analysis, e.g one group might be 2class datasets vs multiclass datasets. The analysis would 
     *          aid in seeing if one classifier has a competitive advantage over the others within different data characteristics/groupings
     */
    public static void performFullEvaluation(
            String outPath,
            String expname,
            List<PerformanceMetric> metrics,
            ClassifierResultsCollection results,
            Map<String, Map<String, String[]>> dsetGroupings)
    {
        //hacky housekeeping
        MultipleClassifiersPairwiseTest.beQuiet = true;
        OneSampleTests.beQuiet = true;

        outPath = outPath.replace("\\", "/");
        if (!outPath.endsWith("/"))
            outPath+="/";
        outPath += expname + "/";
        new File(outPath).mkdirs();

        expRootDirectory = outPath;

        OutFile bigSummary = new OutFile(outPath + expname + "_BIGglobalSummary.csv");
        OutFile smallSummary = new OutFile(outPath + expname + "_SMALLglobalSummary.csv");

        //this will collect the clique arrays for each metric as found by pairwise stats,
        //so that they can later be passed to the cd dia maker 
        ArrayList<String> statCliquesForCDDias = new ArrayList<>();

        // START USER DEFINED STATS
        for (PerformanceMetric metric : metrics) {
            String[] summary = null;
            try {
                summary = eval_metric(outPath, expname, results, metric, dsetGroupings);
            } catch (Exception fnf) {
                System.out.println("Something went wrong while writing " + metric + "files, likely later stages of analysis could "
                        + "not find files that should have been made "
                        + "internally in earlier stages of the pipeline, FATAL");
                fnf.printStackTrace();
                System.exit(0);
            }

            bigSummary.writeString(metric.name+":");
            bigSummary.writeLine(summary[0]);

            smallSummary.writeString(metric.name+":");
            smallSummary.writeLine(summary[1]);

            if (summary[2] != null)
                statCliquesForCDDias.add(summary[2]);
        }
        // END USER DEFINED STATS






        // START TIMINGS 
        //timings will attempt to always be summarised if they are present, so handle them here as a special case
        //and add them onto the list of metrics
        ArrayList<String[]> compResourceSummaries = new ArrayList<>();
        try {
            String[][] compResourcesSummaryRaw = eval_CompResourcesRAW(outPath, expname, results, null); //dont bother with groupings for timings
            compResourceSummaries.addAll(Arrays.asList(compResourcesSummaryRaw));

        } catch (FileNotFoundException fnf) {
            System.out.println("Something went wrong while writing RAW timing files, likely "
                    + "later stages of analysis could not find files that should have been made"
                    + "internally in earlier stages of the pipeline, FATAL");
            fnf.printStackTrace();
            System.exit(0);
        } catch (Exception ex) {
            ex.printStackTrace();
            System.out.println("Something went wrong while writing RAW timing files. But NOT "
                    + "a filenotfound error. Either timings werent found, some NaN errors occurred,"
                    + " etc. Todo look into cases of this as they crop up.\n"
                    + "CONTINUING THE ANALYSIS FOR NOW, but ignoring the RAW timings");
        }

        // TODO proper support for benchmarked timings, link up to the diagram creation code, global summary files, etc
        // currently standalone
        try {
            String[][] compResourcesSummaryBenchmarked = eval_CompResourcesBENCHMARKED(outPath, expname, results, null); //dont bother with groupings for timings
            compResourceSummaries.addAll(Arrays.asList(compResourcesSummaryBenchmarked));
        } catch (FileNotFoundException fnf) {
            System.out.println("Something went wrong while writing BENCHMARKED timing files, likely "
                    + "later stages of analysis could not find files that should have been made"
                    + "internally in earlier stages of the pipeline, FATAL");
            fnf.printStackTrace();
            System.exit(0);
        } catch (Exception ex) {
            ex.printStackTrace();
            System.out.println("Something went wrong while writing BENCHMARKED timing files. But NOT "
                    + "a filenotfound error. Either timings werent found, some NaN errors occurred,"
                    + " etc. Todo look into cases of this as they crop up.\n"
                    + "CONTINUING THE ANALYSIS FOR NOW, but ignoring the BENCHMARKED timings");
        }
        //end benchmarked

        //TODO clean all of this timing stuff up, it's jsut another layer of hacky nonsense.
        //just need a CLEAN break of 'does everything have timings? do all the timing analysis.
        //does ANYTHING not have timings? do NONE of the timing analysis

        //using the presence of summaries for train and test timings as an indicator that they are present
        List<PerformanceMetric> compMetrics = new ArrayList<>();

        if (compResourceSummaries != null) {
            compMetrics.add(PerformanceMetric.buildTime);
            compMetrics.add(testTimeMetric);
            compMetrics.add(memoryMaxMetric);
            compMetrics.add(PerformanceMetric.buildTimeBenchmarked);
            compMetrics.add(benchmarkedTestTimeMetric);

            // if there are train results, get the time needed to make that estimate in a couple different forms
            if (Arrays.asList(results.getSplits()).contains("train")) { //wrong on so many levels
                compMetrics.add(PerformanceMetric.totalBuildPlusEstimateTime);
                compMetrics.add(PerformanceMetric.totalBuildPlusEstimateTimeBenchmarked);
                compMetrics.add(PerformanceMetric.additionalTimeForEstimate);
                compMetrics.add(PerformanceMetric.additionalTimeForEstimateBenchmarked);
            }

            for (int j = compResourceSummaries.size()-1; j >= 0; j--) {
                String label = compMetrics.get(j).name;
                if (compResourceSummaries.get(j) != null) {
                    //present, so add on automatically to the list of metrics for passing around to spreadsheet/image makers etc
                    metrics.add(compMetrics.get(j));

                    bigSummary.writeString(label + ":");
                    bigSummary.writeLine(compResourceSummaries.get(j)[0]);

                    smallSummary.writeString(label + ":");
                    smallSummary.writeLine(compResourceSummaries.get(j)[1]);

                    statCliquesForCDDias.add(compResourceSummaries.get(j)[2]);
                }
                else {
                    //not present, ignore, and remove from list of time-specific metrics
                    //to be passed to the timing dia creator
                    compMetrics.remove(j);

                    bigSummary.writeString(label + ":  MISSING\n\n");
                    smallSummary.writeString(label + ": MISSING\n\n");
                }
            }
        }
        //END TIMINGS

        bigSummary.closeFile();
        smallSummary.closeFile();

        jxl_buildResultsSpreadsheet(outPath, expname, metrics);

        String[] statCliquesForCDDiasArr = statCliquesForCDDias.toArray(new String[] { });
        if(buildMatlabDiagrams) {
            MatlabController proxy = MatlabController.getInstance();
            proxy.eval("addpath(genpath('"+matlabFilePath+"'))");
            matlab_buildCompResourcesDias(compMetrics);
            matlab_buildCDDias(expname, statCliquesForCDDiasArr);
            matlab_buildPairwiseScatterDiagrams(outPath, expname, metrics, results.getDatasetNamesInOutput());
        }
    }

    /**
     * Essentially just a wrapper for what eval_metricOnSplit does, in the simple case that we just have a 3d array of test accs and want summaries for it
     Mostly for legacy results not in the classifier results file format
     */
    public static void performTestAccEvalOnly(String outPath, String filename, double[][][] testFolds, String[] cnames, String[] dsets, Map<String, Map<String, String[]>> dsetGroupings) throws FileNotFoundException {
        eval_metricOnSplit(outPath, filename, null, testLabel, PerformanceMetric.acc, testFolds, cnames, dsets, dsetGroupings);
    }
















    protected static void writeTableFile_ClassifierDataset(String filename, String tableName, double[][] scores, String[] cnames, String[] dsets) {
        OutFile out=new OutFile(filename);
        out.writeLine(tableName + fileHelper_tabulate(scores, cnames, dsets));
        out.closeFile();
    }

    protected static void writeRawTableFile_ClassifierDataset(String filename, double[][] scores, String[] cnames) {
        OutFile out=new OutFile(filename);
        out.writeLine(fileHelper_tabulateRaw(scores, cnames));
        out.closeFile();
    }

    protected static void writeTableFile_ClassifierDatasetFolds(String filename, String tableName, double[][][] scores, String[] cnames, String[] dsets) {
        OutFile out=new OutFile(filename);
        out.writeLine(tableName + fileHelper_tabulate(scores, cnames, dsets));
        out.closeFile();
    }

    protected static void writeRawTableFile_ClassifierDatasetFolds(String filename, double[][][] scores, String[] cnames) {
        OutFile out=new OutFile(filename);
        out.writeLine(fileHelper_tabulateRaw(scores, cnames));
        out.closeFile();
    }

    /**
     * also writes separate win/draw/loss files now
     */
    protected static String[] eval_metricOnSplitStatsFile(String outPath, String evalSet, PerformanceMetric metric, double[][][] statPerFold, double[][] statPerDset, double[][] ranks, double[][] stddevsFoldAccs, String[] cnames, String[] dsets) {
        String splitMetricLabel = evalSet + metric.toString();

        StringBuilder shortSummaryStats = new StringBuilder();
        shortSummaryStats.append(fileHelper_header(cnames)).append("\n");
        shortSummaryStats.append("Avg"+splitMetricLabel+":").append(util_mean(statPerDset)).append("\n");
        shortSummaryStats.append("Avg"+splitMetricLabel+"_RANK:").append(util_mean(ranks)).append("\n");

        StringBuilder longSummaryStats = new StringBuilder();
        longSummaryStats.append(splitMetricLabel).append(fileHelper_header(cnames)).append("\n");
        longSummaryStats.append("Avg"+splitMetricLabel+"OverDsets:").append(util_mean(statPerDset)).append("\n");
        longSummaryStats.append("Avg"+splitMetricLabel+"RankOverDsets:").append(util_mean(ranks)).append("\n");
        longSummaryStats.append("StddevOf"+splitMetricLabel+"OverDsets:").append(util_stddev(statPerDset)).append("\n");
        longSummaryStats.append("AvgOfStddevsOf"+splitMetricLabel+"OverDsetFolds:").append(util_mean(stddevsFoldAccs)).append("\n");
        longSummaryStats.append("StddevsOf"+splitMetricLabel+"RanksOverDsets:").append(util_stddev(ranks)).append("\n");

        String[] wdl =      eval_winsDrawsLosses(statPerDset, cnames, dsets);
        String[] sig01wdl = eval_sigWinsDrawsLosses(0.01, statPerDset, statPerFold, cnames, dsets);
        String[] sig05wdl = eval_sigWinsDrawsLosses(0.05, statPerDset, statPerFold, cnames, dsets);



        String wdlDir = outPath+"/WinsDrawsLosses/";
        (new File(wdlDir)).mkdir();
        OutFile outwdl = null;
        outwdl = new OutFile(wdlDir + splitMetricLabel + "WinDrawLoss_LIST.csv");
        outwdl.writeLine(wdl[1]);
        outwdl.closeFile();
        outwdl = new OutFile(wdlDir + splitMetricLabel + "WinDrawLoss_LIST_Sig01.csv");
        outwdl.writeLine(sig01wdl[1]);
        outwdl.closeFile();
        outwdl = new OutFile(wdlDir + splitMetricLabel + "WinDrawLoss_LIST_Sig05.csv");
        outwdl.writeLine(sig05wdl[1]);
        outwdl.closeFile();

        outwdl = new OutFile(wdlDir + splitMetricLabel + "WinDrawLoss_TABLE.csv");
        outwdl.writeLine(wdl[2]);
        outwdl.closeFile();
        outwdl = new OutFile(wdlDir + splitMetricLabel + "WinDrawLoss_TABLE_Sig01.csv");
        outwdl.writeLine(sig01wdl[2]);
        outwdl.closeFile();
        outwdl = new OutFile(wdlDir + splitMetricLabel + "WinDrawLoss_TABLE_Sig05.csv");
        outwdl.writeLine(sig05wdl[2]);
        outwdl.closeFile();

        String summaryFname = outPath + fileNameBuild_summaryFile(evalSet,metric);
        OutFile out=new OutFile(summaryFname);

        out.writeLine(longSummaryStats.toString());
        out.writeLine(wdl[0]);
        out.writeLine("\n");
        out.writeLine(sig01wdl[0]);
        out.writeLine("\n");
        out.writeLine(sig05wdl[0]);
        out.writeLine("\n");

        String cliques = "";
        String avgsFile = outPath + fileNameBuild_avgsFile(evalSet, metric);
        try {
            out.writeLine(MultipleClassifiersPairwiseTest.runTests(avgsFile).toString());
//            out.writeLine(MultipleClassifiersPairwiseTest.runTests(outPath+filename+"_"+splitMetricLabal+".csv").toString());
            cliques = MultipleClassifiersPairwiseTest.printCliques();
            out.writeLine("\n\n" + cliques);
        } catch (Exception e) {
            System.err.println("\n\n");
            System.err.println("*****".replace("*", "*****"));

            System.err.println("MultipleClassifiersPairwiseTest.runTests() failed. Almost certainly this is because there were"
                    + "too many ties/duplicates within one of the pairwise tests and then an index out of bounds error was thrown. "
                    + "This will be fixed at some point. The analysis will CARRY ON, and everything that is successfully printed out "
                    + "IS CORRECT, however whatever particular table that test would have been summarised as is missing from your files.");
            System.err.println("avgs filename = "+avgsFile);
            e.printStackTrace();

            System.err.println("*****".replace("*", "*****"));
            System.err.println("\n\n");
        }

        out.closeFile();

        return new String[] { longSummaryStats.toString(), shortSummaryStats.toString(), cliques };
    }

    protected static String fileNameBuild_cd(String filename, String statistic) {
        return "cd_"+filename+"_"+statistic+"S";
    }
    protected static String fileNameBuild_pws(String filename, String statistic) {
        return "pws_"+filename+"_"+statistic+"S";
    }
    protected static String fileNameBuild_pwsInd(String c1, String c2, String statistic) {
        return "pws_"+c1+"VS"+c2+"_"+statistic+"S";
    }
    protected static String fileNameBuild_avgsFile(String evalSet, PerformanceMetric metric) {
        return evalSet+metric+"_"+(metric.takeMean?"MEANS":"MEDIANS")+".csv";
    }
    protected static String fileNameBuild_ranksFile(String evalSet, PerformanceMetric metric) {
        return evalSet+metric+"_RANKS.csv";
    }
    protected static String fileNameBuild_stddevFile(String evalSet, PerformanceMetric metric) {
        return evalSet+metric+"_STDDEV.csv";
    }
    protected static String fileNameBuild_rawAvgsFile(String evalSet, PerformanceMetric metric) {
        return evalSet+metric+"_RAW.csv";
    }
    protected static String fileNameBuild_summaryFile(String evalSet, PerformanceMetric metric) {
        return evalSet+metric+"_SUMMARY.csv";
    }
    protected static String fileNameBuild_wdlFile(String evalSet, PerformanceMetric metric) {
        return evalSet+metric+"_SUMMARY.csv";
    }

    protected static String[] eval_metricOnSplit(String outPath, String filename, String groupingName, String evalSet, PerformanceMetric metric, double[][][] foldVals, String[] cnames, String[] dsets, Map<String, Map<String, String[]>> dsetGroupings) throws FileNotFoundException {
        outPath += evalSet + "/";
        if (groupingName != null && !groupingName.equals(""))
            outPath += groupingName + "/";

        //BEFORE ordering, write the individual folds files
        eval_perFoldFiles(outPath+evalSet+"FOLD"+metric+"S/", foldVals, cnames, dsets, evalSet);

        double[][] dsetVals = findAvgsOverFolds(foldVals, metric.takeMean);
        double[][] stddevsFoldVals = findStddevsOverFolds(foldVals);
        double[][] ranks = findRanks(dsetVals, metric.maximise);

        int[] ordering = findOrdering(ranks);
        //ordering is now an array of value referring to the rank-order of the element at each index
        //e.g [1, 4, 2, 3] means that the first (in index 0) classifier is best, third is next, then fourth, then second

        //now order all the info (essentially in parallel arrays) we've collected by the classifier's ranks
        //such that e.g the data referring to the first classifier is still in index 0, the data referring to
        //the second classifier is moved to index 1, etc
        ranks = util_order(ranks, ordering);
        cnames = util_order(cnames, ordering);
        foldVals = util_order(foldVals, ordering);
        dsetVals = util_order(dsetVals, ordering);
        stddevsFoldVals = util_order(stddevsFoldVals, ordering);

        if (evalSet.equalsIgnoreCase("TEST") || allComputationalMetrics.contains(metric)) {
            //qol for cd dia creation, make a copy of all the raw test stat files in a common folder, one for pairwise, one for freidman
            String cdFolder = expRootDirectory + cdDiaFolderName;
            (new File(cdFolder)).mkdirs();
            OutFile out = new OutFile(cdFolder+"readme.txt");
            out.writeLine("remember that nlls are auto-negated now for cd dia ordering\n");
            out.writeLine("and that basic notepad wont show the line breaks properly, view (cliques especially) in notepad++");
            out.closeFile();
            for (String subFolder : new String[] { pairwiseCDDiaDirName, friedmanCDDiaDirName }) {
                (new File(cdFolder+subFolder+"/")).mkdirs();
                String cdName = cdFolder+subFolder+"/"+fileNameBuild_cd(filename,metric.name)+".csv";

                //meta hack for qol, negate the nll (sigh...) for correct ordering on dia
                //ALSO now negating the timings, smaller = better
                if (!metric.maximise) {
                    double[][] negatedDsetVals = new double[dsetVals.length][dsetVals[0].length];
                    for (int i = 0; i < dsetVals.length; i++) {
                        for (int j = 0; j < dsetVals[i].length; j++) {
                            negatedDsetVals[i][j] = dsetVals[i][j] * -1;
                        }
                    }
                    writeRawTableFile_ClassifierDataset(cdName, negatedDsetVals, cnames);
                } else {
                    writeRawTableFile_ClassifierDataset(cdName, dsetVals, cnames);
                }
            } //end cd dia qol

            //qol for pairwisescatter dia creation, make a copy of the test stat files
            String pwsFolder = expRootDirectory + pairwiseScatterDiaPath;
            (new File(pwsFolder)).mkdirs();
            String pwsName = pwsFolder+fileNameBuild_pws(filename,metric.name)+".csv";
            writeRawTableFile_ClassifierDataset(pwsName, dsetVals, cnames);
            //end pairwisescatter qol

            //qol for timing dia creation, make a copy of the avgs files with headers
            if (allComputationalMetrics.contains(metric)) {
                String compDir = expRootDirectory+ computationalDiaFolderName + "/";
                (new File(compDir)).mkdirs();
                String fname = compDir+fileNameBuild_avgsFile(evalSet,metric);
                writeTableFile_ClassifierDataset(fname, evalSet+metric, dsetVals, cnames, dsets);
            }
            //end timing dia qol
        }


        writeTableFile_ClassifierDataset(outPath + fileNameBuild_ranksFile(evalSet, metric), evalSet+metric+"RANKS", ranks, cnames, dsets);
        writeTableFile_ClassifierDataset(outPath + fileNameBuild_avgsFile(evalSet, metric), evalSet+metric, dsetVals, cnames, dsets);
        writeRawTableFile_ClassifierDataset(outPath + fileNameBuild_rawAvgsFile(evalSet, metric), dsetVals, cnames); //for matlab stuff
        writeTableFile_ClassifierDataset(outPath + fileNameBuild_stddevFile(evalSet, metric), evalSet+metric+"STDDEVS", stddevsFoldVals, cnames, dsets);

        String[] groupingSummary = { "" };
        if (dsetGroupings != null && dsetGroupings.size() != 0)
            groupingSummary = eval_metricDsetGroups(outPath, filename, evalSet, metric, foldVals, cnames, dsets, dsetGroupings);


        String[] summaryStrings = {};
        summaryStrings = eval_metricOnSplitStatsFile(outPath, evalSet, metric, foldVals, dsetVals, ranks, stddevsFoldVals, cnames, dsets);

        //write these even if not actually making the dias this execution, might manually make them later
        writeCliqueHelperFiles(expRootDirectory + cdDiaFolderName + pairwiseCDDiaDirName, filename, metric, summaryStrings[2]);

        //this really needs cleaning up at some point... jsut make it a list and stop circlejerking to arrays
        String[] summaryStrings2 = new String[summaryStrings.length+groupingSummary.length];
        int i = 0;
        for ( ; i < summaryStrings.length; i++)
            summaryStrings2[i] = summaryStrings[i];
        for (int j = 0; j < groupingSummary.length; j++)
            summaryStrings2[i] = groupingSummary[j];

        return summaryStrings2;
    }

    public static String[] eval_metricDsetGroups(String outPathBase, String filename, String evalSet, PerformanceMetric metric, double[][][] foldVals, String[] cnames, String[] dsets, Map<String, Map<String, String[]>> dsetGroupings) throws FileNotFoundException {
        String outPath = expRootDirectory + "DatasetGroupings/";
//        String outPath = outPathBase + "DatasetGroupings/";
        (new File(outPath)).mkdir();

        //for each grouping method
        for (Map.Entry<String, Map<String, String[]>> dsetGroupingMethodEntry : dsetGroupings.entrySet()) {
            String groupingMethodName = dsetGroupingMethodEntry.getKey();
            String groupingMethodPath = outPath + groupingMethodName + "/";
            (new File(groupingMethodPath+metric.name+"/"+evalSet+"/")).mkdirs();

            Map<String, String[]> dsetGroupingMethod = dsetGroupingMethodEntry.getValue();

            if (groupingMethodName.equals(clusterGroupingIdentifier)) {
                //if clustering is to be done, build the groups now.
                //can't 'put' these groups back into the dsetGroupings map
                //since we'd be editing a map that we're currently iterating over
                //EDIT: actually, jsut move this process outside the for loop as
                //a preprocess step, if the need ever arises

                assert(dsetGroupingMethod == null);
                dsetGroupingMethod = new HashMap<>();

                int[] assignments = dsetGroups_clusterDsetResults(StatisticalUtilities.averageFinalDimension(foldVals));

                //puts numClusters as final element
                assert(assignments.length == dsets.length+1);
                int numClusters = assignments[dsets.length];

                String[] clusterNames = new String[numClusters];
                String[][] clusterDsets = new String[numClusters][];

                //would generally prefer to jsut loop once over the assignments array, but that would
                //require we already know the size of each cluster and/or wankery with array lists
                for (int cluster = 0; cluster < numClusters; cluster++) {
                    ArrayList<String> dsetAlist = new ArrayList<>();
                    for (int dset = 0; dset < dsets.length; dset++)
                        if (assignments[dset] == cluster)
                            dsetAlist.add(dsets[dset]);

                    clusterNames[cluster] = "Cluster " + (cluster+1);
                    clusterDsets[cluster] = dsetAlist.toArray(new String[] { });
                    dsetGroupingMethod.put(clusterNames[cluster], clusterDsets[cluster]);
                }

                //writing all the clusters to one file start here
                OutFile allDsetsOut = new OutFile(groupingMethodPath+metric.name+"/"+evalSet+"/" + "clusters.csv");

                for (int cluster = 0; cluster < numClusters; cluster++)
                    allDsetsOut.writeString(clusterNames[cluster] + ",");
                allDsetsOut.writeLine("");

                //printing variable length 2d array in table form, columns = clusters, rows = dsets
                int dsetInd = 0;
                boolean allDone = false;
                while (!allDone) {
                    allDone = true;
                    for (int cluster = 0; cluster < numClusters; cluster++) {
                        if (dsetInd < clusterDsets[cluster].length) {
                            allDsetsOut.writeString(clusterDsets[cluster][dsetInd]);
                            allDone = false;
                        }
                        allDsetsOut.writeString(",");
                    }
                    allDsetsOut.writeLine("");
                    dsetInd++;
                }
                allDsetsOut.closeFile();
                //writing all the clusters to one file end here

                String clusterGroupsPath = groupingMethodPath+metric+"/"+evalSet+"/" + "DsetClustersTxtFiles/";
                (new File(clusterGroupsPath)).mkdir();

                //writing each individual clsuter file start here
                for (int cluster = 0; cluster < numClusters; cluster++) {
                    OutFile clusterFile = new OutFile(clusterGroupsPath + clusterNames[cluster] + ".txt");
                    for (String dset : clusterDsets[cluster])
                        clusterFile.writeLine(dset);
                    clusterFile.closeFile();
                }
            }

            int numGroups = dsetGroupingMethod.size();
            String[] groupNames = new String[numGroups];

            //using maps for this because classifiernames could be in different ordering based on rankings
            //within each group. ordering of dataset groups temselves is constant though. jsut skips
            //annoying/bug inducing housekeeping of indices
            Map<String, double[]> groupWins = new HashMap<>(); //will reflect ties, e.g if 2 classifiers tie for first rank, each will get 'half' a win
            Map<String, double[]> groupAccs = new HashMap<>();
            for (int i = 0; i < cnames.length; i++) {
                groupWins.put(cnames[i], new double[numGroups]);
                groupAccs.put(cnames[i], new double[numGroups]);
            }

            //for each group in this grouping method
            StringBuilder [] groupSummaryStringBuilders = new StringBuilder[numGroups];
            int groupIndex = 0;

            for (Map.Entry<String, String[]> dsetGroup : dsetGroupingMethod.entrySet()) {
                String groupName = dsetGroup.getKey();
                groupNames[groupIndex] = groupName;
//                String groupPath = groupingMethodPath + groupName + "/";
//                (new File(groupPath)).mkdir();

                //perform group analysis
                String[] groupDsets = dsetGroup.getValue();
                double[][][] groupFoldVals = dsetGroups_collectDsetVals(foldVals, dsets, groupDsets);
                String groupFileName = filename + "-" + groupName + "-";
//                String[] groupSummaryFileStrings = eval_metricOnSplit(groupPath+statName+"/", groupFileName, groupName, evalSet, statName, groupFoldVals, cnames, groupDsets, null);
                String[] groupSummaryFileStrings = eval_metricOnSplit(groupingMethodPath+metric+"/", groupFileName, groupName, evalSet, metric, groupFoldVals, cnames, groupDsets, null);

                //collect the accuracies for the dataset group
                String[] classifierNamesLine = groupSummaryFileStrings[1].split("\n")[0].split(",");
                assert(classifierNamesLine.length-1 == cnames.length);
                String[] accLineParts = groupSummaryFileStrings[1].split("\n")[1].split(",");
                for (int i = 1; i < accLineParts.length; i++) { //i=1 => skip the row fileHelper_header
                    double[] accs = groupAccs.get(classifierNamesLine[i]);
                    accs[groupIndex] = Double.parseDouble(accLineParts[i]);
                    groupAccs.put(classifierNamesLine[i], accs);
                }

                //collect the wins for the group
                Scanner ranksFileIn = new Scanner(new File(groupingMethodPath+metric+"/"+evalSet+"/"+groupName+"/"+groupFileName+"_"+evalSet+metric+"RANKS.csv"));
                classifierNamesLine = ranksFileIn.nextLine().split(",");
                double[] winCounts = new double[classifierNamesLine.length];
                while (ranksFileIn.hasNextLine()) {
                    //read the ranks on this dataset
                    String[] ranksStr = ranksFileIn.nextLine().split(",");
                    double[] ranks = new double[ranksStr.length];
                    ranks[0] = Double.MAX_VALUE;
                    for (int i = 1; i < ranks.length; i++)
                        ranks[i] = Double.parseDouble(ranksStr[i]);

                    //there might be ties, so cant just look for the rank "1"
                    List<Integer> minRanks = util_min(ranks);
                    for (Integer minRank : minRanks)
                        winCounts[minRank] += 1.0 / minRanks.size();
                }
                ranksFileIn.close();

                for (int i = 1; i < winCounts.length; i++) {
                    double[] wins = groupWins.get(classifierNamesLine[i]);
                    wins[groupIndex] = winCounts[i];
                    groupWins.put(classifierNamesLine[i], wins);
                }

                //build the summary string
                StringBuilder sb = new StringBuilder("Group: " +groupName + "\n");
                sb.append(groupSummaryFileStrings[1]);

                //when will the hacks ever end?
                String cliques = groupSummaryFileStrings[2];
                cliques = cliques.replace("cliques = [", "cliques=,").replace("]", ""); //remove spaces in 'title' before next step
                cliques = cliques.replace(" ", ",").replace("\n", "\n,"); //make vals comma separated, to line up in csv file
                sb.append("\n"+cliques);

                groupSummaryStringBuilders[groupIndex] = sb;
                groupIndex++;
            }

            String groupMethodSummaryFilename = groupingMethodPath + filename + "_" + groupingMethodName + "_" + evalSet + metric + ".csv";
            dsetGroups_writeGroupingMethodSummaryFile(groupMethodSummaryFilename, groupSummaryStringBuilders, cnames, groupNames, groupWins, groupAccs);
        }

        return new String[] { };
    }

    public static void dsetGroups_writeGroupingMethodSummaryFile(String filename, StringBuilder [] groupSummaryStringBuilders, String[] cnames, String[] groupNames,
                                                                 Map<String, double[]> groupWins, Map<String, double[]> groupAccs) {

        OutFile groupingMethodSummaryFile = new OutFile(filename);
        for (StringBuilder groupSummary : groupSummaryStringBuilders) {
            groupingMethodSummaryFile.writeLine(groupSummary.toString());
            groupingMethodSummaryFile.writeLine("\n\n");
        }

        groupingMethodSummaryFile.writeString(dsetGroups_buildAccsTableString(groupAccs, cnames, groupNames));
        groupingMethodSummaryFile.writeLine("\n\n");
        groupingMethodSummaryFile.writeString(dsetGroups_buildWinsTableString(groupWins, cnames, groupNames));

        groupingMethodSummaryFile.closeFile();
    }

    public static String dsetGroups_buildWinsTableString(Map<String, double[]> groupWins, String[] cnames, String[] groupNames) {
        int numGroups = groupNames.length;
        StringBuilder sb = new StringBuilder();

        sb.append("This table accounts for ties on a dset e.g if 2 classifiers share best accuracy "
                + "that will count as half a win for each").append("\n");

        //header row
        sb.append("NumWinsInGroups:");
        for (String cname : cnames)
            sb.append(","+cname);
        sb.append(",TotalNumDsetsInGroup").append("\n");

        //calc the avgs too
        double[] groupSums = new double[numGroups], clsfrSums = new double[cnames.length];
        for (int i = 0; i < numGroups; i++) {
            sb.append(groupNames[i]);
            for (int j = 0; j < cnames.length; j++) {
                double val = groupWins.get(cnames[j])[i];
                groupSums[i] += val;
                clsfrSums[j] += val;
                sb.append(","+val);
            }
            sb.append(","+(groupSums[i])).append("\n");
        }

        //print final row, avg of classifiers
        double globalSum = 0;
        sb.append("TotalNumWinsForClassifier");
        for (int j = 0; j < cnames.length; j++) {
            globalSum += clsfrSums[j];
            sb.append(","+clsfrSums[j]);
        }

        sb.append(","+globalSum).append("\n");

        return sb.toString();
    }

    public static String dsetGroups_buildAccsTableString(Map<String, double[]> groupAccs, String[] cnames, String[] groupNames) {
        int numGroups = groupNames.length;
        StringBuilder sb = new StringBuilder();

        //header row
        sb.append("AvgAccsOnGroups:");
        for (String cname : cnames)
            sb.append(","+cname);
        sb.append(",Averages").append("\n");

        //calc the avgs too
        double[] groupAvgs = new double[numGroups], clsfrAvgs = new double[cnames.length];
        for (int i = 0; i < numGroups; i++) {
            sb.append(groupNames[i]);
            for (int j = 0; j < cnames.length; j++) {
                double val = groupAccs.get(cnames[j])[i];
                groupAvgs[i] += val;
                clsfrAvgs[j] += val;
                sb.append(","+val);
            }
            sb.append(","+(groupAvgs[i]/cnames.length)).append("\n");
        }

        //print final row, avg of classifiers
        double globalAvg = 0;
        sb.append("Averages");
        for (int j = 0; j < cnames.length; j++) {
            double avg = clsfrAvgs[j]/numGroups;
            globalAvg += avg;
            sb.append(","+avg);
        }
        globalAvg /= cnames.length;
        sb.append(","+globalAvg).append("\n");

        return sb.toString();
    }

    public static double[][][] dsetGroups_collectDsetVals(double[][][] foldVals, String[] dsets, String[] groupDsets) {
        //cloning arrays to avoid any potential referencing issues considering we're recursing + doing more stuff after all this grouping shite
        double[][][] groupFoldVals = new double[foldVals.length][groupDsets.length][foldVals[0][0].length];

        for (int groupDsetInd = 0; groupDsetInd < groupDsets.length; ++groupDsetInd) {
            String dset = groupDsets[groupDsetInd];
            int globalDsetInd = Arrays.asList(dsets).indexOf(dset);

            for (int classifier = 0; classifier < foldVals.length; classifier++) {
                for (int fold = 0; fold < foldVals[classifier][globalDsetInd].length; fold++) {
                    groupFoldVals[classifier][groupDsetInd][fold] = foldVals[classifier][globalDsetInd][fold];
                }
            }
        }

        return groupFoldVals;
    }


    protected static String[] eval_metric(String outPath, String filename, ClassifierResultsCollection results, PerformanceMetric metric, Map<String, Map<String, String[]>> dsetGroupings) throws Exception {
        String statName = metric.name;
        outPath += statName + "/";
        new File(outPath).mkdirs();

        String[] cnames = results.getClassifierNamesInOutput();
        String[] dsets = results.getDatasetNamesInOutput();

        double[][][] testFolds = results.sliceSplit("test").retrieveDoubles(metric.getter)[0];

        if (!testResultsOnly) {
            double[][][] trainFolds = results.sliceSplit("train").retrieveDoubles(metric.getter)[0];
            double[][][] trainTestDiffsFolds = findTrainTestDiffs(trainFolds, testFolds);

            eval_metricOnSplit(outPath, filename, null, trainLabel, metric, trainFolds, cnames, dsets, dsetGroupings);
            eval_metricOnSplit(outPath, filename, null, trainTestDiffLabel, metric, trainTestDiffsFolds, cnames, dsets, dsetGroupings);
        }

        return eval_metricOnSplit(outPath, filename, null, testLabel, metric, testFolds, cnames, dsets, dsetGroupings);
    }

    protected static String[/*{train,test}*/][] eval_CompResourcesRAW(String outPath, String filename, ClassifierResultsCollection results, Map<String, Map<String, String[]>> dsetGroupings) throws Exception {
        String[] cnames = results.getClassifierNamesInOutput();
        String[] dsets = results.getDatasetNamesInOutput();

        computationalDiaFolderName = computationalDiaFolderName_raw;

        PerformanceMetric trainTimeMetric = ClassifierResultsAnalysis.trainTimeMetric;

        String timingsOutPath = outPath + "TimingsRAW/"; //special case for timings
        new File(timingsOutPath).mkdirs();

        // NOTE: getting train timings from test files intentionally ( train.. = ..sliceSplit("test")..), avoids check for whether we're actually loading in
        // train files in comparison set up. build times should be same in both trainFoldX and testFoldX file anyway

        double[][][] trainTimes = results.sliceSplit("test").retrieveDoubles(trainTimeMetric.getter)[0];
        String[] trainResStr = null;
        if (trainTimes != null)
            trainResStr = eval_metricOnSplit(timingsOutPath, filename, null, trainLabel, trainTimeMetric, trainTimes, cnames, dsets, dsetGroupings);

        double[][][] testTimes = results.sliceSplit("test").retrieveDoubles(testTimeMetric.getter)[0];
        String[] testResStr = null;
        if (testTimes != null)
            testResStr = eval_metricOnSplit(timingsOutPath, filename, null, testLabel, testTimeMetric, testTimes, cnames, dsets, dsetGroupings);

//        double[][][] estimateTimes = results.sliceSplit("test").retrieveDoubles(estimateTimeMetric.getter)[0];
//        String[] estimateResStr = null;
//        if (estimateTimes != null)
//            estimateResStr = eval_metricOnSplit(outPath, filename, null, estimateLabel, estimateTimeMetric, estimateTimes, cnames, dsets, dsetGroupings);


        String memoryOutPath = outPath + "MaxMemory/"; //special case for timings
        new File(memoryOutPath).mkdirs();

        // NOTE: same as before, just getting the memory from the test files

        double[][][] memoryMax = results.sliceSplit("test").retrieveDoubles(memoryMaxMetric.getter)[0];
        String[] memoryResStr = null;
        if (memoryMax != null)
            memoryResStr = eval_metricOnSplit(memoryOutPath, filename, null, testLabel, memoryMaxMetric, memoryMax, cnames, dsets, dsetGroupings);



        return new String[][] { trainResStr, testResStr, memoryResStr };
//        return new String[][] { trainResStr, testResStr, estimateResStr };
    }

    protected static String[/*{train,test}*/][] eval_CompResourcesBENCHMARKED(String outPath, String filename, ClassifierResultsCollection results, Map<String, Map<String, String[]>> dsetGroupings) throws Exception {
        String[] cnames = results.getClassifierNamesInOutput();
        String[] dsets = results.getDatasetNamesInOutput();

        computationalDiaFolderName = computationalDiaFolderName_benchmark;

        PerformanceMetric trainTimeMetric = benchmarkedTrainTimeMetric;

        outPath += "TimingsBENCHMARKED/"; //special case for timings
        new File(outPath).mkdirs();

        // NOTE: getting train timings from test files intentionally ( train.. = ..sliceSplit("test")..), avoids check for whether we're actually loading in
        // train files in comparison set up. build times should be same in both trainFoldX and testFoldX file anyway

        double[][][] benchmarkedTrainTimes = results.sliceSplit("test").retrieveDoubles(trainTimeMetric.getter)[0];

        String[] trainResStr = null;
        if (benchmarkedTrainTimes != null) {
            trainResStr = eval_metricOnSplit(outPath, filename, null, trainLabel, trainTimeMetric, benchmarkedTrainTimes, cnames, dsets, dsetGroupings);
            writeTableFile_ClassifierDatasetFolds(outPath + "allTrainBenchmarkTimes.csv", "TrainBenchmarkTimes", benchmarkedTrainTimes, cnames, dsets);
        }

        double[][][] benchmarkedTestTimes = results.sliceSplit("test").retrieveDoubles(benchmarkedTestTimeMetric.getter)[0];

        String[] testResStr = null;
        if (benchmarkedTestTimes != null) {
            testResStr = eval_metricOnSplit(outPath, filename, null, testLabel, benchmarkedTestTimeMetric, benchmarkedTestTimes, cnames, dsets, dsetGroupings);
            writeTableFile_ClassifierDatasetFolds(outPath + "allTestBenchmarkTimes.csv", "TestBenchmarkTimes", benchmarkedTestTimes, cnames, dsets);
        }

//        double[][][] estimateTimes = results.sliceSplit("test").retrieveDoubles(estimateTimeMetric.getter)[0];
//        String[] estimateResStr = null;
//        if (estimateTimes != null)
//            estimateResStr = eval_metricOnSplit(outPath, filename, null, estimateLabel, estimateTimeMetric, estimateTimes, cnames, dsets, dsetGroupings);

        return new String[][] { trainResStr, testResStr };
//        return new String[][] { trainResStr, testResStr, estimateResStr };
    }

    protected static void writeCliqueHelperFiles(String cdCSVpath, String expname, PerformanceMetric metric, String cliques) {
        (new File(cdCSVpath)).mkdirs();

        //temp workaround, just write the cliques and readin again from matlab for ease of checking/editing for pairwise edge cases
        OutFile out = new OutFile (cdCSVpath + fileNameBuild_cd(expname, metric.name) + "_cliques.txt");
        out.writeString(cliques);
        out.closeFile();
    }

    /**
     * this will build all the diagrams it can from the average results files that
     * exist in the cddia directory, instead of being given a list of stats that it should expect
     * to find there, carry over from when I made the diagrams manually. todo maybe now force it to take 
     * a list of stats to expect as a form of error checking
     */
    protected static void matlab_buildCDDias(String expname, String[] cliques) {
        MatlabController proxy = MatlabController.getInstance();
        proxy.eval("buildDiasInDirectory('"+expRootDirectory+cdDiaFolderName+"/"+friedmanCDDiaDirName+"', 0, "+FRIEDMANCDDIA_PVAL+");"); //friedman
        proxy.eval("clear");
        proxy.eval("buildDiasInDirectory('"+expRootDirectory+cdDiaFolderName+"/"+pairwiseCDDiaDirName+"', 1);");  //pairwise
        proxy.eval("clear");
    }


    protected static void matlab_buildCompResourcesDias(List<PerformanceMetric> metrics) {
        MatlabController proxy = MatlabController.getInstance();

        for (PerformanceMetric metric : metrics) {

            String diaFolder = expRootDirectory + "/" + (metric.name.toLowerCase().contains("benchmark") ? computationalDiaFolderName_benchmark : computationalDiaFolderName_raw) + "/";

            String evalSet = metric.equals(PerformanceMetric.totalTestTime) || metric.equals(testTimeMetric) ||
                            metric.equals(PerformanceMetric.totalTestTimeBenchmarked) || metric.equals(benchmarkedTestTimeMetric)
                            || metric.equals(memoryMaxMetric)
                    ? testLabel
                    : trainLabel;
            String filenameNoExtension = fileNameBuild_avgsFile(evalSet, metric).replace(".csv", "");

            String ylabel = metric.equals(memoryMaxMetric) ?
                    "Max Memory (MB)" :
                    "Time, " + evalSet.toLowerCase() + " (ms)";

            proxy.eval("compResourcesLinePlot('" + diaFolder + filenameNoExtension + "', '" + evalSet.toLowerCase() + "','" + ylabel + "');");
        }
    }

    protected static void eval_perFoldFiles(String outPath, double[][][] folds, String[] cnames, String[] dsets, String splitLabel) {
        new File(outPath).mkdirs();

        StringBuilder headers = new StringBuilder("folds:");
        for (int f = 0; f < folds[0][0].length; f++)
            headers.append(","+f);

        for (int c = 0; c < folds.length; c++) {
            OutFile out=new OutFile(outPath + cnames[c]+"_"+splitLabel+"FOLDS.csv");
            out.writeLine(headers.toString());

            for (int d = 0; d < folds[c].length; d++) {
                out.writeString(dsets[d]);
                for (int f = 0; f < folds[c][d].length; f++)
                    out.writeString("," + folds[c][d][f]);
                out.writeLine("");
            }

            out.closeFile();
        }

        writeRawTableFile_ClassifierDatasetFolds(outPath + "TEXASPLOT_"+splitLabel+".csv", folds, cnames);
    }



    protected static String fileHelper_tabulate(double[][] res, String[] cnames, String[] dsets) {
        StringBuilder sb = new StringBuilder();
        sb.append(fileHelper_header(cnames));

        for (int dset = 0; dset < res[0].length; ++dset) {
            sb.append("\n").append(dsets[dset]);

            for (int classifier = 0; classifier < res.length; classifier++)
                sb.append("," + res[classifier][dset]);
        }
        return sb.toString();
    }

    protected static String fileHelper_tabulateRaw(double[][] res, String[] cnames) {
        StringBuilder sb = new StringBuilder();
        sb.append(fileHelper_header(cnames).substring(1));

        for (int dset = 0; dset < res[0].length; ++dset) {
            sb.append("\n").append(res[0][dset]);
            for (int classifier = 1; classifier < res.length; classifier++)
                sb.append("," + res[classifier][dset]);
        }
        return sb.toString();
    }

    protected static String fileHelper_tabulate(double[][][] res, String[] cnames, String[] dsets) {
        StringBuilder sb = new StringBuilder();
        sb.append(fileHelper_header(cnames));

        for (int dset = 0; dset < res[0].length; ++dset) {
            for (int fold = 0; fold < res[0][0].length; fold++) {
                sb.append("\n").append(dsets[dset]).append("_").append(fold);
                for (int classifier = 0; classifier < res.length; classifier++)
                    sb.append("," + res[classifier][dset][fold]);
            }
        }
        return sb.toString();
    }

    protected static String fileHelper_tabulateRaw(double[][][] res, String[] cnames) {
        StringBuilder sb = new StringBuilder();
        sb.append(fileHelper_header(cnames).substring(1));

        for (int dset = 0; dset < res[0].length; ++dset) {
            for (int fold = 0; fold < res[0][0].length; fold++) {
                sb.append("\n").append(res[0][dset][fold]);
                for (int classifier = 1; classifier < res.length; classifier++)
                    sb.append("," + res[classifier][dset][fold]);
            }
        }
        return sb.toString();
    }

    protected static String fileHelper_header(String[] names) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < names.length; i++)
            sb.append(",").append(names[i]);
        return sb.toString();
    }

    protected static String util_mean(double[][] res) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < res.length; i++)
            sb.append(",").append(StatisticalUtilities.mean(res[i], false));

        return sb.toString();
    }

    protected static String util_stddev(double[][] res) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < res.length; i++)
            sb.append(",").append(StatisticalUtilities.standardDeviation(res[i], false, StatisticalUtilities.mean(res[i], false)));

        return sb.toString();
    }

//    protected static double[][][] util_correctTimingsForBenchmarks(double[][][] timings, double[][][] benchmarks) {
//        for (int classifier = 0; classifier < timings.length; classifier++)
//            for (int dset = 0; dset < timings[0].length; dset++)
//                for (int fold = 0; fold < timings[0][0].length; fold++)
//                    timings[classifier][dset][fold] /= benchmarks[classifier][dset][fold];
//
//        return timings;
//    }

    protected static double[][][] findTrainTestDiffs(double[][][] trainFoldAccs, double[][][] testFoldAccs) {
        double[][][] diffs = new double[trainFoldAccs.length][trainFoldAccs[0].length][trainFoldAccs[0][0].length];

        for (int c = 0; c < diffs.length; c++)
            for (int d = 0; d < diffs[c].length; d++)
                for (int f = 0; f < diffs[c][d].length; f++)
                    diffs[c][d][f] =  trainFoldAccs[c][d][f] - testFoldAccs[c][d][f];

        return diffs;
    }

    /**
     * todo maybe enum for mode etc
     *
     * @param takeMean if true, will average by taking mean, else will take median
     */
    protected static double[][] findAvgsOverFolds(double[][][] foldaccs, boolean takeMean) {
        double[][] accs = new double[foldaccs.length][foldaccs[0].length];
        for (int i = 0; i < accs.length; i++)
            for (int j = 0; j < accs[i].length; j++)
                if (takeMean)
                    accs[i][j] = StatisticalUtilities.mean(foldaccs[i][j], false);
                else
                    accs[i][j] = StatisticalUtilities.median(foldaccs[i][j]);

        return accs;
    }

    protected static double[][] findStddevsOverFolds(double[][][] foldaccs) {
        double[][] devs = new double[foldaccs.length][foldaccs[0].length];
        for (int i = 0; i < devs.length; i++)
            for (int j = 0; j < devs[i].length; j++)
                devs[i][j] = StatisticalUtilities.standardDeviation(foldaccs[i][j], false, StatisticalUtilities.mean(foldaccs[i][j], false));

        return devs;
    }

    protected static int[] findOrdering(double[][] r) {
        double[] avgranks = new double[r.length];
        for (int i = 0; i < r.length; i++)
            avgranks[i] = StatisticalUtilities.mean(r[i], false);

        int[] res = new int[avgranks.length];

        int i = 0;
        while (i < res.length) {
            ArrayList<Integer> mins = util_min(avgranks);

            for (int j = 0; j < mins.size(); j++) {
                res[mins.get(j)] = i++;
                avgranks[mins.get(j)] = Double.MAX_VALUE;
            }
        }

        return res;
    }

    protected static int[] findReverseOrdering(double[][] r) {
        double[] avgranks = new double[r.length];
        for (int i = 0; i < r.length; i++)
            avgranks[i] = StatisticalUtilities.mean(r[i], false);

        int[] res = new int[avgranks.length];

        int i = 0;
        while (i < res.length) {
            ArrayList<Integer> maxs = util_max(avgranks);

            for (int j = 0; j < maxs.size(); j++) {
                res[maxs.get(j)] = i++;
                avgranks[maxs.get(j)] = -Double.MAX_VALUE;
            }
        }

        return res;
    }

    protected static ArrayList<Integer> util_min(double[] d) {
        double min = d.length+1;
        ArrayList<Integer> minIndices = null;

        for (int c = 0; c < d.length; c++) {
            if(d[c] < min){
                min = d[c];
                minIndices = new ArrayList<>();
                minIndices.add(c);
            }else if(d[c] == min){
                minIndices.add(c);
            }
        }

        return minIndices;
    }

    protected static ArrayList<Integer> util_max(double[] d) {
        double max = -1;
        ArrayList<Integer> maxIndices = null;

        for (int c = 0; c < d.length; c++) {
            if(d[c] > max){
                max = d[c];
                maxIndices = new ArrayList<>();
                maxIndices.add(c);
            }else if(d[c] == max){
                maxIndices.add(c);
            }
        }

        return maxIndices;
    }

    protected static String[] util_order(String[] s, int[] ordering) {
        String[] res = new String[s.length];

        for (int i = 0; i < ordering.length; i++)
            res[ordering[i]] = s[i];

        return res;
    }

    protected static double[][] util_order(double[][] s, int[] ordering) {
        double[][] res = new double[s.length][];

        for (int i = 0; i < ordering.length; i++)
            res[ordering[i]] = s[i];

        return res;
    }

    protected static double[][][] util_order(double[][][] s, int[] ordering) {
        double[][][] res = new double[s.length][][];

        for (int i = 0; i < ordering.length; i++)
            res[ordering[i]] = s[i];

        return res;
    }

    /**
     * @param accs [classifiers][acc on datasets]
     * @param higherIsBetter if true, larger values will receive a better (i.e. lower) rank, false vice versa. e.g want to maximise acc, but want to minimise time
     * @return [classifiers][rank on dataset]
     */
    protected static double[][] findRanks(double[][] accs, boolean higherIsBetter) {
        double[][] ranks = new double[accs.length][accs[0].length];

        for (int d = 0; d < accs[0].length; d++) {
            Double[] orderedAccs = new Double[accs.length];
            for (int c = 0; c < accs.length; c++)
                orderedAccs[c] = accs[c][d];

            if (higherIsBetter)
                Arrays.sort(orderedAccs, Collections.reverseOrder());
            else
                Arrays.sort(orderedAccs);

//            //README - REDACTED, this problem is currently just being ignored, since it makes so many headaches and is so insignificant anyway
//            //to create parity between this and the matlab critical difference diagram code,
//            //rounding the *accuracies used to calculate ranks* to 15 digits (after the decimal) 
//            //this affects the average rank summary statistic, but not e.g the average accuracy statistic
//            //matlab has a max default precision of 16. in a tiny number of cases, there are differences 
//            //in accuracy that are smaller than this maximum precision, which were being taken into
//            //acount here (by declaring one as havign a higher rank than the other), but not being 
//            //taken into account in matlab (which considered them a tie). 
//            //one could argue the importance of a difference less than 1x10^-15 when comparing classifiers,
//            //so for ranks only, will round to matlab's precision. rounding the accuracies everywhere
//            //creates a number of headaches, therefore the tiny inconsistency as a result of this
//            //will jsut have to be lived with
//            final int DEFAULT_MATLAB_PRECISION = 15;
//            for (int c = 0; c < accs.length; c++) {
//                MathContext mc = new MathContext(DEFAULT_MATLAB_PRECISION, RoundingMode.DOWN);
//                BigDecimal bd = new BigDecimal(orderedAccs[c],mc);
//                orderedAccs[c] = bd.doubleValue();
//            }


            for (int rank = 0; rank < accs.length; rank++) {
                for (int c = 0; c < accs.length; c++) {
//                    if (orderedAccs[rank] == new BigDecimal(accs[c][d], new MathContext(DEFAULT_MATLAB_PRECISION, RoundingMode.DOWN)).doubleValue()) {
                    if (orderedAccs[rank] == accs[c][d]) {
                        ranks[c][d] = rank; //count from one
                    }
                }
            }

            //correcting ties
            int[] hist = new int[accs.length];
            for (int c = 0; c < accs.length; c++)
                ++hist[(int)ranks[c][d]];

            for (int r = 0; r < hist.length; r++) {
                if (hist[r] > 1) {//ties
                    double newRank = 0;
                    for (int i = 0; i < hist[r]; i++)
                        newRank += r-i;
                    newRank/=hist[r];
                    for (int c = 0; c < ranks.length; c++)
                        if (ranks[c][d] == r)
                            ranks[c][d] = newRank;
                }
            }

            //correcting for index from 1
            for (int c = 0; c < accs.length; c++)
                ++ranks[c][d];
        }

        return ranks;
    }

    protected static String[] eval_winsDrawsLosses(double[][] accs, String[] cnames, String[] dsets) {
        StringBuilder table = new StringBuilder();
        ArrayList<ArrayList<ArrayList<String>>> wdlList = new ArrayList<>(); //[classifierPairing][win/draw/loss][dsetNames]
        ArrayList<String> wdlListNames = new ArrayList<>();

        String[][] wdlPlusMinus = new String[cnames.length*cnames.length][dsets.length];

        table.append("flat" + fileHelper_header(cnames)).append("\n");

        int count = 0;
        for (int c1 = 0; c1 < accs.length; c1++) {
            table.append(cnames[c1]);
            for (int c2 = 0; c2 < accs.length; c2++) {
                wdlListNames.add(cnames[c1] + "_VS_" + cnames[c2]);
                wdlList.add(new ArrayList<>());
                wdlList.get(count).add(new ArrayList<>());
                wdlList.get(count).add(new ArrayList<>());
                wdlList.get(count).add(new ArrayList<>());

                int wins=0, draws=0, losses=0;
                for (int d = 0; d < dsets.length; d++) {
                    if (accs[c1][d] > accs[c2][d]) {
                        wins++;
                        wdlList.get(count).get(0).add(dsets[d]);
                        wdlPlusMinus[count][d] = "1";
                    }
                    else if ((accs[c1][d] == accs[c2][d])) {
                        draws++;
                        wdlList.get(count).get(1).add(dsets[d]);
                        wdlPlusMinus[count][d] = "0";
                    }
                    else {
                        losses++;
                        wdlList.get(count).get(2).add(dsets[d]);
                        wdlPlusMinus[count][d] = "-1";
                    }
                }
                table.append(","+wins+"|"+draws+"|"+losses);
                count++;
            }
            table.append("\n");
        }

        StringBuilder list = new StringBuilder();
        for (int i = 0; i < wdlListNames.size(); ++i) {
            list.append(wdlListNames.get(i));
            list.append("\n");
            list.append("Wins("+wdlList.get(i).get(0).size()+"):");
            for (String dset : wdlList.get(i).get(0))
                list.append(",").append(dset);
            list.append("\n");
            list.append("Draws("+wdlList.get(i).get(1).size()+"):");
            for (String dset : wdlList.get(i).get(1))
                list.append(",").append(dset);
            list.append("\n");
            list.append("Losses("+wdlList.get(i).get(2).size()+"):");
            for (String dset : wdlList.get(i).get(2))
                list.append(",").append(dset);
            list.append("\n\n");
        }

        StringBuilder plusMinuses = new StringBuilder();
        for (int j = 0; j < wdlPlusMinus.length; j++)
            plusMinuses.append(",").append(wdlListNames.get(j));

        for (int i = 0; i < dsets.length; i++) {
            plusMinuses.append("\n").append(dsets[i]);
            for (int j = 0; j < wdlPlusMinus.length; j++)
                plusMinuses.append(",").append(wdlPlusMinus[j][i]);
        }

        return new String[] { table.toString(), list.toString(), plusMinuses.toString() };
    }

    protected static String[] eval_sigWinsDrawsLosses(double pval, double[][] accs, double[][][] foldAccs, String[] cnames, String[] dsets) {
        StringBuilder table = new StringBuilder();
        ArrayList<ArrayList<ArrayList<String>>> wdlList = new ArrayList<>(); //[classifierPairing][win/draw/loss][dsetNames]
        ArrayList<String> wdlListNames = new ArrayList<>();

        String[][] wdlPlusMinus = new String[cnames.length*cnames.length][dsets.length];

        table.append("p=" + pval + fileHelper_header(cnames)).append("\n");

        int count = 0;
        for (int c1 = 0; c1 < foldAccs.length; c1++) {
            table.append(cnames[c1]);
            for (int c2 = 0; c2 < foldAccs.length; c2++) {
                wdlListNames.add(cnames[c1] + "_VS_" + cnames[c2]);
                wdlList.add(new ArrayList<>());
                wdlList.get(count).add(new ArrayList<>());
                wdlList.get(count).add(new ArrayList<>());
                wdlList.get(count).add(new ArrayList<>());

                int wins=0, draws=0, losses=0;
                for (int d = 0; d < dsets.length; d++) {
                    if (accs[c1][d] == accs[c2][d]) {
                        //when the accuracies are identical, p == NaN. 
                        //because NaN < 0.05 apparently it wont be counted as a draw, but a loss
                        //so handle it here                        
                        draws++;
                        wdlList.get(count).get(1).add(dsets[d]);
                        wdlPlusMinus[count][d] = "0";
                        continue;
                    }

                    double p = TwoSampleTests.studentT_PValue(foldAccs[c1][d], foldAccs[c2][d]);

                    if (p > pval) {
                        draws++;
                        wdlList.get(count).get(1).add(dsets[d]);
                        wdlPlusMinus[count][d] = "0";
                    }
                    else { //is sig
                        if (accs[c1][d] > accs[c2][d]) {
                            wins++;
                            wdlList.get(count).get(0).add(dsets[d]);
                            wdlPlusMinus[count][d] = "1";
                        }
                        else  {
                            losses++;
                            wdlList.get(count).get(2).add(dsets[d]);
                            wdlPlusMinus[count][d] = "-1";
                        }
                    }
                }
                table.append(","+wins+"|"+draws+"|"+losses);
                count++;
            }
            table.append("\n");
        }

        StringBuilder list = new StringBuilder();
        for (int i = 0; i < wdlListNames.size(); ++i) {
            list.append(wdlListNames.get(i));
            list.append("\n");
            list.append("Wins("+wdlList.get(i).get(0).size()+"):");
            for (String dset : wdlList.get(i).get(0))
                list.append(",").append(dset);
            list.append("\n");
            list.append("Draws("+wdlList.get(i).get(1).size()+"):");
            for (String dset : wdlList.get(i).get(1))
                list.append(",").append(dset);
            list.append("\n");
            list.append("Losses("+wdlList.get(i).get(2).size()+"):");
            for (String dset : wdlList.get(i).get(2))
                list.append(",").append(dset);
            list.append("\n\n");
        }

        StringBuilder plusMinuses = new StringBuilder();
        for (int j = 0; j < wdlPlusMinus.length; j++)
            plusMinuses.append(",").append(wdlListNames.get(j));

        for (int i = 0; i < dsets.length; i++) {
            plusMinuses.append("\n").append(dsets[i]);
            for (int j = 0; j < wdlPlusMinus.length; j++)
                plusMinuses.append(",").append(wdlPlusMinus[j][i]);
        }

        return new String[] { table.toString(), list.toString(), plusMinuses.toString() };
    }


    /**
     * Intended for potentially new stats that are introduced over time (at time of writing this function,
     * build and especially test times), where maybe some older files in the intended analysis 
     * dont have the stat but newer ones do, or some classifiers that write their own files 
     * (via e.g TrainAccuracyEstimate) aren't properly writing them.
     *
     * Missing for timings is defined as -1. why cant i hold all this spaghetti?
     *
     * Looking ONLY at the test files, a) because they should all be here anyway else 
     * wouldnt have got as far as needing to call this, b) because the 'testtime' stored 
     * in the testfold files are the test timing we're generally actually interested in,
     * i.e. the total prediction time of the fully trained classifier on the test set, 
     * as opposed to the test time of the classifier on (e.g) corssvalidation folds in training
     * that is stored in the train file
     *
     * @returns null if any of the wanted info is missing, else the score described by the stat for each results
     */
    private static double[][][] getTimingsIfAllArePresent(List<ClassifierEvaluation> res, Function<ClassifierResults, Double> getter) {
        double[][][] info = new double[res.size()][res.get(0).testResults.length][res.get(0).testResults[0].length];

        for (int i = 0; i < res.size(); i++) {
            for (int j = 0; j < res.get(i).testResults.length; j++) {
                for (int k = 0; k < res.get(i).testResults[j].length; k++) {
                    info[i][j][k] = getter.apply(res.get(i).testResults[j][k]);

                    if (info[i][j][k] == -1)
                        return null;
                }
            }
        }

        return info;
    }

    protected static double[][][] getInfo(List<ClassifierEvaluation> res, Function<ClassifierResults, Double> getter, String trainortest) {
        double[][][] info = new double[res.size()][res.get(0).testResults.length][res.get(0).testResults[0].length];
        for (int i = 0; i < res.size(); i++) {
            if (trainortest.equalsIgnoreCase(trainLabel))
                for (int j = 0; j < res.get(i).trainResults.length; j++)
                    for (int k = 0; k < res.get(i).trainResults[j].length; k++)
                        info[i][j][k] = getter.apply(res.get(i).trainResults[j][k]);
            else if (trainortest.equalsIgnoreCase(testLabel))
                for (int j = 0; j < res.get(i).testResults.length; j++)
                    for (int k = 0; k < res.get(i).testResults[j].length; k++)
                        info[i][j][k] = getter.apply(res.get(i).testResults[j][k]);
            else {
                System.out.println("getInfo(), trainortest="+trainortest);
                System.exit(0);
            }
        }
        return info;
    }

    protected static String[] getNames(List<ClassifierEvaluation> res) {
        String[] names = new String[res.size()];
        for (int i = 0; i < res.size(); i++)
            names[i] = res.get(i).classifierName;
        return names;
    }

    protected static void jxl_buildResultsSpreadsheet(String basePath, String expName, List<PerformanceMetric> metrics) {
        WritableWorkbook wb = null;
        WorkbookSettings wbs = new WorkbookSettings();
        wbs.setLocale(new Locale("en", "EN"));

        try {
            wb = Workbook.createWorkbook(new File(basePath + expName + "ResultsSheet.xls"), wbs);
        } catch (Exception e) {
            System.out.println("ERROR CREATING RESULTS SPREADSHEET");
            System.out.println(e);
            System.exit(0);
        }

        WritableSheet summarySheet = wb.createSheet("GlobalSummary", 0);
        String summaryCSV = basePath + expName + "_SMALLglobalSummary.csv";
        jxl_copyCSVIntoSheet(summarySheet, summaryCSV);

        for (int i = 0; i < metrics.size(); i++) {
            if (metrics.get(i).equals(PerformanceMetric.buildTime))
                jxl_buildStatSheets_timings(wb, basePath, metrics.get(i), i, trainLabel, "RAW");
            else if (metrics.get(i).equals(testTimeMetric))
                jxl_buildStatSheets_timings(wb, basePath, metrics.get(i), i, testLabel, "RAW");
            else if (metrics.get(i).equals(PerformanceMetric.buildTimeBenchmarked))
                jxl_buildStatSheets_timings(wb, basePath, metrics.get(i), i, trainLabel, "BENCHMARKED");
            else if (metrics.get(i).equals(benchmarkedTestTimeMetric))
                jxl_buildStatSheets_timings(wb, basePath, metrics.get(i), i, testLabel, "BENCHMARKED");
            else
                jxl_buildStatSheets(wb, basePath, metrics.get(i), i);
        }

        try {
            wb.write();
            wb.close();
        } catch (Exception e) {
            System.out.println("ERROR WRITING AND CLOSING RESULTS SPREADSHEET");
            System.out.println(e);
            System.exit(0);
        }
    }

    protected static void jxl_buildStatSheets(WritableWorkbook wb, String basePath, PerformanceMetric metric, int statIndex) {
        String metricPath = basePath + metric + "/";
        String testMetricPath = metricPath + testLabel + "/";

        WritableSheet testSheet = wb.createSheet(metric+"Test", wb.getNumberOfSheets());
        String testCSV = testMetricPath+ fileNameBuild_avgsFile(testLabel, metric);
        jxl_copyCSVIntoSheet(testSheet, testCSV);

        WritableSheet summarySheet = wb.createSheet(metric+"TestSigDiffs", wb.getNumberOfSheets());
        String summaryCSV = testMetricPath + fileNameBuild_summaryFile(testLabel, metric);
        jxl_copyCSVIntoSheet(summarySheet, summaryCSV);
    }

    protected static void jxl_buildStatSheets_timings(WritableWorkbook wb, String basePath, PerformanceMetric metric, int statIndex, String evalSet, String timingType) {
        // ************* the difference: timings folder assumed instead of going by the specific metric name
        //i.e Timings/TRAIN/TrainTimings and Timings/TEST/TestTimings    
        //instead of TrainTimings/TRAIN/TrainTimings ... 
        String metricPath = basePath + "Timings"+timingType+"/" + evalSet + "/";

        WritableSheet avgsSheet = wb.createSheet(metric.name, wb.getNumberOfSheets());
        String testCSV = metricPath + fileNameBuild_avgsFile(evalSet, metric);
        jxl_copyCSVIntoSheet(avgsSheet, testCSV);

        WritableSheet summarySheet = wb.createSheet(metric.name+"SigDiffs", wb.getNumberOfSheets());
        String summaryCSV = metricPath + fileNameBuild_summaryFile(evalSet, metric);
        jxl_copyCSVIntoSheet(summarySheet, summaryCSV);

    }

    protected static void jxl_copyCSVIntoSheet(WritableSheet sheet, String csvFile) {
        try {
            Scanner fileIn = new Scanner(new File(csvFile));

            int rowInd = 0;
            while (fileIn.hasNextLine()) {
                Scanner lineIn = new Scanner(fileIn.nextLine());
                lineIn.useDelimiter(",");

                int colInd = -1;
                while (lineIn.hasNext()) {
                    colInd++; //may not reach end of block, so incing first and initialising at -1

                    String cellContents = lineIn.next();
                    WritableFont font = new WritableFont(WritableFont.ARIAL, 10);
                    WritableCellFormat format = new WritableCellFormat(font);

                    try {
                        int iCellContents = Integer.parseInt(cellContents);
                        sheet.addCell(new jxl.write.Number(colInd, rowInd, iCellContents, format));
                        continue; //if successful, val was int, has been written, move on
                    } catch (NumberFormatException nfm) { }

                    try {
                        double dCellContents = Double.parseDouble(cellContents);
                        sheet.addCell(new jxl.write.Number(colInd, rowInd, dCellContents, format));
                        continue; //if successful, val was int, has been written, move on
                    } catch (NumberFormatException nfm) { }


                    sheet.addCell(new jxl.write.Label(colInd, rowInd, cellContents, format));
                }
                rowInd++;
            }
        } catch (Exception e) {
            System.out.println("ERROR BUILDING RESULTS SPREADSHEET, COPYING CSV");
            System.out.println(e);
            System.exit(0);
        }
    }

    public static Pair<String[], double[][]> matlab_readRawFile(String file, int numDsets) throws FileNotFoundException {
        ArrayList<String> cnames = new ArrayList<>();

        Scanner in = new Scanner(new File(file));

        Scanner linein = new Scanner(in.nextLine());
        linein.useDelimiter(",");

        while (linein.hasNext())
            cnames.add(linein.next());

        double[][] vals = new double[cnames.size()][numDsets];

        for (int d = 0; d < numDsets; d++) {
            linein = new Scanner(in.nextLine());
            linein.useDelimiter(",");
            for (int c = 0; c < cnames.size(); c++)
                vals[c][d] = linein.nextDouble();
        }
        return new Pair<>(cnames.toArray(new String[] { }), vals);
    }

    public static void matlab_buildPairwiseScatterDiagrams(String outPath, String expName, List<PerformanceMetric> metrics, String[] dsets) {
        outPath += pairwiseScatterDiaPath;

        for (PerformanceMetric metric : metrics) {
            try {
                boolean compStat = allComputationalMetrics.contains(metric);
                boolean originIsZero = !compStat; // if not a computational stat, probably in the range 0..1, keep that instead of min..max
                boolean drawFitLine = compStat;

                Pair<String[], double[][]> asd = matlab_readRawFile(outPath + fileNameBuild_pws(expName, metric.name) + ".csv", dsets.length);
                String[] classifierNames = asd.var1;
                double[][] allResults = asd.var2;

                int numClassifiers = allResults.length;

                MatlabController proxy = MatlabController.getInstance();

                for (int c1 = 0; c1 < numClassifiers-1; c1++) {
                    for (int c2 = c1+1; c2 < numClassifiers; c2++) {

                        String c1name = classifierNames[c1];
                        String c2name = classifierNames[c2];

                        double[] c1res = allResults[c1];
                        double[] c2res = allResults[c2];

                        if (c1name.compareTo(c2name) > 0) {
                            String t = c1name;
                            c1name = c2name;
                            c2name = t;

                            double[] t2 = c1res;
                            c1res = c2res;
                            c2res = t2;
                        }

                        String pwFolderName = outPath + c1name + "vs" + c2name + "/";
                        (new File(pwFolderName)).mkdir();

                        StringBuilder sb = new StringBuilder("array = [");
                        for (int i = 0; i < dsets.length; i++) {
                            sb.append(c1res[i] + "," + c2res[i] + ";");
                        }
                        proxy.eval(sb.toString() + "];");

                        final StringBuilder concat = new StringBuilder();
                        concat.append("'");
                        concat.append(c1name.replaceAll("_", "\\\\_"));
                        concat.append("',");
                        concat.append("'");
                        concat.append(c2name.replaceAll("_", "\\\\_"));
                        concat.append("'");
                        proxy.eval("labels = {" + concat.toString() + "};");
//
                        proxy.eval("'" + fileNameBuild_pwsInd(c1name, c2name, metric.name) + "'"); //just print the filename in the matlab window, for log of progress. no longer printing fig details
                        proxy.eval("pairedscatter('" + pwFolderName + fileNameBuild_pwsInd(c1name, c2name, metric.name).replaceAll("\\.", "") + "',array(:,1),array(:,2),labels,'"+metric.name+"','"+metric.comparisonDescriptor+"',"+drawFitLine+","+originIsZero+");");
                        proxy.eval("clear");
                    }
                }
            } catch (Exception io) {
                System.out.println("buildPairwiseScatterDiagrams("+outPath+") failed loading " + metric.name + " file\n" + io);
            }
        }
    }


    public static int[] dsetGroups_clusterDsetResults(double[/*classifier*/][/*dataset*/] results) {
        double[/*dataset*/][/*classifier*/] dsetScores = GenericTools.cloneAndTranspose(results);
        int numDsets = dsetScores.length;

        for (int dset = 0; dset < dsetScores.length; dset++) {
            double dsetAvg = StatisticalUtilities.mean(dsetScores[dset], false);
            for (int clsfr = 0; clsfr < dsetScores[dset].length; clsfr++)
                dsetScores[dset][clsfr] -= dsetAvg;
        }

        Instances clusterData = InstanceTools.toWekaInstances(dsetScores);

        XMeans xmeans = new XMeans();
        xmeans.setMaxNumClusters(Math.min((int)Math.sqrt(numDsets), 5));
        xmeans.setSeed(0);

        try {
            xmeans.buildClusterer(new Instances(clusterData));
            //pass copy, just in case xmeans does any kind of reordering of 
            //instances. we want to maintain order of dsets/instances for indexing purposes
        } catch (Exception e) {
            System.out.println("Problem building clusterer for post hoc dataset groupings\n" + e);
        }

        int numClusters = xmeans.numberOfClusters();

        int[] assignments = new int[numDsets+1];
        assignments[numDsets] = numClusters;

        for (int i = 0; i < numDsets; i++) {
            try {
                assignments[i] = xmeans.clusterInstance(clusterData.instance(i));
            } catch (Exception e) {
                System.out.println("Problem assigning clusters in post hoc dataset groupings, dataset " + i + "\n" + e);
            }
        }

        return assignments;
    }

    public static void main(String[] args) throws Exception {
        String[] settings=new String[6];
        settings[0]="Z:/Data/UCIDelgado/";
        settings[1]="Z:/Results_7_2_19/CAWPEReproducabiltyTests/CAWPEReproducabiltyTest23/Results/";
        settings[2]="false";
        settings[3]="a";
        settings[4]="a";
        settings[5]="1";

        String[] datasets = { "flags","glass","haberman-survival","hayes-roth","heart-cleveland","heart-hungarian","heart-switzerland","heart-va","hepatitis","hill-valley","horse-colic","ilpd-indian-liver","image-segmentation","ionosphere","iris","led-display","lenses","letter","libras","low-res-spect","lung-cancer","lymphography","mammographic",
                "molec-biol-promoter","molec-biol-splice","monks-1","monks-2","monks-3","mushroom","musk-1","musk-2","nursery","oocytes_merluccius_nucleus_4d","oocytes_merluccius_states_2f","oocytes_trisopterus_nucleus_2f",
                "oocytes_trisopterus_states_5b","optical","ozone","page-blocks","parkinsons","pendigits","pima","pittsburg-bridges-MATERIAL","pittsburg-bridges-REL-L","pittsburg-bridges-SPAN","pittsburg-bridges-T-OR-D",
                "pittsburg-bridges-TYPE","planning","plant-margin","plant-shape","plant-texture","post-operative","primary-tumor","ringnorm","seeds","semeion","soybean","spambase","spect","spectf","statlog-australian-credit",
                "statlog-german-credit","statlog-heart","statlog-image","statlog-landsat","statlog-shuttle","statlog-vehicle","steel-plates","synthetic-control","teaching","thyroid","tic-tac-toe","titanic","trains","twonorm",
                "vertebral-column-2clases","vertebral-column-3clases","wall-following","waveform","waveform-noise","wine","wine-quality-red","wine-quality-white","yeast","zoo"
        };

        String[] classifiers = new String[]{ "NN", "C45",  "Logistic", "SVML" };

//        Experiments.ExperimentalArguments expSettings = new Experiments.ExperimentalArguments(settings);
//        setupAndRunMultipleExperimentsThreaded(expSettings, classifiers,datasets,0,3);
//        

        new MultipleClassifierEvaluation("Z:/Results_7_2_19/CAWPEReproducabiltyTests/CAWPEReproducabiltyTest23/Analysis", "timingsDiaTest", 3).
                setTestResultsOnly(true).
//            setTestResultsOnly(false).
        setBuildMatlabDiagrams(true).
//            setBuildMatlabDiagrams(false).
        setUseAccuracyOnly().
                setDatasets(datasets).
                readInClassifiers(classifiers, classifiers, "Z:/Results_7_2_19/CAWPEReproducabiltyTests/CAWPEReproducabiltyTest22/Results/").
                runComparison();
    }
}
