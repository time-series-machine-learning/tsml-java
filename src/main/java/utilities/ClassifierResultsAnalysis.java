
package utilities;

import ResultsProcessing.CreatePairwiseScatter;
import ResultsProcessing.MatlabController;
import ResultsProcessing.ResultColumn;
import ResultsProcessing.ResultTable;
import development.MultipleClassifiersPairwiseTest;
import fileIO.OutFile;
import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.io.FilenameFilter;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Scanner;
import java.util.function.Function;
import jxl.Workbook;
import jxl.WorkbookSettings;
import jxl.write.WritableCellFormat;
import jxl.write.WritableFont;
import jxl.write.WritableSheet;
import jxl.write.WritableWorkbook;
import statistics.tests.OneSampleTests;
import statistics.tests.TwoSampleTests;
import utilities.ClassifierResults;
import utilities.StatisticalUtilities;
import utilities.generic_storage.Pair;
import weka.clusterers.SimpleKMeans;
import weka.clusterers.XMeans;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * This is a monster of a class, with some bad code and not enough documentation. It's improving over time however.
 * If there are any questions about it, best bet would be to email me (see below).
 * 
 * Basically functions to analyse/handle COMPLETED (i.e no folds missing out of those expected of the specified classifierXdatasetXfoldXsplit set) 
 * sets of results in ClassifierResults format
 * 
 * For some reason, the excel workbook writer library i found/used makes xls files (instead of xlsx) and doens't 
 * support recent excel default fonts. Just open it and saveas if you want to
 
 Future work when wanted/needed/motivation could be to handle incomplete results (e.g random folds missing), more matlab figures over time, 
 and a MASSIVE refactor to remove the crap code
 
 Two ways to use, the big one-off static method writeALLEvaluationFiles(...) if you have everything in memory
 in the correct format already (it is assumed that findAllStats on each classifierResults has already been called, and 
 if desired any results cleaning/nulling for space has already occurred)
 
 or you'd normally use development.MultipleClassifierEvaluation to set up results located in memory or
 on disk and call runComparison(), which essentially just wraps writeALLEvaluationFiles(...). Using this method 
 will call findAllStats on each of the classifier results, and there's a bool (default true) to set whether 
 to null the instance prediction info after stats are found to save memory. if some custom or future analysis 
 method not defined natively in classifierresults uses the individual prediction info, will need to keep it 
 * 
 * @author James Large james.large@uea.ac.uk
 */


public class ClassifierResultsAnalysis {
    
    
    public static String expRootDirectory;
    protected static String matlabFilePath = "matlabfiles/";
    protected static String pairwiseScatterDiaPath = "PairwiseScatterDias/";
    protected static String cdDiaPath = "cddias/";
    protected static String pairwiseCDDiaDirectoryName = "pairwise/";
    protected static String friedmanCDDiaDirectoryName = "friedman/";
    public static double FRIEDMANCDDIA_PVAL = 0.05;
        
    public static boolean buildMatlabDiagrams = false;
    public static boolean testResultsOnly = false;
    
    static boolean doStats = true; //the ultimate in hacks and spaghetti 
    //brought in with buildtime compilation, turns off stat code while 
    //making those files, thisll be cleaned up with time
    
    protected static final Function<ClassifierResults, Double> getAccs = (ClassifierResults cr) -> {return cr.acc;};
    protected static final Function<ClassifierResults, Double> getBalAccs = (ClassifierResults cr) -> {return cr.balancedAcc;};
    protected static final Function<ClassifierResults, Double> getAUROCs = (ClassifierResults cr) -> {return cr.meanAUROC;};
    protected static final Function<ClassifierResults, Double> getNLLs = (ClassifierResults cr) -> {return cr.nll;};
    protected static final Function<ClassifierResults, Double> getF1s = (ClassifierResults cr) -> {return cr.f1;};
    protected static final Function<ClassifierResults, Double> getMCCs = (ClassifierResults cr) -> {return cr.mcc;};
    protected static final Function<ClassifierResults, Double> getPrecisions = (ClassifierResults cr) -> {return cr.precision;};
    protected static final Function<ClassifierResults, Double> getRecalls = (ClassifierResults cr) -> {return cr.recall;};
    protected static final Function<ClassifierResults, Double> getSensitivities = (ClassifierResults cr) -> {return cr.sensitivity;};
    protected static final Function<ClassifierResults, Double> getSpecificities = (ClassifierResults cr) -> {return cr.specificity;};
    
    //START TIMING SHIT
//    //NOTE TODO BUG CHECK: due to double precision, this cast (long to double) may end up causing problems in the future. 
//    //if we were assume for now buildtime is no longer than a week, this won't cause problems
//    //some post processed ensembles (e.g HIVE COTE) might break this assumption though
//    //if the builtimes of it's members are summed
    
    //the fix for now, needed since i started using nanoseconds for part of hesca timings
    public static final int TIMING_NO_CONVERSION = 1; //aka, millis to millis
    public static final int TIMING_MILLIS_TO_SECONDS = 1000;
    public static final int TIMING_NANO_TO_MILLIS = 1000000;
    public static final int TIMING_NANO_TO_SECONDS = 1000000000;
    public static int TIMING_CONVERSION = TIMING_NO_CONVERSION;
    
    
    //todo revisit this at some point, since these are done via static variables, need a way to 
    //'remake' the func after TIMING_CONVERSION is set to a different value, since it would otherwise 
    //use the default it had to work with during its own initialisation (TIMING_NO_CONVERSION)
    public static Function<ClassifierResults, Double> initBuildTimesGetter() {
        return (ClassifierResults cr) -> {
            if (TIMING_CONVERSION == TIMING_NO_CONVERSION) {
                return (double)cr.buildTime;
            }
            else {
                //since doubles have better precision closer to zero, knock off a bunch of the lower sig values,
                //convert those to double, then add as after the decimal to get the full number expressed as a lower base
                long rawBuildTime = cr.buildTime;
                long pre = rawBuildTime / TIMING_CONVERSION;
                long post = rawBuildTime % TIMING_CONVERSION;
                double convertedPost = (double)post / TIMING_CONVERSION;
                return pre + convertedPost;
            }
        };
    }
    //END TIMING SHIT

    private static final String testLabel = "TEST";
    private static final String trainLabel = "TRAIN";
    private static final String trainTestDiffLabel = "TRAINTESTDIFFS";
    
    public static final String clusterGroupingIdentifier = "PostHocXmeansClustering";
    
    public static class ClassifierEvaluation  {
        public String classifierName;
        public ClassifierResults[][] testResults; //[dataset][fold]
        public ClassifierResults[][] trainResults; //[dataset][fold]
        public BVOnDset[] bvs; //bias variance decomposition, each element for one dataset
        
        
        public ClassifierEvaluation(String name, ClassifierResults[][] testResults, ClassifierResults[][] trainResults, BVOnDset[] bvs) {
            this.classifierName = name;
            this.testResults = testResults;
            this.trainResults = trainResults;
            this.bvs = bvs;
        }
    }
    
    public static class BVOnDset {
        //bias variance decomposition for a single classifier on a single dataset
        
        public int numClasses;
        public int[] trueClassVals; //[dataset][inst]
        public ArrayList<Integer>[] allPreds; //[dataset][inst][fold]
                
        public BVOnDset(int[] trueClassVals, int numClasses) {
            this.trueClassVals = trueClassVals;
            this.numClasses = numClasses;
            
            allPreds = new ArrayList[trueClassVals.length];
            for (int i = 0; i < allPreds.length; i++)
                allPreds[i] = new ArrayList<>();
        }
        
        /**
         * stores the test predictions on a single fold of this dataset
         */
        public void storePreds(int[] testInds, int[] testPreds) {
            for (int i = 0; i < testInds.length; i++)
                allPreds[testInds[i]].add(testPreds[i]);
        }
        
        public int[][] getFinalPreds() {
            int[][] res = new int[allPreds.length][] ;
            for (int i = 0; i < res.length; i++) {
                res[i] = new int[allPreds[i].size()];
                for (int j = 0; j < allPreds[i].size(); j++) 
                    res[i][j] = allPreds[i].get(j);
            }
            return res;
        }
    }
        
    public static ArrayList<Pair<String, Function<ClassifierResults, Double>>> getDefaultStatistics() { 
        ArrayList<Pair<String, Function<ClassifierResults, Double>>> stats = new ArrayList<>();
        stats.add(new Pair<>("ACC", getAccs));
        stats.add(new Pair<>("BALACC", getBalAccs));
        stats.add(new Pair<>("AUROC", getAUROCs));
        stats.add(new Pair<>("NLL", getNLLs));
        return stats;
    }
    
    public static ArrayList<Pair<String, Function<ClassifierResults, Double>>> getDefaultStatistics_AndF1MCC() { 
        ArrayList<Pair<String, Function<ClassifierResults, Double>>> stats = new ArrayList<>();
        stats.add(new Pair<>("ACC", getAccs));
        stats.add(new Pair<>("BALACC", getBalAccs));
        stats.add(new Pair<>("AUROC", getAUROCs));
        stats.add(new Pair<>("NLL", getNLLs));
        stats.add(new Pair<>("F1", getF1s));
        stats.add(new Pair<>("MCC", getMCCs));
        return stats;
    }
    
        
    public static ArrayList<Pair<String, Function<ClassifierResults, Double>>> getAllStatistics() { 
        ArrayList<Pair<String, Function<ClassifierResults, Double>>> stats = new ArrayList<>();
        stats.add(new Pair<>("ACC", getAccs));
        stats.add(new Pair<>("BALACC", getBalAccs));
        stats.add(new Pair<>("AUROC", getAUROCs));
        stats.add(new Pair<>("NLL", getNLLs));
        stats.add(new Pair<>("F1", getF1s));
        stats.add(new Pair<>("MCC", getMCCs));
        stats.add(new Pair<>("Prec", getPrecisions));
        stats.add(new Pair<>("Recall", getRecalls));
        stats.add(new Pair<>("Sens", getSensitivities));
        stats.add(new Pair<>("Spec", getSpecificities));
        return stats;
    }
    
    public static ArrayList<Pair<String, Function<ClassifierResults, Double>>> getAccuracyStatisticOnly() { 
        ArrayList<Pair<String, Function<ClassifierResults, Double>>> stats = new ArrayList<>();
        stats.add(new Pair<>("ACC", getAccs));
        return stats;
    }
    
    protected static void writeTableFile(String filename, String tableName, double[][] accs, String[] cnames, String[] dsets) {
        OutFile out=new OutFile(filename);
        out.writeLine(tableName + ":" + tabulate(accs, cnames, dsets));
//        out.writeLine("\navg:" + mean(accs));
        out.closeFile();
    }
    
    protected static void writeTableFileRaw(String filename, double[][] accs, String[] cnames) {
        OutFile out=new OutFile(filename);
        out.writeLine(tabulateRaw(accs, cnames));
        out.closeFile();
    }
    
    /**
     * also writes separate win/draw/loss files now
     */
    protected static String[] writeStatisticSummaryFile(String outPath, String filename, String statName, double[][][] statPerFold, double[][] statPerDset, double[][] ranks, double[][] stddevsFoldAccs, String[] cnames, String[] dsets) {   
        StringBuilder suppressedSummaryStats = new StringBuilder();
        suppressedSummaryStats.append(header(cnames)).append("\n");
        suppressedSummaryStats.append("Avg"+statName+":").append(mean(statPerDset)).append("\n");
        suppressedSummaryStats.append("Avg"+statName+"_RANK:").append(mean(ranks)).append("\n");
        
        StringBuilder summaryStats = new StringBuilder();
        
        summaryStats.append(statName).append(header(cnames)).append("\n");
        summaryStats.append("avgOverDsets:").append(mean(statPerDset)).append("\n");
        summaryStats.append("stddevOverDsets:").append(stddev(statPerDset)).append("\n");
        summaryStats.append("avgStddevsOverFolds:").append(mean(stddevsFoldAccs)).append("\n");
        summaryStats.append("avgRankOverDsets:").append(mean(ranks)).append("\n");
        summaryStats.append("stddevsRankOverDsets:").append(stddev(ranks)).append("\n");

        String[] wdl = winsDrawsLosses(statPerDset, cnames, dsets);
        String[] sig01wdl = sigWinsDrawsLosses(0.01, statPerDset, statPerFold, cnames, dsets);
        String[] sig05wdl = sigWinsDrawsLosses(0.05, statPerDset, statPerFold, cnames, dsets);
        
        (new File(outPath+"/WinsDrawsLosses/")).mkdir();
        OutFile outwdl = new OutFile(outPath+"/WinsDrawsLosses/" + filename + "_listWDLFLAT_"+statName+".csv");
        outwdl.writeLine(wdl[1]);
        outwdl.closeFile();
        outwdl = new OutFile(outPath+"/WinsDrawsLosses/" + filename + "_listWDLSig01_"+statName+".csv");
        outwdl.writeLine(sig01wdl[1]);
        outwdl.closeFile();
        outwdl = new OutFile(outPath+"/WinsDrawsLosses/" + filename + "_listWDLSig05_"+statName+".csv");
        outwdl.writeLine(sig05wdl[1]);
        outwdl.closeFile();
        
        outwdl = new OutFile(outPath+"/WinsDrawsLosses/" + filename + "_tableWDLFLAT_"+statName+".csv");
        outwdl.writeLine(wdl[2]);
        outwdl.closeFile();
        outwdl = new OutFile(outPath+"/WinsDrawsLosses/" + filename + "_tableWDLSig01_"+statName+".csv");
        outwdl.writeLine(sig01wdl[2]);
        outwdl.closeFile();
        outwdl = new OutFile(outPath+"/WinsDrawsLosses/" + filename + "_tableWDLSig05_"+statName+".csv");
        outwdl.writeLine(sig05wdl[2]);
        outwdl.closeFile();
        
        
        OutFile out=new OutFile(outPath+filename+"_"+statName+"_SUMMARY.csv");
        
        out.writeLine(summaryStats.toString());
        
        out.writeLine(wdl[0]);
        out.writeLine("\n");
        out.writeLine(sig01wdl[0]);
        out.writeLine("\n");
        out.writeLine(sig05wdl[0]);
        out.writeLine("\n");
        
        String cliques = "";
        try {
            //System.out.println(filename+"_"+statistic+".csv");
            out.writeLine(MultipleClassifiersPairwiseTest.runTests(outPath+filename+"_"+statName+".csv").toString());       
            cliques = MultipleClassifiersPairwiseTest.printCliques();
            out.writeLine("\n\n" + cliques);
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        out.closeFile();
        
        return new String[] { summaryStats.toString(), suppressedSummaryStats.toString(), cliques };
    }
    
    protected static String cdFileName(String filename, String statistic) {
        return "cd_"+filename+"_"+statistic+"S";
    }
    
    protected static String pwsFileName(String filename, String statistic) {
        return "pws_"+filename+"_"+statistic+"S";
    }
    
    
    protected static String pwsIndFileName(String c1, String c2, String statistic) {
        return "pws_"+c1+"VS"+c2+"_"+statistic+"S";
    }
    
    protected static String[] writeStatisticOnSplitFiles(String outPath, String filename, String groupingName, String evalSet, String statName, double[][][] foldVals, String[] cnames, String[] dsets, Map<String, Map<String, String[]>> dsetGroupings) throws FileNotFoundException {
        outPath += evalSet + "/";
        if (groupingName != null && !groupingName.equals(""))
            outPath += groupingName + "/";
        
        double[][] dsetVals = findAvgsOverFolds(foldVals);
        double[][] stddevsFoldVals = findStddevsOverFolds(foldVals);
        double[][] ranks = findRanks(dsetVals);
        
        //BEFORE ordering, write the individual folds files
        writePerFoldFiles(outPath+evalSet+"FOLD"+statName+"S/", foldVals, cnames, dsets, evalSet);
        
        int[] ordering = findOrdering(ranks);
        ranks = order(ranks, ordering);
        cnames = order(cnames, ordering);
        
        foldVals = order(foldVals, ordering);
        dsetVals = order(dsetVals, ordering);
        stddevsFoldVals = order(stddevsFoldVals, ordering);
        
        if (doStats) {
            if (evalSet.equalsIgnoreCase("TEST")) {
                //qol for cd dia creation, make a copy of all the raw test stat files in a common folder, one for pairwise, one for freidman
                String cdFolder = expRootDirectory + cdDiaPath;
                (new File(cdFolder)).mkdirs();
                OutFile out = new OutFile(cdFolder+"readme.txt");
                out.writeLine("remember that nlls are auto-negated now for cd dia ordering\n");
                out.writeLine("and that basic notepad wont show the line breaks properly, view (cliques especially) in notepad++");
                out.closeFile();
                for (String subFolder : new String[] { pairwiseCDDiaDirectoryName, friedmanCDDiaDirectoryName }) {
                    (new File(cdFolder+subFolder+"/")).mkdirs();
                    String cdName = cdFolder+subFolder+"/"+cdFileName(filename,statName)+".csv";
                    //meta hack for qol, negate the nll (sigh...) for correct ordering on dia
                    if (statName.contains("NLL")) {
                        double[][] negatedDsetVals = new double[dsetVals.length][dsetVals[0].length];
                        for (int i = 0; i < dsetVals.length; i++) {
                            for (int j = 0; j < dsetVals[i].length; j++) {
                                negatedDsetVals[i][j] = dsetVals[i][j] * -1;
                            }
                        }
                        writeTableFileRaw(cdName, negatedDsetVals, cnames);
                    } else {
                        writeTableFileRaw(cdName, dsetVals, cnames);
                    } 
                } //end qol

                //qol for pairwisescatter dia creation, make a copy of the test stat files 
                String pwsFolder = expRootDirectory + pairwiseScatterDiaPath;
                (new File(pwsFolder)).mkdirs();
                String pwsName = pwsFolder+pwsFileName(filename,statName)+".csv";
                writeTableFileRaw(pwsName, dsetVals, cnames);
            }
        }
        
        writeTableFile(outPath+filename+"_"+evalSet+statName+"RANKS.csv", evalSet+statName+"RANKS", ranks, cnames, dsets);
        writeTableFile(outPath+filename+"_"+evalSet+statName+".csv", evalSet+statName, dsetVals, cnames, dsets);
        writeTableFileRaw(outPath+filename+"_"+evalSet+statName+"RAW.csv", dsetVals, cnames); //for matlab stuff
        writeTableFile(outPath+filename+"_"+evalSet+statName+"STDDEVS.csv", evalSet+statName+"STDDEVS", stddevsFoldVals, cnames, dsets);
        
        String[] groupingSummary = { "" };
        if (doStats)
            if (dsetGroupings != null && dsetGroupings.size() != 0)
                groupingSummary = writeDatasetGroupingsFiles(outPath, filename, evalSet, statName, foldVals, cnames, dsets, dsetGroupings);
        
        
        String[] summaryStrings = {};
        if (doStats) {
            summaryStrings = writeStatisticSummaryFile(outPath, filename, evalSet+statName, foldVals, dsetVals, ranks, stddevsFoldVals, cnames, dsets); 

            //write these even if not actually making the dias this execution
            writeCliqueHelperFiles(expRootDirectory + cdDiaPath + pairwiseCDDiaDirectoryName, filename, statName, summaryStrings[2]); 
        }
        
        
        //this really needs cleaning up at some point... jsut make it a list and stop circlejerking to arrays
        String[] summaryStrings2 = new String[summaryStrings.length+groupingSummary.length];
        int i = 0;
        for ( ; i < summaryStrings.length; i++) 
            summaryStrings2[i] = summaryStrings[i];
        for (int j = 0; j < groupingSummary.length; j++) 
            summaryStrings2[i] = groupingSummary[j];
               
        return summaryStrings2;
    }
    
    public static String[] writeDatasetGroupingsFiles(String outPathBase, String filename, String evalSet, String statName, double[][][] foldVals, String[] cnames, String[] dsets, Map<String, Map<String, String[]>> dsetGroupings) throws FileNotFoundException {
        String outPath = expRootDirectory + "DatasetGroupings/";
//        String outPath = outPathBase + "DatasetGroupings/";
        (new File(outPath)).mkdir();
        
        //for each grouping method 
        for (Map.Entry<String, Map<String, String[]>> dsetGroupingMethodEntry : dsetGroupings.entrySet()) {
            String groupingMethodName = dsetGroupingMethodEntry.getKey();
            String groupingMethodPath = outPath + groupingMethodName + "/";
            (new File(groupingMethodPath+statName+"/"+evalSet+"/")).mkdirs();
            
            Map<String, String[]> dsetGroupingMethod = dsetGroupingMethodEntry.getValue();
            
            if (groupingMethodName.equals(clusterGroupingIdentifier)) {
                //if clustering is to be done, build the groups now.
                //can't 'put' these groups back into the dsetGroupings map
                //since we'd be editing a map that we're currently iterating over
                //EDIT: actually, jsut move this process outside the for loop as
                //a preprocess step, if the need ever arises 
                
                assert(dsetGroupingMethod == null);
                dsetGroupingMethod = new HashMap<>();
                
                int[] assignments = performDatasetResultsClustering(StatisticalUtilities.averageFinalDimension(foldVals));
                
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
                OutFile allDsetsOut = new OutFile(groupingMethodPath+statName+"/"+evalSet+"/" + "clusters.csv");
                
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
            
                String clusterGroupsPath = groupingMethodPath+statName+"/"+evalSet+"/" + "DsetClustersTxtFiles/";
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
                double[][][] groupFoldVals = collectDsetVals(foldVals, dsets, groupDsets);
                String groupFileName = filename + "-" + groupName + "-";
//                String[] groupSummaryFileStrings = writeStatisticOnSplitFiles(groupPath+statName+"/", groupFileName, groupName, evalSet, statName, groupFoldVals, cnames, groupDsets, null);
                String[] groupSummaryFileStrings = writeStatisticOnSplitFiles(groupingMethodPath+statName+"/", groupFileName, groupName, evalSet, statName, groupFoldVals, cnames, groupDsets, null);
                
                //collect the accuracies for the dataset group 
                String[] classifierNamesLine = groupSummaryFileStrings[1].split("\n")[0].split(",");
                assert(classifierNamesLine.length-1 == cnames.length);
                String[] accLineParts = groupSummaryFileStrings[1].split("\n")[1].split(",");
                for (int i = 1; i < accLineParts.length; i++) { //i=1 => skip the row header
                    double[] accs = groupAccs.get(classifierNamesLine[i]);
                    accs[groupIndex] = Double.parseDouble(accLineParts[i]);
                    groupAccs.put(classifierNamesLine[i], accs);
                }
                
                //collect the wins for the group
                Scanner ranksFileIn = new Scanner(new File(groupingMethodPath+statName+"/"+evalSet+"/"+groupName+"/"+groupFileName+"_"+evalSet+statName+"RANKS.csv"));      
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
                    List<Integer> minRanks = min(ranks);
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
            
            String groupMethodSummaryFilename = groupingMethodPath + filename + "_" + groupingMethodName + "_" + evalSet + statName + ".csv";
            datasetGroupings_writeGroupingMethodSummaryFile(groupMethodSummaryFilename, groupSummaryStringBuilders, cnames, groupNames, groupWins, groupAccs);
        }
        
        return new String[] { };
    }
    
    public static void datasetGroupings_writeGroupingMethodSummaryFile(String filename, StringBuilder [] groupSummaryStringBuilders, String[] cnames, String[] groupNames, 
            Map<String, double[]> groupWins, Map<String, double[]> groupAccs) {
        
        OutFile groupingMethodSummaryFile = new OutFile(filename);
        for (StringBuilder groupSummary : groupSummaryStringBuilders) {
            groupingMethodSummaryFile.writeLine(groupSummary.toString());
            groupingMethodSummaryFile.writeLine("\n\n");
        }

        groupingMethodSummaryFile.writeString(datasetGroupings_buildAccsTableString(groupAccs, cnames, groupNames));
        groupingMethodSummaryFile.writeLine("\n\n");
        groupingMethodSummaryFile.writeString(datasetGroupings_buildWinsTableString(groupWins, cnames, groupNames));

        groupingMethodSummaryFile.closeFile();
    }
    
    public static String datasetGroupings_buildWinsTableString(Map<String, double[]> groupWins, String[] cnames, String[] groupNames) {
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
    
    public static String datasetGroupings_buildAccsTableString(Map<String, double[]> groupAccs, String[] cnames, String[] groupNames) {
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
    
    public static double[][][] collectDsetVals(double[][][] foldVals, String[] dsets, String[] groupDsets) {
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
    
    /**
     * Essentially just a wrapper for what writeStatisticOnSplitFiles does, in the simple case that we just have a 3d array of test accs and want summaries for it
     * Mostly for legacy results not in the classifier results file format 
     */
    public static void summariseTestAccuracies(String outPath, String filename, double[][][] testFolds, String[] cnames, String[] dsets, Map<String, Map<String, String[]>> dsetGroupings) throws FileNotFoundException {
        writeStatisticOnSplitFiles(outPath, filename, null, testLabel, "ACC", testFolds, cnames, dsets, dsetGroupings);
    }
    
    protected static String[] writeStatisticFiles(String outPath, String filename, ArrayList<ClassifierEvaluation> results, Pair<String, Function<ClassifierResults, Double>> evalStatistic, String[] cnames, String[] dsets, Map<String, Map<String, String[]>> dsetGroupings) throws FileNotFoundException {
        String statName = evalStatistic.var1;
        outPath += statName + "/";
        new File(outPath).mkdirs();        
        
        double[][][] testFolds = getInfo(results, evalStatistic.var2, testLabel);
        
        if (!testResultsOnly) {
            double[][][] trainFolds = getInfo(results, evalStatistic.var2, trainLabel);
            double[][][] trainTestDiffsFolds = findTrainTestDiffs(trainFolds, testFolds);
            writeStatisticOnSplitFiles(outPath, filename, null, trainLabel, statName, trainFolds, cnames, dsets, dsetGroupings); 
            writeStatisticOnSplitFiles(outPath, filename, null, trainTestDiffLabel, statName, trainTestDiffsFolds, cnames, dsets, dsetGroupings);
        }
        
        return writeStatisticOnSplitFiles(outPath, filename, null, testLabel, statName, testFolds, cnames, dsets, dsetGroupings);
    }
    
    /**
     * TODO this is a quick edit in to get some buildtime info out, needs to be merged in properly 
     * part of problem is that we want to do this *iff* train data is available
     * other part is that buildtimes are in longs, converting to doubles can be a pain as described above
     * at the lambda definition
     */
    protected static String[] writeBuildTimeFiles(String outPath, String filename, ArrayList<ClassifierEvaluation> results, String[] cnames, String[] dsets, Map<String, Map<String, String[]>> dsetGroupings) throws FileNotFoundException {
        if (results.get(0).testResults[0][0].buildTime <= 0) { //is not present. TODO god forbid naive bayes on balloons takes less than a millisecond...
            System.out.println("Warning: No buildTimes found, or buildtimes == 0");
            return null;
        }
        
        //future proofing the spaghetti, if dostats was already false becuase of something else
        //we can leave it as it was.
        boolean prevStateOfDoStats = doStats;
        doStats = false; 
        
        Function<ClassifierResults, Double> stat = initBuildTimesGetter();
        String statName = "BuildTimes";
        outPath += statName + "/";
        new File(outPath).mkdirs();        
        
        if (!testResultsOnly) {
            double[][][] trainFolds = getInfo(results, stat, trainLabel);
            writeStatisticOnSplitFiles(outPath, filename, null, trainLabel, statName, trainFolds, cnames, dsets, dsetGroupings);
        }
        
        double[][][] testFolds = getInfo(results, stat, testLabel);
        String[] res = writeStatisticOnSplitFiles(outPath, filename, null, testLabel, statName, testFolds, cnames, dsets, dsetGroupings);

        doStats = prevStateOfDoStats;
        return res;
    }
    
    /**
     * for legacy code, will call the overloaded version with acc,balacc,nll,auroc as the default statisitics
     */
    public static void writeAllEvaluationFiles(String outPath, String expname, ArrayList<ClassifierEvaluation> results, String[] dsets) {  
        writeAllEvaluationFiles(outPath, expname, getDefaultStatistics(), results, dsets, new HashMap<String, Map<String, String[]>>());
    }
    
    public static void writeAllEvaluationFiles(String outPath, String expname, ArrayList<Pair<String,Function<ClassifierResults,Double>>> statistics, ArrayList<ClassifierEvaluation> results, String[] dsets, Map<String, Map<String, String[]>> dsetGroupings) {
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
        
        String[] cnames = getNames(results);
        String[] statCliques = new String[statistics.size()];
        String[] statNames = new String[statistics.size()];
        
        for (int i = 0; i < statistics.size(); ++i) {
            Pair<String, Function<ClassifierResults, Double>> stat = statistics.get(i);
            
            String[] summary = null;
            try { 
                summary = writeStatisticFiles(outPath, expname, results, stat, cnames, dsets, dsetGroupings);
            } catch (FileNotFoundException fnf) {
                System.out.println("Something went wrong, later stages of analysis could not find files that should have been made"
                        + "internally in earlier stages of the pipeline, FATAL");
                fnf.printStackTrace();
                System.exit(0);
            }
            
            
            bigSummary.writeString(stat.var1+":");
            bigSummary.writeLine(summary[0]);
            
            smallSummary.writeString(stat.var1+":");
            smallSummary.writeLine(summary[1]);
            
            statNames[i] = stat.var1;
            statCliques[i] = summary[2];
        }
        
        bigSummary.closeFile();
        smallSummary.closeFile();
                
        buildResultsSpreadsheet(outPath, expname, statistics);
        
        //the hacky include of buildtime files
        try { 
            writeBuildTimeFiles(outPath, expname, results, cnames, dsets, dsetGroupings);
        } catch (FileNotFoundException fnf) {
            System.out.println("Buildtime files couldnt be made");
            fnf.printStackTrace();
            System.exit(0);
        }
        
        if(buildMatlabDiagrams) {
            MatlabController proxy = MatlabController.getInstance();
            proxy.eval("addpath(genpath('"+matlabFilePath+"'))");
            buildCDDias(expname, statNames, statCliques);
            buildPairwiseScatterDiagrams(outPath, expname, statNames, dsets);
        }
    }
    
    protected static void writeCliqueHelperFiles(String cdCSVpath, String expname, String stat, String cliques) {
        (new File(cdCSVpath)).mkdirs();
        
        //temp workaround, just write the cliques and readin again from matlab for ease of checking/editing for pairwise edge cases
        OutFile out = new OutFile (cdCSVpath + cdFileName(expname, stat) + "_cliques.txt");
        out.writeString(cliques);
        out.closeFile();
    }
    
    protected static void buildCDDias(String expname, String[] stats, String[] cliques) {        
        MatlabController proxy = MatlabController.getInstance();
        proxy.eval("buildDiasInDirectory('"+expRootDirectory+"/cdDias/"+friedmanCDDiaDirectoryName+"', 0, "+FRIEDMANCDDIA_PVAL+")"); //friedman 
        proxy.eval("clear");
        proxy.eval("buildDiasInDirectory('"+expRootDirectory+"/cdDias/"+pairwiseCDDiaDirectoryName+"', 1)");  //pairwise
        proxy.eval("clear"); 
    }
        
    protected static void writePerFoldFiles(String outPath, double[][][] folds, String[] cnames, String[] dsets, String splitLabel) {
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
        
        
        OutFile out = new OutFile(outPath + "TEXASPLOT_"+splitLabel+".csv");
        out.writeString(cnames[0]);
        for (int c = 1; c < cnames.length; c++)
            out.writeString("," + cnames[c]);
        out.writeLine("");
        
        for (int d = 0; d < dsets.length; d++) {
            for (int f = 0; f < folds[0][0].length; f++) {
                out.writeDouble(folds[0][d][f]);
                for (int c = 1; c < cnames.length; c++)
                    out.writeString("," + folds[c][d][f]);
                out.writeLine("");
            }
        }
        out.closeFile();
    }
    
    protected static String tabulate(double[][] res, String[] cnames, String[] dsets) {
        StringBuilder sb = new StringBuilder();
        sb.append(header(cnames));
        
        for (int i = 0; i < res[0].length; ++i) {
            sb.append("\n").append(dsets[i]);

            for (int j = 0; j < res.length; j++)
                sb.append("," + res[j][i]);
        }      
        return sb.toString();
    }
    
    protected static String tabulateRaw(double[][] res, String[] cnames) {
        StringBuilder sb = new StringBuilder();
        sb.append(header(cnames).substring(1));
        
        for (int i = 0; i < res[0].length; ++i) {
            sb.append("\n").append(res[0][i]);
            for (int j = 1; j < res.length; j++)
                sb.append("," + res[j][i]);
        }      
        return sb.toString();
    }
    
    protected static String header(String[] names) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < names.length; i++)
            sb.append(",").append(names[i]);
        return sb.toString();
    }
    
    protected static String mean(double[][] res) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < res.length; i++) 
            sb.append(",").append(StatisticalUtilities.mean(res[i], false));
        
        return sb.toString();
    }
    
    protected static String stddev(double[][] res) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < res.length; i++) 
            sb.append(",").append(StatisticalUtilities.standardDeviation(res[i], false, StatisticalUtilities.mean(res[i], false)));
        
        return sb.toString();
    }
    
    protected static double[][][] findTrainTestDiffs(double[][][] trainFoldAccs, double[][][] testFoldAccs) {
        double[][][] diffs = new double[trainFoldAccs.length][trainFoldAccs[0].length][trainFoldAccs[0][0].length];
        
        for (int c = 0; c < diffs.length; c++) 
            for (int d = 0; d < diffs[c].length; d++) 
                for (int f = 0; f < diffs[c][d].length; f++) 
                    diffs[c][d][f] =  trainFoldAccs[c][d][f] - testFoldAccs[c][d][f];
        
        return diffs;
    }
        
    protected static double[][] findAvgsOverFolds(double[][][] foldaccs) {
        double[][] accs = new double[foldaccs.length][foldaccs[0].length];
        for (int i = 0; i < accs.length; i++)
            for (int j = 0; j < accs[i].length; j++)
                accs[i][j] = StatisticalUtilities.mean(foldaccs[i][j], false);
        
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
        
        double[] cpy = Arrays.copyOf(avgranks, avgranks.length);
        
        int[] res = new int[avgranks.length];
        
        int i = 0;
        while (i < res.length) {
            ArrayList<Integer> mins = min(avgranks);
            
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
        
        double[] cpy = Arrays.copyOf(avgranks, avgranks.length);
        
        int[] res = new int[avgranks.length];
        
        int i = 0;
        while (i < res.length) {
            ArrayList<Integer> maxs = max(avgranks);
            
            for (int j = 0; j < maxs.size(); j++) {
                res[maxs.get(j)] = i++;
                avgranks[maxs.get(j)] = -Double.MAX_VALUE;
            }
        }
        
        return res;
    }
    
    protected static ArrayList<Integer> min(double[] d) {
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
    
    protected static ArrayList<Integer> max(double[] d) {
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
    
    protected static String[] order(String[] s, int[] ordering) {
        String[] res = new String[s.length];
        
        for (int i = 0; i < ordering.length; i++) 
            res[ordering[i]] = s[i];
        
        return res;
    }
    
    protected static double[][] order(double[][] s, int[] ordering) {
        double[][] res = new double[s.length][];
        
        for (int i = 0; i < ordering.length; i++) 
            res[ordering[i]] = s[i];
        
        return res;
    }
    
    protected static double[][][] order(double[][][] s, int[] ordering) {
        double[][][] res = new double[s.length][][];
        
        for (int i = 0; i < ordering.length; i++) 
            res[ordering[i]] = s[i];
        
        return res;
    }
    
    /**
     * @param accs [classifiers][acc on datasets]
     * @return [classifiers][rank on dataset]
     */
    protected static double[][] findRanks(double[][] accs) {
        double[][] ranks = new double[accs.length][accs[0].length];
        
        for (int d = 0; d < accs[0].length; d++) {
            Double[] orderedAccs = new Double[accs.length];
            for (int c = 0; c < accs.length; c++) 
                orderedAccs[c] = accs[c][d];
            
            Arrays.sort(orderedAccs, Collections.reverseOrder());
            
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
    
    protected static String[] winsDrawsLosses(double[][] accs, String[] cnames, String[] dsets) {
        StringBuilder table = new StringBuilder();
        ArrayList<ArrayList<ArrayList<String>>> wdlList = new ArrayList<>(); //[classifierPairing][win/draw/loss][dsetNames]
        ArrayList<String> wdlListNames = new ArrayList<>();
        
        String[][] wdlPlusMinus = new String[cnames.length*cnames.length][dsets.length];
        
        table.append("flat" + header(cnames)).append("\n");
        
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
    
    protected static String[] sigWinsDrawsLosses(double pval, double[][] accs, double[][][] foldAccs, String[] cnames, String[] dsets) {
        StringBuilder table = new StringBuilder();
        ArrayList<ArrayList<ArrayList<String>>> wdlList = new ArrayList<>(); //[classifierPairing][win/draw/loss][dsetNames]
        ArrayList<String> wdlListNames = new ArrayList<>();
        
        String[][] wdlPlusMinus = new String[cnames.length*cnames.length][dsets.length];
        
        table.append("p=" + pval + header(cnames)).append("\n");
        
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
    
    protected static double[][][] getInfo(ArrayList<ClassifierEvaluation> res, Function<ClassifierResults, Double> getter, String trainortest) {
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
                System.out.println("teh fook? getInfo(), trainortest="+trainortest);
                System.exit(0);
            }
        }
        return info;
    }
        
    protected static String[] getNames(ArrayList<ClassifierEvaluation> res) {
        String[] names = new String[res.size()];
        for (int i = 0; i < res.size(); i++)
            names[i] = res.get(i).classifierName;
        return names;
    }
    
    protected static void buildResultsSpreadsheet(String basePath, String expName, ArrayList<Pair<String,Function<ClassifierResults,Double>>> statistics) {        
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
        copyCSVIntoSheet(summarySheet, summaryCSV);
        
        for (int i = 0; i < statistics.size(); i++) {
            String statName = statistics.get(i).var1;
            String path = basePath + statName + "/";
            buildStatSheets(wb, expName, path, statName, i);
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
    
    protected static void buildStatSheets(WritableWorkbook wb, String expName, String filenameprefix, String statName, int statIndex) {
        final int initialSummarySheetOffset = 1;
        int numSubStats = 3;    
        int testOffset = 0;
        
        if (!testResultsOnly) {
            numSubStats = 5; 
            testOffset = 2;
            
            WritableSheet trainSheet = wb.createSheet(statName+"Train", initialSummarySheetOffset+statIndex*numSubStats+0);
            String trainCSV = filenameprefix + trainLabel + "/" + expName + "_" +trainLabel+statName+".csv";
            copyCSVIntoSheet(trainSheet, trainCSV);

            WritableSheet trainTestDiffSheet = wb.createSheet(statName+"TrainTestDiffs", initialSummarySheetOffset+statIndex*numSubStats+1);
            String trainTestDiffCSV = filenameprefix + trainTestDiffLabel + "/" + expName + "_" +trainTestDiffLabel+statName+".csv";
            copyCSVIntoSheet(trainTestDiffSheet, trainTestDiffCSV);
        }
        
        WritableSheet testSheet = wb.createSheet(statName+"Test", initialSummarySheetOffset+statIndex*numSubStats+0+testOffset);
        String testCSV = filenameprefix + testLabel + "/" + expName + "_" +testLabel+statName+".csv";
        copyCSVIntoSheet(testSheet, testCSV);
        
        WritableSheet rankSheet = wb.createSheet(statName+"TestRanks", initialSummarySheetOffset+statIndex*numSubStats+1+testOffset);
        String rankCSV = filenameprefix + testLabel + "/" + expName + "_" +testLabel+statName+"RANKS.csv";
        copyCSVIntoSheet(rankSheet, rankCSV);
        
        WritableSheet summarySheet = wb.createSheet(statName+"TestSigDiffs", initialSummarySheetOffset+statIndex*numSubStats+2+testOffset);
        String summaryCSV = filenameprefix + testLabel + "/" + expName + "_" +testLabel+statName+"_SUMMARY.csv";
        copyCSVIntoSheet(summarySheet, summaryCSV);
        
        
    }
    
    protected static void copyCSVIntoSheet(WritableSheet sheet, String csvFile) {
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
    
    public static Pair<String[], double[][]> readTheFileFFS(String file, int numDsets) throws FileNotFoundException {
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

    public static void buildPairwiseScatterDiagrams(String outPath, String expName, String[] statNames, String[] dsets) {      
        outPath += pairwiseScatterDiaPath;
        
        for (String statName : statNames) {
            try {
                Pair<String[], double[][]> asd = readTheFileFFS(outPath + pwsFileName(expName, statName) + ".csv", dsets.length);
                ResultTable rt = new ResultTable(ResultTable.createColumns(asd.var1, dsets, asd.var2));

                int numClassiifers = rt.getColumns().size();

                MatlabController proxy = MatlabController.getInstance();
                
                for (int c1 = 0; c1 < numClassiifers-1; c1++) {
                    for (int c2 = c1+1; c2 < numClassiifers; c2++) {
                        String c1name = rt.getColumns().get(c1).getName();
                        String c2name = rt.getColumns().get(c2).getName();
                        
                        if (c1name.compareTo(c2name) > 0) {
                            String t = c1name;
                            c1name = c2name;
                            c2name = t;
                        }
                        
                        String pwFolderName = outPath + c1name + "vs" + c2name + "/";
                        (new File(pwFolderName)).mkdir();
                        
                        List<ResultColumn> pwrl = new ArrayList<>(2);
                        pwrl.add(rt.getColumn(c1name).get());
                        pwrl.add(rt.getColumn(c2name).get());
                        ResultTable pwrt = new ResultTable(pwrl);

                        proxy.eval("array = ["+ pwrt.toStringValues(false) + "];");

                        final StringBuilder concat = new StringBuilder();
                        concat.append("'");
                        concat.append(c1name.replaceAll("_", "\\\\_"));
                        concat.append("',");
                        concat.append("'");
                        concat.append(c2name.replaceAll("_", "\\\\_"));
                        concat.append("'");
                        proxy.eval("labels = {" + concat.toString() + "}");
                        
//                        System.out.println("array = ["+ pwrt.toStringValues(false) + "];");
//                        System.out.println("labels = {" + concat.toString() + "}");
//                        System.out.println("pairedscatter('" + pwFolderName + pwsIndFileName(c1name, c2name, statName).replaceAll("\\.", "") + "',array(:,1),array(:,2),labels,'"+statName+"')");
                        
                        proxy.eval("pairedscatter('" + pwFolderName + pwsIndFileName(c1name, c2name, statName).replaceAll("\\.", "") + "',array(:,1),array(:,2),labels,'"+statName+"')");
                        proxy.eval("clear");
                    }
                }
            } catch (Exception io) {
                System.out.println("buildPairwiseScatterDiagrams("+outPath+") failed loading " + statName + " file\n" + io);
            }
        }
    }
    
    
    public static int[] performDatasetResultsClustering(double[/*classifier*/][/*dataset*/] results) { 
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
}
