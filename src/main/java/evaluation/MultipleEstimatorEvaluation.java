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
import evaluation.storage.EstimatorResultsCollection;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import utilities.DebugPrinting;

/**
 * This essentially just wraps EstimatorResultsAnalysis.performFullEvaluation(...) in a nicer to use way. Will be updated over time
 * 
 * Builds summary stats, sig tests, and optionally matlab dias for the EstimatorResults objects provided/files pointed to on disk. Can optionally use
 * just the test results, if that's all that is available, or both train and test (will also compute the train test diff)
 * 
 * USAGE: see workingExampleCodeRunnableOnTSCServerMachine() for fleshed out example, in short though:
 *      Construct object, set any non-default bool options, set any non-default statistics to use, set datasets to compare on, and (rule of thumb) LASTLY add 
 *      estimators/results located in memory or on disk and call runComparison().
 * 
 *      Least-code one-off use case that's good enough for most problems is: 
 *          new MultipleEstimatorEvaluation("write/path/", "experimentName", numFolds).
 *              setDatasets(development.experiments.DataSets.UCIContinuousFileNames).
 *              readInEstimators(new String[] {"NN", "C4.5"}, baseReadingPath).
 *              runComparison();  
 * 
 * Will call findAllStatsOnce on each of the EstimatorResults (i.e. will do nothing if findAllStats has already been called elsewhere before),
 * and there's a bool (default true) to set whether to null the instance prediction info after stats are found to save memory. 
 * If some custom analysis method not defined natively in estimatorresults that uses the individual prediction info,
 * (defined using addEvaluationStatistic(String statName, Function<EstimatorResults, Double> estimatorResultsManipulatorFunction))
 will need to keep the info, but that can get problematic depending on how many estimator/datasets/folds there are
 
 For some reason, the first excel workbook writer library i found/used makes xls files (instead of xlsx) and doesn't 
 support recent excel default fonts. Just open it and saveas if you want to switch it over. There's a way to globally change font in a workbook 
 if you want to change it back

 Future work (here and in EstimatorResultsAnalysis.performFullEvaluation(...)) when wanted/needed could be to
 handle incomplete results (e.g random folds missing), more matlab figures over time, and more refactoring of the obviously bad parts of the code
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class MultipleEstimatorEvaluation implements DebugPrinting {
    private String writePath; 
    private String experimentName;
    private List<String> datasets;
    private List<String> estimatorsInStorage;
    private List<String> estimatorsInOutput;
    private List<String> readPaths;
    private Map<String, Map<String, String[]>> datasetGroupings; // Map<GroupingMethodTitle(e.g "ByNumAtts"), Map<GroupTitle(e.g "<100"), dsetsInGroup(must be subset of datasets)>>
    private EstimatorResultsCollection resultsCollection;
    private int numFolds;
    private List<PerformanceMetric> metrics;

    private EstimatorResultsCollection.ResultsType resultsType = EstimatorResultsCollection.ResultsType.CLASSIFICATION;
    
    /**
     * if true, the relevant .m files must be located in the netbeans project directory
     */
    private boolean buildMatlabDiagrams;
    
    /**
     * if true, will not attempt to load trainFold results, and will not produce stats for train or traintestdiffs results
     */
    private boolean testResultsOnly;
    
    /**
     * if true, will basically just transpose the results, and swap the dataset names for the estimatornames.
     * ranks, sig tests, etc, will then compare the 'performance of datasets'. Intended use when comparing 
     * e.g. different preprocessing techniques which are saved as arffs and then a collection of estimators
     * are evaluated on each.
     */
    private boolean evaluateDatasetsOverEstimators;
    
    /**
     * if true, will perform xmeans clustering on the estimatorXdataset results, to find data-driven datasetgroupings, as well
     * as any extra dataset groupings you've defined.
     * 
     * 1) for each dataset, each estimator's [stat] is replaced by its difference to the util_mean for that dataset
     * e.g if scores of 3 estimators on a dataset are { 0.8, 0.7, 0.6 }, the new vals will be { 0.1, 0, -0.1 }
     * 
     * 2) weka instances are formed from this data, with estimators as atts, datasets as insts
     * 
     * 3) xmeans clustering performed, as a (from a human input pov) quick way of determining number of clusters + those clusters
     * 
     * 4) perform the normal grouping analysis based on those clusters
     */
    private boolean performPostHocDsetResultsClustering;
    
    /**
     * if true, will close the matlab connected once analysis complete (if it was opened)
     * if false, will allow for multiple stats runs in a single execution, but the 
     * thread will not end while the matlab instance is open, so the connection must 
     * be closed or execution terminated manually
     */
    private boolean closeMatlabConnectionWhenFinished = true;
    
    
    /**
     * If false, all combinations of all splits/estimators/datasets/folds must be present,
     * else the evaluation will not proceed. 
     * 
     * If true, missing results shall be ignored, and only the "minimal complete subset" 
     * shall be evaluated. The minimal complete subsets comprised of  the datasets that ALL estimators
     * have completed ALL folds on. 
     * 
     * As such, the evaluation shall only be performed on datasets that all the estimators
     * have completed. If this is 0, nothing will happen, of course. 
     */
    private boolean ignoreMissingResults = false;
    
    
    /**
     * @param experimentName forms the analysis directory name, and the prefix to most files
     */
    public MultipleEstimatorEvaluation(String writePath, String experimentName, int numFolds) {
        this.writePath = writePath;
        this.experimentName = experimentName;
        this.numFolds = numFolds;
        
        this.buildMatlabDiagrams = false;
        this.testResultsOnly = true;
        this.performPostHocDsetResultsClustering = false;
        
        this.datasets = new ArrayList<>();
        this.datasetGroupings = new HashMap<>();
        this.resultsCollection = new EstimatorResultsCollection();
        
        this.estimatorsInOutput = new ArrayList<>();
        this.estimatorsInStorage = new ArrayList<>();
        this.readPaths = new ArrayList<>();
        
        this.metrics = PerformanceMetric.getDefaultStatistics();
    }

    /**
     * if true, will basically just transpose the results, and swap the dataset names for the estimatornames.
     * ranks, sig tests, etc, will then compare the 'performance of datasets'. Intended use when comparing 
     * e.g. different preprocessing techniques which are saved as arffs and then a collection of estimators
     * are evaluated on each.
     */
//    public void setEvaluateDatasetsOverEstimators(boolean evaluateDatasetsOverEstimators) {
//        this.evaluateDatasetsOverEstimators = evaluateDatasetsOverEstimators;
//    }
    
    /**
     * if true, will not attempt to load trainFold results, and will not produce stats for train or traintestdiffs results
     */
    public MultipleEstimatorEvaluation setTestResultsOnly(boolean b) {
        testResultsOnly = b;
        return this;
    }
    
    /**
     * if true, the relevant .m files must be located in the netbeans project directory
     */
    public MultipleEstimatorEvaluation setBuildMatlabDiagrams(boolean b) {
        buildMatlabDiagrams = b;
        closeMatlabConnectionWhenFinished = true;
        return this;
    }
    
    /**
     * if true, the relevant .m files must be located in the netbeans project directory
     */
    public MultipleEstimatorEvaluation setBuildMatlabDiagrams(boolean b, boolean closeMatlabConnectionWhenFinished) {
        buildMatlabDiagrams = b;
        this.closeMatlabConnectionWhenFinished = closeMatlabConnectionWhenFinished;
        return this;
    }

    /**
     * if true, will null the individual prediction info of each EstimatorResults object after stats are found
     */
    public MultipleEstimatorEvaluation setCleanResults(boolean cleanResults) {
        resultsCollection.setCleanResults(cleanResults);
        return this;
    }
    
    public MultipleEstimatorEvaluation setIgnoreMissingDistributions(boolean ignoreMissingDistributions) {
        resultsCollection.setIgnoreMissingDistributions(ignoreMissingDistributions);
        return this;
    }
    
    /**
     * if true, will perform xmeans clustering on the estimatorXdataset results, to find data-driven datasetgroupings, as well
     * as any extra dataset groupings you've defined.
     * 
     * 1) for each dataset, each estimator's [stat] is replaced by its difference to the util_mean for that dataset
      e.g if scores of 3 estimators on a dataset are { 0.8, 0.7, 0.6 }, the new vals will be { 0.1, 0, -0.1 }
 
 2) weka instances are formed from this data, with estimators as atts, datasets as insts
 
 3) xmeans clustering performed, as a (from a human input pov) quick way of determining number of clusters + those clusters
 
 4) perform the normal grouping analysis based on those clusters
     */
    public MultipleEstimatorEvaluation setPerformPostHocDsetResultsClustering(boolean b) {
        performPostHocDsetResultsClustering = b;
        return this;
    }
    
    /**
     * @param datasetListFilename the path and name of a file containing a list of datasets, one per line
     * @throws FileNotFoundException 
     */
    public MultipleEstimatorEvaluation setDatasets(String datasetListFilename) throws FileNotFoundException {
        Scanner filein = new Scanner(new File(datasetListFilename));
        
        List<String> dsets = new ArrayList<>();
        while (filein.hasNextLine())
            dsets.add(filein.nextLine());
        
        return setDatasets(dsets);
    }
    
    public MultipleEstimatorEvaluation setDatasets(List<String> datasets) {
        this.datasets = datasets;
        return this;
    }
    
    public MultipleEstimatorEvaluation setDatasets(String[] datasets) {
        this.datasets = Arrays.asList(datasets);
        return this;
    }
    
    /**
     * Pass a directory containing a number of text files. The directory name (not including path)
     * becomes the groupingMethodName (e.g ByNumAtts). Each text file contains a newline-separated
     * list of datasets for an individual group. The textfile's name (excluding .txt file suffix)
     * is the name of that group.
     */
    public MultipleEstimatorEvaluation setDatasetGroupingFromDirectory(String groupingDirectory) throws FileNotFoundException {
        setDatasetGroupingFromDirectory(groupingDirectory, (new File(groupingDirectory)).getName());
        return this;
    }
    
    /**
     * Use this if you want to define a different grouping method name to the directory name
     * for clean printing purposes/clarity. E.g directory name might be 'UCRDsetGroupingByNumAtts_2groups', but the 
     * name you define to be printed on the analysis could just be 'ByNumAtts'
     * 
     * Pass a directory containing a number of text files. Each text file contains a newline-separated
     * list of datasets for an individual group. The textfile's name (excluding .txt file suffix)
     * is the name of that group.
     */
    public MultipleEstimatorEvaluation setDatasetGroupingFromDirectory(String groupingDirectory, String customGroupingMethodName) throws FileNotFoundException {
        clearDatasetGroupings();
        addDatasetGroupingFromDirectory(groupingDirectory, customGroupingMethodName);
        return this;
    }

    /**
     * Sets the type of results to load in, i.e. classification or clustering
     */
    public void setResultsType(EstimatorResultsCollection.ResultsType resultsType) { this.resultsType = resultsType; }
    
    /**
     * Pass a directory containing a number of DIRECTORIES that define groupings. Each subdirectory contains 
     * a number of text files. The names of these subdirectories define the grouping method names. 
     * Each text file within contains a newline-separated
     * list of datasets for an individual group. The textfile's name (excluding .txt file suffix)
     * is the name of that group.
     */
    public MultipleEstimatorEvaluation addAllDatasetGroupingsInDirectory(String groupingSuperDirectory) throws FileNotFoundException {
        for (String groupingDirectory : (new File(groupingSuperDirectory)).list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String name) {
                return dir.isDirectory();
            }
        })) {
            
            addDatasetGroupingFromDirectory(groupingSuperDirectory + groupingDirectory);
        }
        return this;
    }
    
    /**
     * Pass a directory containing a number of text files. Each text file contains a newline-separated
     * list of datasets for an individual group. The textfile's name (excluding .txt file suffix)
     * is the name of that group.
     */
    public MultipleEstimatorEvaluation addDatasetGroupingFromDirectory(String groupingDirectory) throws FileNotFoundException {
        addDatasetGroupingFromDirectory(groupingDirectory, (new File(groupingDirectory)).getName());
        return this;
    }
    
    /**
     * Use this if you want to define a different grouping method name to the directory name
     * for clean printing purposes/clarity. E.g directory name might be 'UCRDsetGroupingByNumAtts_2groups', but the 
     * name you define to be printed on the analysis could just be 'ByNumAtts'
     * 
     * Pass a directory containing a number of text files. Each text file contains a newline-separated
     * list of datasets for an individual group. The textfile's name (excluding .txt file suffix)
     * is the name of that group.
     */
    public MultipleEstimatorEvaluation addDatasetGroupingFromDirectory(String groupingDirectory, String customGroupingMethodName) throws FileNotFoundException {
        File[] groups = (new File(groupingDirectory)).listFiles();
        String[] groupNames = new String[groups.length];
        String[][] dsets = new String[groups.length][];
        
        for (int i = 0; i < groups.length; i++) {
            groupNames[i] = groups[i].getName().replace(".txt", "").replace(".csv", "");
            Scanner filein = new Scanner(groups[i]);
        
            List<String> groupDsets = new ArrayList<>();
            while (filein.hasNextLine())
                groupDsets.add(filein.nextLine());
            
            dsets[i] = groupDsets.toArray(new String [] { });
        }
        
        addDatasetGrouping(customGroupingMethodName, groupNames, dsets);
        return this;
    }
    
    /**
     * The purely array based method for those inclined
     * 
     * @param groupingMethodName e.g "ByNumAtts"
     * @param groupNames e.g { "<100", ">100" }, where group name indices line up with outer array of 'groups' 
     * @param groups [groupNames.length][variablelength number of datasets]
     */
    public MultipleEstimatorEvaluation setDatasetGrouping(String groupingMethodName, String[] groupNames, String[][] groups) {
        clearDatasetGroupings();
        addDatasetGrouping(groupingMethodName, groupNames, groups);
        return this;
    }
    
    /**
     * The purely array based method for those inclined
     * 
     * @param groupingMethodName e.g "ByNumAtts"
     * @param groupNames e.g { "<100", ">100" }, where group name indices line up with outer array of 'groups' 
     * @param groups [groupNames.length][variablelength number of datasets]
     */
    public MultipleEstimatorEvaluation addDatasetGrouping(String groupingMethodName, String[] groupNames, String[][] groups) {
        Map<String, String[]> groupsMap = new HashMap<>();
        for (int i = 0; i < groupNames.length; i++)
            groupsMap.put(groupNames[i], groups[i]);
        
        datasetGroupings.put(groupingMethodName, groupsMap);
        return this;
    }
    
    public MultipleEstimatorEvaluation clearDatasetGroupings() {
        this.datasetGroupings.clear();
        return this;
    }

    /**
     * 4 stats: acc, balanced acc, auroc, nll
     */
    public MultipleEstimatorEvaluation setUseDefaultEvaluationStatistics() {
        metrics = PerformanceMetric.getDefaultStatistics();
        return this;
    }

    public MultipleEstimatorEvaluation setUseAccuracyOnly() {
        metrics = PerformanceMetric.getAccuracyStatistic();
        return this;
    }
    
    public MultipleEstimatorEvaluation setUseAllStatistics() {
        metrics = PerformanceMetric.getAllPredictionStatistics();
        return this;
    }

    public MultipleEstimatorEvaluation setUseEarlyClassificationStatistics() {
        metrics = PerformanceMetric.getEarlyClassificationStatistics();
        return this;
    }

    public MultipleEstimatorEvaluation setUseClusteringStatistics() {
        metrics = PerformanceMetric.getClusteringStatistics();
        return this;
    }

    public MultipleEstimatorEvaluation setUseRegressionStatistics() {
        metrics = PerformanceMetric.getRegressionStatistics();
        return this;
    }


    /**
     * Read in the results from file estimator by estimator, can be used if results are in different locations
     * (e.g beast vs local)
     * 
     * @param estimatorName Should exactly match the directory name of the results to use
     * @param baseReadPath Should be a directory containing a subdirectory named [estimatorName]
     * @return 
     */
    public MultipleEstimatorEvaluation readInEstimator(String estimatorName, String baseReadPath) throws Exception {
        return readInEstimator(estimatorName, estimatorName, baseReadPath);
    }
    
    /**
     * Read in the results from file estimator by estimator, can be used if results are in different locations
     * (e.g beast vs local)
     * 
     * @param estimatorNameInStorage Should exactly match the directory name of the results to use
     * @param estimatorNameInOutput Can provide a different 'human' friendly or context-aware name if appropriate, to be printed in the output files/on images
     * @param baseReadPath Should be a directory containing a subdirectory named [estimatorName]
     * @return 
     */
    public MultipleEstimatorEvaluation readInEstimator(String estimatorNameInStorage, String estimatorNameInOutput, String baseReadPath) throws Exception {
        estimatorsInStorage.add(estimatorNameInStorage);
        estimatorsInOutput.add(estimatorNameInOutput);
        readPaths.add(baseReadPath);
        return this;
    }
    /**
     * Read in the results from file from a common base path
     * 
     * @param estimatorNames Should exactly match the directory name of the results to use
     * @param baseReadPath Should be a directory containing subdirectories with the names in estimatorNames
     * @return 
     */
    public MultipleEstimatorEvaluation readInEstimators(String[] estimatorNames, String baseReadPath) throws Exception {
        return readInEstimators(estimatorNames, estimatorNames, baseReadPath);
    }
    
    /**
     * Read in the results from file from a common base path
     * 
     * @param estimatorNamesInOutput Should exactly match the directory name of the results to use
     * @param baseReadPath Should be a directory containing subdirectories with the names in estimatorNames
     * @return 
     */
    public MultipleEstimatorEvaluation readInEstimators(String[] estimatorNamesInStorage, String[] estimatorNamesInOutput, String baseReadPath) throws Exception {
        if (estimatorNamesInOutput.length != estimatorNamesInStorage.length)
            throw new Exception("Sizes of the estimator names to read in and use in output differ: estimatorNamesInStorage.length="
                    + estimatorNamesInStorage.length + ", estimatorNamesInOutput.length="+estimatorNamesInOutput.length);
        
        for (int i = 0; i < estimatorNamesInStorage.length; i++)
            readInEstimator(estimatorNamesInStorage[i], estimatorNamesInOutput[i], baseReadPath);
        
        return this;
    }
    
    /**
     * If false, all combinations of all splits/estimators/datasets/folds must be present,
     * else the evaluation will not proceed. 
     * 
     * If true, missing results shall be ignored, and only the "minimal complete subset" 
     * shall be evaluated. The minimal complete subsets comprised of  the datasets that ALL estimators
     * have completed ALL folds on. 
     * 
     * As such, the evaluation shall only be performed on datasets that all the estimators
     * have completed. If this is 0, nothing will happen, of course. 
     */
    public MultipleEstimatorEvaluation setIgnoreMissingResults(boolean ignoreMissingResults) {
        this.ignoreMissingResults = ignoreMissingResults;
        resultsCollection.setAllowMissingResults(ignoreMissingResults);
        return this;
    }
    
    private void transposeEverything() { 
//        //need to put the classifier names into the datasets list
//        //repalce the entries of the classifier results map with entries for each dataset
//        //to go from this:    Map<String/*classifierNames*/, ClassifierResults[/* train/test */][/* dataset */][/* fold */]> classifiersResults;
//        //           and a list of datasetnames
//        //to this:            Map<String/*datasetNames*/, ClassifierResults[/* train/test */][/* classifier */][/* fold */]> classifiersResults;
//        //           and a list of classifiernames
//
//        int numClassifiers = classifiersResults.size();
//        int numDatasets = datasets.size();
//
//        //going to pull everything out into parallel arrays and work that way...
//        //innefficient, but far more likely to actually work
//        String[] origClassifierNames = new String[numClassifiers];
//        ClassifierResults[][][][] origClassifierResults = new ClassifierResults[numClassifiers][][][];
//
//        int i = 0;
//        for (Map.Entry<String, ClassifierResults[][][]> origClassiiferResultsEntry : classifiersResults.entrySet()) {
//            origClassifierNames[i] = origClassiiferResultsEntry.getKey();
//            origClassifierResults[i] = origClassiiferResultsEntry.getValue();
//            i++;
//        }
//
//        ClassifierResults[][][][] newDataseResultsArr = new ClassifierResults[numDatasets][2][numClassifiers][numFolds];
//
//
//        //do the transpose
//        for (int dset = 0; dset < numDatasets; dset++) {
//
//            int splitStart = 0;
//            if (testResultsOnly) {
//                newDataseResultsArr[dset][0] = null; //no train results
//                splitStart = 1; //dont try and copythem over
//            }
//
//            for (int split = splitStart; split < 2; split++) {
//                for (int classifier = 0; classifier < numClassifiers; classifier++) {
//                    //leaving commented for reference, but can skip this loop, and copy across fold array refs instead of individual fold refs
//                    //for (int fold = 0; fold < numFolds; fold++)
//                    //    newDataseResultsArr[dset][split][classifier][fold] = origClassifierResults[classifier][split][dset][fold];
//
////                    System.out.println("newDataseResultsArr[dset]" + newDataseResultsArr[dset].toString().substring(0, 30));
////                    System.out.println("newDataseResultsArr[dset][split]" + newDataseResultsArr[dset][split].toString().substring(0, 30));
////                    System.out.println("newDataseResultsArr[dset][split][classifier]" + newDataseResultsArr[dset][split][classifier].toString().substring(0, 30));
////                    System.out.println("origClassifierResults[classifier]" + origClassifierResults[classifier].toString().substring(0, 30));
////                    System.out.println("origClassifierResults[classifier][split]" + origClassifierResults[classifier][split].toString().substring(0, 30));
////                    System.out.println("origClassifierResults[classifier][split][dset]" + origClassifierResults[classifier][split][dset].toString().substring(0, 30));
//
//                    newDataseResultsArr[dset][split][classifier] = origClassifierResults[classifier][split][dset];
//                }
//            }
//        }
//
//        //and put back into a map
//        Map<String, ClassifierResults[][][]> newDsetResultsMap = new HashMap<>();
//        for (int dset = 0; dset < numDatasets; dset++)
//            newDsetResultsMap.put(datasets.get(dset), newDataseResultsArr[dset]);
//
//        this.classifiersResults = newDsetResultsMap;
//        this.datasets = Arrays.asList(origClassifierNames);
    }
    
    public void runComparison() throws Exception {
        
        resultsCollection.setEstimators(estimatorsInStorage.toArray(new String[] { }),
                estimatorsInOutput.toArray(new String[] { }),
                readPaths.toArray(new String[] { }));
        resultsCollection.setDatasets(datasets.toArray(new String[] { }));
        resultsCollection.setFolds(numFolds);
        if (testResultsOnly)
            resultsCollection.setSplit_Test();
        else 
            resultsCollection.setSplit_TrainTest();

        resultsCollection.setResultsType(resultsType);
        
        resultsCollection.load();
        
        if (ignoreMissingResults) 
            resultsCollection = resultsCollection.reduceToMinimalCompleteResults_datasets();
        
        if (evaluateDatasetsOverEstimators)
            transposeEverything();
        
        EstimatorResultsAnalysis.buildMatlabDiagrams = buildMatlabDiagrams;
        EstimatorResultsAnalysis.testResultsOnly = testResultsOnly;
        EstimatorResultsAnalysis.resultsType = resultsType;
        
        //EstimatorResultsAnalysis will find this flag internally as queue to do clustering
        if (performPostHocDsetResultsClustering) 
            datasetGroupings.put(EstimatorResultsAnalysis.clusterGroupingIdentifier, null);
        
        printlnDebug("Writing started");
        EstimatorResultsAnalysis.performFullEvaluation(writePath, experimentName, metrics, resultsCollection, datasetGroupings);
        printlnDebug("Writing finished");
        
        if (buildMatlabDiagrams && closeMatlabConnectionWhenFinished)
            MatlabController.getInstance().discconnectMatlab();
    }

    public static void main(String[] args) throws Exception {
//        String basePath = "C:/JamesLPHD/HESCA/UCI/UCIResults/";
////            String basePath = "Z:/Results/FinalisedUCIContinuous/";
//
//        MultipleEstimatorEvaluation mcc =
//            new MultipleEstimatorEvaluation("C:/JamesLPHD/analysisTest/", "testrunPWS10", 30);
//
//        mcc.setTestResultsOnly(true); //as is default
//        mcc.setBuildMatlabDiagrams(true); //as is default
//        mcc.setCleanResults(true); //as is default
//        mcc.setDebugPrinting(true);
//
//        mcc.setUseDefaultEvaluationStatistics(); //as is default, acc,balacc,auroc,nll
////        mcc.setUseAccuracyOnly();
////        mcc.addEvaluationStatistic("F1", (ClassifierResults cr) -> {return cr.f1;}); //add on the f1 stat too
////        mcc.setUseAllStatistics();
//
//        mcc.setDatasets(development.experiments.DataSets.UCIContinuousFileNames);
//
//        //general rule of thumb: set/add/read the classifiers as the last thing before running
//        mcc.readInClassifiers(new String[] {"NN", "C4.5", "RotF", "RandF"}, basePath);
////        mcc.readInClassifier("RandF", basePath); //
//
//        mcc.runComparison();

        
//        new MultipleEstimatorEvaluation("Z:/Results/FinalisedUCIContinuousAnalysis/", "testy_mctestface", 30).
//            setTestResultsOnly(false).
//            setDatasets(development.experiments.DataSets.UCIContinuousFileNames).
//            readInClassifiers(new String[] {"1NN", "C4.5"}, "Z:/Results/FinalisedUCIContinuous/").
//            runComparison();
//        new MultipleEstimatorEvaluation("C:\\JamesLPHD\\DatasetGroups\\anatesting\\", "test29", 30).
////            setBuildMatlabDiagrams(true).
////            setUseAllStatistics().
////            setDatasets(Arrays.copyOfRange(development.experiments.DataSets.UCIContinuousFileNames, 0, 10)). //using only 10 datasets just to make it faster...
////            setDatasets("C:/Temp/dsets.txt").
//            setDatasets("C:/Temp/dsets.txt").
//            setDatasetGroupingFromDirectory("C:\\JamesLPHD\\DatasetGroups\\TestGroups").
//            setPerformPostHocDsetResultsClustering(true).
//            readInClassifiers(new String[] {"1NN", "C4.5", "MLP", "RotF", "RandF"}, "C:\\JamesLPHD\\HESCA\\UCR\\UCRResults").
//            runComparison();

        workingExampleCodeRunnableOnTSCServerMachine();
    }
    
    public static void workingExampleCodeRunnableOnTSCServerMachine() throws FileNotFoundException, Exception {
        //Running from my PC, this code takes 34 seconds to run, despite looking at only 10 folds of 10 datasets. 
        //The majority of this time is eaten up by reading the results from the server. If you have results on your local PC, this runs in a second.
        
        //to rerun this from a clean slate to check validity, delete any existing 'Example1' folder in here: 
        String folderToWriteAnalysisTo = "Z:/Backups/Results_7_2_19/FinalisedUCIContinuousAnalysis/WORKINGEXAMPLE/";
        String nameOfAnalysisWhichWillBecomeFolderName = "ExampleTranspose";
        int numberOfFoldsAKAResamplesOfEachDataset = 10;
        MultipleEstimatorEvaluation mee = new MultipleEstimatorEvaluation(folderToWriteAnalysisTo, nameOfAnalysisWhichWillBecomeFolderName, numberOfFoldsAKAResamplesOfEachDataset); //10 folds only to make faster...
        
        String aFileWithListOfDsetsToUse = "Z:/Backups/Results_7_2_19/FinalisedUCIContinuousAnalysis/WORKINGEXAMPLE/dsets.txt";
        mee.setDatasets(aFileWithListOfDsetsToUse);
        
        String aDirectoryContainingFilesThatDefineDatasetGroupings = "Z:/Backups/Results_7_2_19/FinalisedUCIContinuousAnalysis/WORKINGEXAMPLE/dsetGroupings/evenAndOddDsets/";
        String andAnother = "Z:/Backups/Results_7_2_19/FinalisedUCIContinuousAnalysis/WORKINGEXAMPLE/dsetGroupings/topAndBotHalves/";
        mee.addDatasetGroupingFromDirectory(aDirectoryContainingFilesThatDefineDatasetGroupings);
        mee.addDatasetGroupingFromDirectory(andAnother);
        
        mee.setPerformPostHocDsetResultsClustering(true); //will create 3rd data-driven grouping automatically
        
        String[] classifiers = new String[] {"1NN", "C4.5", "NB"};
        String directoryWithResultsClassifierByClassifier =  "Z:/Backups/Results_7_2_19/FinalisedUCIContinuous/";
        mee.readInEstimators(classifiers, directoryWithResultsClassifierByClassifier);
        
//        mee.setEvaluateDatasetsOverEstimators(true); //cannot use with the dataset groupings, in this example. could define classifier groupings though !
        
        mee.runComparison();
        
        //minimal version of above: 
//        MultipleEstimatorEvaluation mee = new MultipleEstimatorEvaluation("Z:/Results/FinalisedUCIContinuousAnalysis/WORKINGEXAMPLE/", "Example1", 10); //10 folds only to make faster...
//        mee.setDatasets("Z:/Results/FinalisedUCIContinuousAnalysis/WORKINGEXAMPLE/dsets.txt");
//        mee.addDatasetGroupingFromDirectory("Z:/Results/FinalisedUCIContinuousAnalysis/WORKINGEXAMPLE/dsetGroups/randomGrouping1/");
//        mee.addDatasetGroupingFromDirectory("Z:/Results/FinalisedUCIContinuousAnalysis/WORKINGEXAMPLE/dsetGroups/randomGrouping2/");
//        mee.setPerformPostHocDsetResultsClustering(true); //will create 3rd data-driven grouping automatically
//        mee.readInEstimators(new String[] {"1NN", "C4.5", "MLP", "RotF", "RandF"}, "Z:/Results/FinalisedUCIContinuous/");
//        mee.runComparison();
    }
}
