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
package evaluation.storage;

import experiments.data.DatasetLists;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import utilities.DebugPrinting;
import utilities.ErrorReport;

/**
 * Essentially a loader for many results over a given set of estimators, datasets, folds, and splits
 * 
 * This as been implemented as barebones arrays instead of large collections for speed (of execution and implementation) 
 * and memory efficiency, however depending on demand, use cases and time could be redone to be represented by e.g. maps underneath
 * 
 * Usage: 
 *      Construct the object
 *      Set estimators, datasets, folds and splits to read at MINIMUM
 *      Set any optional settings on how to read and store the results
 *      Call LOAD()
 *          Either use the big old EstimatorResults[][][][] returned, or interact with the
 *          collection via the SLICE or RETRIEVE methods 
 * 
 *      SLICE...() methods get subsets of the results already loaded into memory
 *      RETRIEVE...(...) methods get a particular stat or info from each results object
 *          retrieveAccuracies() wraps the accuracies getter as a shortcut/example 
 * 
 * todo integrate into multipleestimatorevaluation/estimatorresultsanalysis
 * todo replace old DebugPrinting stuff with loggers if/when going full enterprise
 * todo proper missing results summaries, option to reduce to largest complete subset 
 *      of split/estimator/dataset/folds
 * todo maybe use this class for other things to, e.g. instead of loading results, just check 
 *      existence, large-scale zipping/copying/moving of results files, etc
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class EstimatorResultsCollection implements DebugPrinting {

    public enum ResultsType { CLASSIFICATION, CLUSTERING, REGRESSION }
    private ResultsType resultsType = ResultsType.CLASSIFICATION;

    /**
     * EstimatorResults[split][estimator][dataset][fold]
     * Split taken to be first dimension for easy retrieval of a single split if only one is loaded
     * and you want results in the 3d form [estimator][dataset][fold]
     */
    private EstimatorResults[][][][] allResults;

    private int numDatasets;
    private String[] datasetNamesInStorage;
    private String[] datasetNamesInOutput;
    
    private int numEstimators;
    private String[] estimatorNamesInStorage;
    private String[] estimatorNamesInOutput;
    
    private int numFolds;
    private int[] folds;
    
    private int numSplits;
    private String[] splits;
    
    private int numMissingResults;
    private HashSet<String> splitsWithMissingResults;
    private HashSet<String> estimatorsWithMissingResults;
    private HashSet<String> datasetsWithMissingResults;
    private HashSet<Integer> foldsWithMissingResults;

    public static boolean printOnEstimatorNameMismatch = true;
    
    /**
     * Paths to directories containing all the estimatorNamesInStorage directories
     * with the results, in format {baseReadPath}/{estimators}/Predictions/{datasets}/{split}Fold{folds}.csv
     * 
     * If readResultsFilesDirectories.length == 1, all estimator's results read from that one path
     * else, resultsPaths.length must equal estimators.length, with each index aligning
     * to the path to read the estimator's results from.
     *
     * e.g to read 2 estimators from one directory, and another 2 from 2 different directories:
     *
     *     Index |  Paths  | Estimatorr
     *     --------------------------
     *       0   |  pathA  |   e1
     *       1   |  pathA  |   e2
     *       2   |  pathB  |   e3
     *       3   |  pathC  |   e4
     *
     */
    private String resultsFilesDirectories[];
    
    
    /**
     * If true, will null the individual prediction info of each EstimatorResults object after stats are found for it
     * 
     * Defaults to true
     */
    private boolean cleanResults = true;
    
    /**
     * If true, the returned lists are guaranteed to be of size numEstimators*numDsets*numFolds*2,
     * but entries may be null;
     * 
     * Defaults to false
     */
    private boolean allowMissingResults = false;

    /**
     * If true, will fill in missing probability distributions with one-hot vectors
     * for files read in that are missing them. intended for very old files, where you still 
     * want to calc auroc etc (metrics that need dists) for all the other classifiers 
     * that DO provide them, but also want to compare e.g accuracy with classifier that don't
     * 
     * Defaults to false
     */
    private boolean ignoreMissingDistributions = false;
    
    public EstimatorResultsCollection() {
        
    }
    
    /**
     * Creates complete copy of the other collection, but keeps the subResults instead of the 
     * other's results. Intended for use when slicing, and then manually edit the particular 
     * bit of meta info that was sliced 
     */
    private EstimatorResultsCollection(EstimatorResultsCollection other, EstimatorResults[][][][] subResults) {
        this.allResults = subResults;
        
        this.numDatasets = other.numDatasets;
        this.datasetNamesInStorage = other.datasetNamesInStorage;
        this.datasetNamesInOutput = other.datasetNamesInOutput;

        this.numEstimators = other.numEstimators;
        this.estimatorNamesInStorage = other.estimatorNamesInStorage;
        this.estimatorNamesInOutput = other.estimatorNamesInOutput;

        this.numFolds = other.numFolds;
        this.folds = other.folds;

        this.numSplits = other.numSplits;
        this.splits = other.splits;

        this.resultsFilesDirectories = other.resultsFilesDirectories;

        this.cleanResults = other.cleanResults;
        this.allowMissingResults = other.allowMissingResults;
        this.ignoreMissingDistributions = other.ignoreMissingDistributions;
    }

    /**
     * Sets the type of results to load in, i.e. classification or clustering
     */
    public void setResultsType(ResultsType resultsType) { this.resultsType = resultsType; }
    
    /**
     * Sets the number folds/resamples to read in for each estimator/dataset.
     * Will create a range of fold ids from 0(inclusive) to maxFolds(exclusive)
     */
    public void setFolds(int maxFolds) { 
        setFolds(0, maxFolds);
    }
    
    /**
     * Sets the folds/resamples to read in for each estimator/dataset
     * Will create a range of fold ids from minFolds(inclusive) to maxFolds(exclusive)
     */
    public void setFolds(int minFolds, int maxFolds) { 
        setFolds(buildRange(minFolds, maxFolds));
    }
    
    /**
     * Sets the specific folds/resamples to read in for each estimator/dataset,
     * to be used if the folds wanted to not lie in a continuous range for example
     */
    public void setFolds(int[] foldIds) {
        this.folds = foldIds;
        this.numFolds = foldIds.length;                
    }
        
    /**
     * Sets the estimators to be read in from baseReadPath. Names must correspond to directory names
     * in which {estimators}/Predictions/{datasets}/{split}Fold{folds}.csv directories/files exist
     */
    public void setEstimators(String[] estimatorNames, String[] baseReadPaths) {
        setEstimators(estimatorNames, estimatorNames, baseReadPaths);
    }
    
    /**
     * Sets the estimators to be read in from baseReadPath. Names must correspond to directory names
     * in which {estimator}/Predictions/{datasets}/{split}Fold{folds}.csv directories/files exist
     */
    public void setEstimators(String[] estimatorNamesInStorage, String[] estimatorNamesInOutput, String[] baseReadPaths) {
        if (estimatorNamesInStorage.length != estimatorNamesInOutput.length
                || estimatorNamesInStorage.length != baseReadPaths.length)
            throw new IllegalArgumentException("Estimator names lengths and paths not equal, "
                    + "estimatorNamesInStorage.length="+estimatorNamesInStorage.length
                    + " estimatorNamesInOutput.length="+estimatorNamesInOutput.length
                    + " baseReadPaths.length="+baseReadPaths.length);
        
       this.estimatorNamesInStorage = estimatorNamesInStorage;
       this.estimatorNamesInOutput = estimatorNamesInOutput;
       this.resultsFilesDirectories = baseReadPaths;
       numEstimators = estimatorNamesInOutput.length;
    }

    
    /**
     * Adds estimators to be read in from baseReadPath. Names must correspond to directory names
     * in which {estimators}/Predictions/{datasets}/{split}Fold{folds}.csv directories/files exist
     */
    public void addEstimators(String[] estimatorNames, String baseReadPath) {
        addEstimators(estimatorNames, estimatorNames, baseReadPath);
    }
    
    /**
     * Adds estimators to be read in from baseReadPath. cestimatorNamesInStorage must correspond to directory names
     * in which {estimators}/Predictions/{datasets}/{split}Fold{folds}.csv directories/files exist,
     * while estimatorNamesInOutputs can be 'cleaner' names intended for image or spreadsheet
     * outputs. The two arrays should be parallel
     */
    public void addEstimators(String[] estimatorNamesInStorage, String[] estimatorNamesInOutput, String baseReadPath) {
        if (estimatorNamesInStorage.length != estimatorNamesInOutput.length)
            throw new IllegalArgumentException("Estimator names lengths not equal, "
                    + "estimatorNamesInStorage.length="+estimatorNamesInStorage.length
                    + " estimatorNamesInOutput.length="+estimatorNamesInOutput.length);
        
        if (this.estimatorNamesInOutput == null) { //nothing initialisd yet, just set them directly
            String[] t = new String[estimatorNamesInOutput.length];
            for (int i = 0; i < estimatorNamesInOutput.length; i++)
                t[i] = baseReadPath;
            setEstimators(estimatorNamesInStorage, estimatorNamesInOutput, t);
            
            return;
        }
        
        //yay arrays 
        int origLength = this.estimatorNamesInStorage.length;
        int addedLength = estimatorNamesInStorage.length;
        
        this.estimatorNamesInStorage = Arrays.copyOf(this.estimatorNamesInStorage, origLength + addedLength);
        for (int i = 0; i < addedLength; i++)
            this.estimatorNamesInStorage[origLength + i] = estimatorNamesInStorage[i];
        
        this.estimatorNamesInOutput = Arrays.copyOf(this.estimatorNamesInOutput, origLength + addedLength);
        for (int i = 0; i < addedLength; i++)
            this.estimatorNamesInOutput[origLength + i] = estimatorNamesInOutput[i];
        
        baseReadPath.replace("\\", "/");
        if (baseReadPath.charAt(baseReadPath.length()-1) != '/')
            baseReadPath += "/";
        this.resultsFilesDirectories = Arrays.copyOf(this.resultsFilesDirectories, origLength + addedLength);
        for (int i = 0; i < addedLength; i++)
            this.resultsFilesDirectories[origLength + i] = baseReadPath;
        
        numEstimators = origLength + addedLength;
    }
    
    
    
    
    /**
     * Sets the datasets to be read in for each estimator. Names must correspond to directory names
     * in which {datasets}/{split}Fold{folds}.csv directories/files exist for each estimator,
     */
    public void setDatasets(String[] datasetNames) {
        setDatasets(datasetNames, datasetNames);
    }
    
    /**
     * Sets the datasets to be read in for each estimator. datasetNamesInStorage must correspond to directory names
     * in which {datasets}/{split}Fold{folds}.csv directories/files exist for each estimator,
     * while estimatorNamesInOutputs can be 'cleaner' names intended for image or spreadsheet
     * outputs. The two arrays should be parallel
     */
    public void setDatasets(String[] datasetNamesInStorage, String[] datasetNamesInOutput) {
        if (datasetNamesInStorage.length != datasetNamesInStorage.length)
            throw new IllegalArgumentException("Estimator datasetNamesInOutput lengths not equal, "
                    + "datasetNamesInStorage.length="+datasetNamesInStorage.length
                    + " datasetNamesInOutput.length="+datasetNamesInOutput.length);
        
        this.numDatasets = datasetNamesInStorage.length;
        this.datasetNamesInStorage = datasetNamesInStorage;
        this.datasetNamesInOutput = datasetNamesInOutput;
    }
    
    /**
     * Set to look for train fold files only for each estimator/dataset/fold
     */
    public void setSplit_Train() {
        setSplit("train");
    }

    /**
     * Sets to look for test fold files only for each estimator/dataset/fold
     */
    public void setSplit_Test() {
        setSplit("test");
    }
    
    /**
     * Sets to look for train AND test fold files for each estimator/dataset/fold
     */
    public void setSplit_TrainTest() {
        setSplits(new String[] { "train", "test" });
    }
    
    /**
     * Sets to look for a particular dataset split, test and test are currently 
     * the only options generated by e.g. ClassifierExperiments.java. In the future, things
     * like validation, cvFoldX, etc might be possible
     */
    public void setSplit(String split) {
        this.splits = new String[] { split };
        this.numSplits = 1;
    }
    
    /**
     * Sets to look for a particular dataset split, test and test are currently 
     * the only options generated by e.g. ClassifierExperiments.java. In the future, things
     * like validation, cvFoldX, etc might be possible
     */
    public void setSplits(String[] splits) {
        this.splits = splits;
        this.numSplits = splits.length;
    }
    
    /**
     * if true, will null the individual prediction info of each EstimatorResults object after stats are found for it
     * 
     * defaults to true
     */
    public void setCleanResults(boolean cleanResults) {
        this.cleanResults = cleanResults;
    }

    /**
     * if true, the returned lists are guaranteed to be of size numEstimator*numDsets*numFolds*2,
     * but entries may be null;
     * 
     * defaults to false
     */
    public void setAllowMissingResults(boolean allowMissingResults) {
        this.allowMissingResults = allowMissingResults;
    }

    /**
     * if true, will fill in missing probability distributions with one-hot vectors
     * for files read in that are missing them. intended for very old files, where you still 
     * want to calc auroc etc (metrics that need dists) for all the other classifiers
     * that DO provide them, but also want to compare e.g accuracy with classifier that don't
     * 
     * defaults to false
     */
    public void setIgnoreMissingDistributions(boolean ignoreMissingDistributions) {
        this.ignoreMissingDistributions = ignoreMissingDistributions;
    }
    
    public int getNumDatasets() {
        return numDatasets;
    }

    public String[] getDatasetNamesInStorage() {
        return datasetNamesInStorage;
    }

    public String[] getDatasetNamesInOutput() {
        return datasetNamesInOutput;
    }

    public int getNumEstimators() {
        return numEstimators;
    }

    public String[] getEstimatorNamesInStorage() {
        return estimatorNamesInStorage;
    }

    public String[] getEstimatorNamesInOutput() {
        return estimatorNamesInOutput;
    }

    public int getNumFolds() {
        return numFolds;
    }

    public int[] getFolds() {
        return folds;
    }

    public int getNumSplits() {
        return numSplits;
    }

    public String[] getSplits() {
        return splits;
    }

    public String[] getBaseReadPaths() {
        return resultsFilesDirectories;
    }

    public int getNumMissingResults() {
        return numMissingResults;
    }
        
    /**
     * If true, will null the individual prediction info of each EstimatorResults object after stats are found for it
     *
     * Defaults to true
     */
    public boolean getCleanResults() {
        return cleanResults;
    }

    /**
     * If true, the returned lists are guaranteed to be of size numEstimators*numDsets*numFolds*2,
     * but entries may be null;
     * 
     * Defaults to false
     */
    public boolean getAllowMissingResults() {
        return allowMissingResults;
    }

    /**
     * If true, will fill in missing probability distributions with one-hot vectors
     * for files read in that are missing them. intended for very old files, where you still 
     * want to calc auroc etc (metrics that need dists) for all the other classifiers 
     * that DO provide them, but also want to compare e.g accuracy with classifier that don't
     * 
     * Defaults to false
     */
    public boolean getIgnoreMissingDistributions() {
        return ignoreMissingDistributions;
    }
    
    
    public int getTotalNumResultsIgnoreMissing() { 
        return (numSplits * numEstimators * numDatasets * numFolds);
    }
        
    public int getTotalNumResults() { 
        return getTotalNumResultsIgnoreMissing() - numMissingResults;
    }
    
    @Override
    public String toString() { 
        StringBuilder sb = new StringBuilder("EstimatorResultsCollection: " + getTotalNumResults() + " total, " + numMissingResults + " missing");
        sb.append("\n\tSplits: ").append(Arrays.toString(splits));
        sb.append("\n\tEstimator: ").append(Arrays.toString(estimatorNamesInOutput));
        sb.append("\n\tDatasets: ").append(Arrays.toString(datasetNamesInOutput));
        sb.append("\n\tFolds: ").append(Arrays.toString(folds));
        
        return sb.toString();
    }
    
    
    private void confirmMinimalInfoGivenAndValid() throws Exception {
        ErrorReport err = new ErrorReport("Required results collection info missing:\n");
        
        if (resultsFilesDirectories == null) {
            err.log("\tBase path to read results from not set\n");
        } else if (resultsFilesDirectories.length == 1) {
            if (!(new File(resultsFilesDirectories[0]).exists())) {
                err.log("\tBase path to read results from cannot be found: " + resultsFilesDirectories[0] + "\n");
            }
        }
        else { //many read paths
            if (resultsFilesDirectories.length != estimatorNamesInOutput.length) {
                err.log("\tEither need to specify a single read path, or a read path for each estimator. Read paths given: " + resultsFilesDirectories.length
                        + ", estimators given: " + estimatorNamesInOutput.length + "\n");
            }
            
            for (String dir : resultsFilesDirectories) {
                if (!(new File(dir).exists())) {
                    err.log("\tA base path to read results from cannot be found: " + dir + "\n");
                }
            }
        }
                    
        if (estimatorNamesInStorage == null || estimatorNamesInStorage.length == 0)
            err.log("\tEstimators to read not set\n");
        
        if (datasetNamesInStorage == null || datasetNamesInStorage.length == 0)
            err.log("\tDatasets to read not set\n");
        
        if (folds == null || folds.length == 0)
            err.log("\tFolds to read not set\n");
        
        if (splits == null || splits.length == 0)
            err.log("\tSplits to read not set\n");
        
        err.throwIfErrors();
    }
    
    private static int[] buildRange(int minFolds, int maxFolds) { 
        int[] folds = new int[maxFolds - minFolds];
        
        int c = minFolds;
        for (int i = 0; i < maxFolds - minFolds; i++, c++)
            folds[i] = c;
        
        return folds;
    }
    
    private static int find(String[] arr, String k) {
        for (int i = 0; i < arr.length; i++)
            if (arr[i].equals(k))
                return i;
        return -1;
    }
    
    private static int find(int[] arr, int k) {
        for (int i = 0; i < arr.length; i++)
            if (arr[i] == (k))
                return i;
        return -1;
    }

    public class SliceException extends Exception {
        public SliceException(String str, String[] arr, String key) {
            super("SLICE ERROR: Attempted to slice " + str + " by " + key + " but that does not exist in " + Arrays.toString(arr));
        }

        public SliceException(String str, int[] arr, int key) {
            super("SLICE ERROR: Attempted to slice " + str + " by " + key + " but that does not exist in " + Arrays.toString(arr));
        }

    }

    public EstimatorResults loadEstimator(String path) throws FileNotFoundException, Exception {
        if (resultsType == ResultsType.CLASSIFICATION){
            return new ClassifierResults(path);
        }
        else if (resultsType == ResultsType.REGRESSION){
            return new RegressorResults(path);
        }
        else if (resultsType == ResultsType.CLUSTERING){
            return new ClustererResults(path);
        }
        else{
            throw new Exception("Invalid ResultType.");
        }
    }
    
    /**
     * Loads the splits, estimators, datasets, and folds specified from disk into memory
     * subject to the options set. 
     * 
     * @return the EstimatorResults[splits][estimators][datasets][folds] loaded in, also accessible after the call with retrieveResults()
     * @throws Exception on any number of missing file if allowMissingResults is false
     */
    public EstimatorResults[][][][] load() throws Exception {
        confirmMinimalInfoGivenAndValid();
        
        ErrorReport masterError = new ErrorReport("Results files not found:\n");

        allResults = new EstimatorResults[numSplits][numEstimators][numDatasets][numFolds];
        numMissingResults = 0;
        
        //train files may be produced via TrainAccuracyEstimate, older code
        //while test files likely by experiments, but still might be a very old file
        //so having separate checks for each.
        boolean ignoringDistsFirstTime = true;
        
        splitsWithMissingResults = new HashSet<>(splits.length);
        estimatorsWithMissingResults = new HashSet<>(estimatorNamesInOutput.length);
        datasetsWithMissingResults = new HashSet<>(datasetNamesInOutput.length);
        foldsWithMissingResults = new HashSet<>(folds.length);
        
        for (int c = 0; c < numEstimators; c++) {
            String estimatorStorage = estimatorNamesInStorage[c];
            String estimatorOutput = estimatorNamesInOutput[c];
            printlnDebug(estimatorStorage + "(" + estimatorOutput + ") reading");
            
            int estimatorFnfs = 0;
            try {
                ErrorReport perEstimatorError = new ErrorReport("FileNotFoundExceptions thrown:\n");

                for (int d = 0; d < numDatasets; d++) {
                    String datasetStorage = datasetNamesInStorage[d];
                    String datasetOutput = datasetNamesInOutput[d];
                    printlnDebug("\t" + datasetStorage + "(" + datasetOutput + ") reading");

                    for (int f = 0; f < numFolds; f++) {
                        int fold = folds[f];
                        printlnDebug("\t\t" + fold + " reading");

                        for (int s = 0; s < numSplits; s++) {
                            String split = splits[s];     
                            printlnDebug("\t\t\t" + split + " reading");

                            String readPath = resultsFilesDirectories.length == 1 ? resultsFilesDirectories[0] : resultsFilesDirectories[c];
                            try {
                                //Look for a Resample first (new name), else look for a Fold (old name).
                                try {
                                    allResults[s][c][d][f] = loadEstimator(readPath + estimatorStorage +
                                            "/Predictions/" + datasetStorage + "/" + split + "Resample" + fold + ".csv");
                                }
                                catch (FileNotFoundException ex) {
                                    allResults[s][c][d][f] = loadEstimator(readPath + estimatorStorage +
                                            "/Predictions/" + datasetStorage + "/" + split + "Fold" + fold + ".csv");
                                }

                                //This is only an issue for old ClassifierResults files, we should probably stop
                                //accepting those and just alter the results files if there are any left.
                                if (ignoreMissingDistributions && allResults[s][c][d][f] instanceof ClassifierResults) {
                                    boolean wasMissing = ((ClassifierResults)allResults[s][c][d][f]).populateMissingDists();
                                    if (wasMissing && ignoringDistsFirstTime) {
                                        System.out.println("---------Probability distributions missing, but ignored: " 
                                                + estimatorStorage + " - " + datasetStorage + " - " + f + " - train");
                                        ignoringDistsFirstTime = false;
                                    }
                                }

                                if (printOnEstimatorNameMismatch && !allResults[s][c][d][f].estimatorName.equalsIgnoreCase(estimatorNamesInStorage[c])){
                                    System.err.println("Estimator file name: \"" + allResults[s][c][d][f].estimatorName
                                            + "\" is different from input name \"" + estimatorNamesInStorage[c] +
                                            "\".");
                                }

                                allResults[s][c][d][f].findAllStatsOnce();
                                if (cleanResults)
                                    allResults[s][c][d][f].cleanPredictionInfo();
                            } catch (FileNotFoundException ex) {
                                String fileName = readPath + estimatorStorage + "/Predictions/" + datasetStorage + "/"
                                        + split + "(Resample/Fold)" + fold + ".csv";
                                if (allowMissingResults) {
                                    allResults[s][c][d][f] = null;
                                    System.out.println("Failed to load " + fileName);
                                }
                                else {
                                    perEstimatorError.log(fileName + "\n");
                                }
                                
                                estimatorFnfs++;
                                
                                splitsWithMissingResults.add(split);
                                estimatorsWithMissingResults.add(estimatorStorage);
                                datasetsWithMissingResults.add(datasetStorage);
                                foldsWithMissingResults.add(fold);
                                
                            }

                            printlnDebug("\t\t\t" + split + " successfully read in");
                        }
                        printlnDebug("\t\t" + fold + " successfully read in");
                    }
                    printlnDebug("\t" + datasetStorage + "(" + datasetOutput + ") successfully read in");
                }

                if (!perEstimatorError.isEmpty())
                    perEstimatorError.log("Total num errors for " + estimatorStorage + ": " + estimatorFnfs);
                perEstimatorError.throwIfErrors();
                printlnDebug(estimatorStorage + "(" + estimatorOutput + ") successfully read in");
            } catch (Exception e) {
                masterError.log("Estimator Errors: " + estimatorNamesInStorage[c] + "\n" + e+" ");
                e.printStackTrace();
            }
            
            numMissingResults += estimatorFnfs;
        }
        
        masterError.throwIfErrors();
        
        return allResults;
    }
    
    
    /**
     * Returns a EstimatorResultsCollection that contains the same estimator, dataset and fold
     * sets, but only the SPLITS for which all results exist for all estimators, datasets and folds.
     */
    public EstimatorResultsCollection reduceToMinimalCompleteResults_splits() throws Exception {
        if (!allowMissingResults || splitsWithMissingResults.size() == 0)
            return new EstimatorResultsCollection(this, this.allResults);
        else {
            List<String> completeSplits = new ArrayList<>(Arrays.asList(splits));
            completeSplits.removeAll(splitsWithMissingResults);
            
            EstimatorResultsCollection reducedCol = sliceSplits(completeSplits.toArray(new String[] { }));
            reductionSummary("SPLITS", completeSplits, splitsWithMissingResults);
                    
            return reducedCol;
        }
    }
    
    /**
     * Returns a EstimatorResultsCollection that contains the same split, dataset and fold
     * sets, but only the ESTIMATORS for which all results exist for all splits, datasets and folds.
     */
    public EstimatorResultsCollection reduceToMinimalCompleteResults_estimators() throws Exception {
        if (!allowMissingResults)
            return new EstimatorResultsCollection(this, this.allResults); //should be all populated anyway
        else {
            List<String> completeEstimators = new ArrayList<>(Arrays.asList(estimatorNamesInStorage));
            completeEstimators.removeAll(estimatorsWithMissingResults);
            
            EstimatorResultsCollection reducedCol = sliceEstimator(completeEstimators.toArray(new String[] { }));
            reductionSummary("ESTIMATORS", completeEstimators, estimatorsWithMissingResults);
                    
            return reducedCol;
        }
    }
    
    /**
     * Returns a EstimatorResultsCollection that contains the same split, estimator and fold
     * sets, but only the DATASETS for which all results exist for all splits, estimators and folds.
     * 
     * Mainly for use with MultipleEstimatorEvaluation, where for prototyping etc we only want
     * to compare over completed datasets for a fair comparison. 
     */
    public EstimatorResultsCollection reduceToMinimalCompleteResults_datasets() throws Exception {
        if (!allowMissingResults)
            return new EstimatorResultsCollection(this, this.allResults); //should be all populated anyway
        else {
            List<String> completeDsets = new ArrayList<>(Arrays.asList(datasetNamesInStorage));
            completeDsets.removeAll(datasetsWithMissingResults);
            
            EstimatorResultsCollection reducedCol = sliceDatasets(completeDsets.toArray(new String[] { }));
            reductionSummary("DATASETS", completeDsets, datasetsWithMissingResults);
                    
            return reducedCol;
        }
    }
    
    /**
     * Returns a EstimatorResultsCollection that contains the same split, estimator and dataset
     * sets, but only the FOLDS for which all results exist for all splits, estimators and datasets.
     */
    public EstimatorResultsCollection reduceToMinimalCompleteResults_folds() throws Exception {
        if (!allowMissingResults)
            return new EstimatorResultsCollection(this, this.allResults); //should be all populated anyway
        else {
            // ayy java 8
            List<Integer> completeFolds = new ArrayList<>(Arrays.stream(folds).boxed().collect(Collectors.toList()));
            completeFolds.removeAll(foldsWithMissingResults);
            
            EstimatorResultsCollection reducedCol = sliceFolds(completeFolds.stream().mapToInt(Integer::intValue).toArray());
            reductionSummary("FOLDS", completeFolds, foldsWithMissingResults);
                    
            return reducedCol;
        }
    }
    
    private void reductionSummary(String dim, Collection<? extends Object> remaining, Collection<? extends Object> removed) { 
        System.out.println("\n\n\n*****".replace("*", "**********"));
        System.out.println("*****".replace("*", "**********"));
        System.out.println("*****".replace("*", "**********"));
        
        System.out.println("Not all results were present. Have reduced the results space "
                + "in order to only compare the results across mutually completed "+dim+".");
        
        System.out.println("\n"+dim+" removed ("+removed.size()+"): " + removed.toString());
        
        System.out.println("\n"+dim+" remaining for comparison ("+remaining.size()+"): " + remaining.toString());
        
        System.out.println("*****".replace("*", "**********"));
        System.out.println("*****".replace("*", "**********"));
        System.out.println("*****\n\n\n".replace("*", "**********"));
    }
    
    
    /**
     * Returns a new EstimatorResultsCollection that is identical to this one (in terms of
     * settings etc) aside from only the results of the provided split is returned for each estimator/dataset/fold
     * 
     * @param split split to keep
     * @return new EstimatorResultsCollection with results for all estimators/datasets/folds, but only the split given
     * @throws java.lang.Exception if the split searched for was not loaded into this collection
     */
    public EstimatorResultsCollection sliceSplit(String split) throws Exception {
        return sliceSplits(new String[] { split });
    }
       
    /**
     * Returns a new EstimatorResultsCollection that is identical to this one (in terms of
     * settings etc) aside from only the results of the provided splits are returned for each estimator/dataset/fold
     * 
     * @param splitsToSlice splits to keep
     * @return new EstimatorResultsCollection with results for all estimators/datasets/folds, but only the splits given
     * @throws java.lang.Exception if any of the splits were not loaded into this collection
     */
    public EstimatorResultsCollection sliceSplits(String[] splitsToSlice) throws Exception {
        //perform existence checks before allocating the mem
        for (String split : splitsToSlice)
            if (find(splits, split) == -1)
                throw new SliceException("splits", splits, split);
        
        //copy across the results, for splits it's nice and easy
        EstimatorResults[][][][] subResults = new EstimatorResults[splitsToSlice.length][][][];
        for (int sts = 0; sts < splitsToSlice.length; sts++) {
            int sidOrig = find(splits, splitsToSlice[sts]); //know it exists, did checks above
            subResults[sts] = this.allResults[sidOrig];
        }
        
        //copy across the meta info to new collection object
        EstimatorResultsCollection newCol = new EstimatorResultsCollection(this, subResults);
        newCol.setSplits(splitsToSlice); //setting the particular meta info sliced
        return newCol;
    }
       
    /**
     * Returns a new EstimatorResultsCollection that is identical to this one (in terms of
     * settings etc) aside from only the results of the provided estimator is returned for each split/dataset/fold
     * 
     * If different names were provided for storage and output, the name in storage should be provided
     * 
     * @param estimator to keep
     * @return new EstimatorResultsCollection with results for all split/datasets/folds, but only the estimator given
     * @throws java.lang.Exception if the estimator searched for was not loaded into this collection
     */
    public EstimatorResultsCollection sliceEstimator(String estimator) throws Exception {
        return sliceEstimator(new String[] { estimator });
    }
    
    /**
     * Returns a new EstimatorResultsCollection that is identical to this one (in terms of
     * settings etc) aside from only the results of the provided estimators are returned for each split/dataset/fold
     * 
     * If different names were provided for storage and output, the name in storage should be provided
     * 
     * @param estimatorsToSlice estimators to keep
     * @return new EstimatorResultsCollection with results for all split/datasets/folds, but only the estimators given
     * @throws java.lang.Exception if the estimators searched for were not loaded into this collection
     */
    public EstimatorResultsCollection sliceEstimator(String[] estimatorsToSlice) throws Exception {
        int[] origEstimatorIds = new int[estimatorsToSlice.length];
        String[] keptNamesStorage = new String[estimatorsToSlice.length];
        String[] keptNamesOutput = new String[estimatorsToSlice.length];
        String[] keptReadPaths = new String[estimatorsToSlice.length];
        
        //perform existence checks before allocating the mem
        for (int i = 0; i < estimatorsToSlice.length; i++) {
            String estimator = estimatorsToSlice[i];
            origEstimatorIds[i] = find(estimatorNamesInStorage, estimator);
            if (origEstimatorIds[i] == -1)
                throw new SliceException("estimators", estimatorNamesInStorage, estimator);
            else {
                keptNamesStorage[i] = estimatorNamesInStorage[origEstimatorIds[i]];
                keptNamesOutput[i] = estimatorNamesInOutput[origEstimatorIds[i]];
                keptReadPaths[i] = resultsFilesDirectories[origEstimatorIds[i]];
            }
        }
                
        //copy across the results
        EstimatorResults[][][][] subResults = new EstimatorResults[numSplits][estimatorsToSlice.length][][];
        for (int s = 0; s < numSplits; s++)
            for (int cts = 0; cts < estimatorsToSlice.length; cts++)
                subResults[s][cts] = this.allResults[s][origEstimatorIds[cts]];
        
        //copy across the meta info to new collection object
        EstimatorResultsCollection newCol = new EstimatorResultsCollection(this, subResults);
        newCol.setEstimators(keptNamesStorage, keptNamesOutput, keptReadPaths); //setting the particular meta info sliced
        return newCol;
    }
    
    /**
     * Returns a new EstimatorResultsCollection that is identical to this one (in terms of
     * settings etc) aside from only the results of the provided dataset is returned for each split/estimator/fold
     * 
     * If different names were provided for storage and output, the name in storage should be provided
     * 
     * @param dataset dataset to keep
     * @return new EstimatorResultsCollection with results for all split/estimator/folds, but only the dataset given
     * @throws java.lang.Exception if the dataset searched for was not loaded into this collection
     */
    public EstimatorResultsCollection sliceDataset(String dataset) throws Exception {
        return sliceDatasets(new String[] { dataset });
    }
    
     /**
     * Returns a new EstimatorResultsCollection that is identical to this one (in terms of
     * settings etc) aside from only the results of the provided datasets are returned for each split/estimator/fold
     * 
     * If different names were provided for storage and output, the name in storage should be provided
     * 
     * @param datasetsToSlice datasets to keep
     * @return new EstimatorResultsCollection with results for all split/estimator/folds, but only the datasets given
     * @throws java.lang.Exception if the datasets searched for were not loaded into this collection
     */
    public EstimatorResultsCollection sliceDatasets(String[] datasetsToSlice) throws Exception {
        //perform existence checks before allocating the mem
        for (String dataset : datasetsToSlice)
            if (find(datasetNamesInStorage, dataset) == -1)
                throw new SliceException("datasets", datasetNamesInStorage, dataset);
                
        //copy across the results
        EstimatorResults[][][][] subResults = new EstimatorResults[numSplits][numEstimators][datasetsToSlice.length][];
        for (int s = 0; s < numSplits; s++) {
            for (int c = 0; c < numEstimators; c++) {
                for (int dts = 0; dts < datasetsToSlice.length; dts++) {
                    int didOrig = find(datasetNamesInStorage, datasetsToSlice[dts]); //know it exists, did checks above
                    subResults[s][c][dts] = this.allResults[s][c][didOrig];
                }
            }
        }
        
        //copy across the meta info to new collection object
        EstimatorResultsCollection newCol = new EstimatorResultsCollection(this, subResults);
        newCol.setDatasets(datasetsToSlice); //setting the particular meta info sliced
        return newCol;
    }
    
    /**
     * Returns a new EstimatorResultsCollection that is identical to this one (in terms of
     * settings etc) aside from only the results of the provided fold is returned for each split/estimator/dataset
     * 
     * @param fold fold to keep
     * @return new EstimatorResultsCollection with results for all split/estimator/dataset, but only the fold given
     * @throws java.lang.Exception if the fold searched for was not loaded into this collection
     */
    public EstimatorResultsCollection sliceFold(int fold) throws Exception {
        return sliceFolds(new int[] { fold });
    }
    
    /**
     * Returns a new EstimatorResultsCollection that is identical to this one (in terms of
     * settings etc) aside from only the results of the folds in the provided range are returned for each split/estimator/dataset
     * 
     * @param minFolds bottom of range, inclusive
     * @param maxFolds top of range, exclusive
     * @return new EstimatorResultsCollection with results for all split/estimator/datasets, but only the fold range given
     * @throws java.lang.Exception if the fold range searched for was not loaded into this collection
     */
    public EstimatorResultsCollection sliceFolds(int minFolds, int maxFolds) throws Exception {
        return sliceFolds(buildRange(minFolds, maxFolds));
    }
    
    /**
     * Returns a new EstimatorResultsCollection that is identical to this one (in terms of
     * settings etc) aside from only the results of the folds provided are returned for each split/estimator/dataset
     * 
     * @param foldsToSlice individual fold ids to keep
     * @return new EstimatorResultsCollection with results for all split/estimator/datasets, but only the folds given
     * @throws java.lang.Exception if any of the folds searched for was not loaded into this collection
     */
    public EstimatorResultsCollection sliceFolds(int[] foldsToSlice) throws Exception {
        //perform existence checks before allocating the mem
        for (int fold : foldsToSlice)
            if (find(folds, fold) == -1)
                throw new SliceException("folds", folds, fold);
                
        //copy across the results
        EstimatorResults[][][][] subResults = new EstimatorResults[numSplits][numEstimators][numDatasets][foldsToSlice.length];
        for (int s = 0; s < numSplits; s++) {
            for (int c = 0; c < numEstimators; c++) {
                for (int d = 0; d < numDatasets; d++) {
                    for (int fts = 0; fts < foldsToSlice.length; fts++) {
                        int fidOrig = find(folds, foldsToSlice[fts]); //know it exists, did checks above
                        subResults[s][c][d][fts] = this.allResults[s][c][d][fidOrig];
                    }
                }
            }
        }
        
        //copy across the meta info to new collection object
        EstimatorResultsCollection newCol = new EstimatorResultsCollection(this, subResults);
        newCol.setFolds(foldsToSlice); //setting the particular meta info sliced
        return newCol;
    }

    
    

    
    
    
    
    /**
     * Returns the accuracy (or MSE for regression) of each result object loaded in as a large array
 double[split][estimator][dataset][fold]
 
 Wrapper retrieveDoubles for accuracies
     
     * @return Array [split][estimator][dataset][fold] of doubles with accuracy from each result
     */
    public double[][][][] retrieveAccuracies() throws Exception {
        if (resultsType == ResultsType.CLASSIFICATION){
            return retrieveDoubles(ClassifierResults.GETTER_Accuracy);
        }
        else if (resultsType == ResultsType.REGRESSION){
            return retrieveDoubles(RegressorResults.GETTER_MSE);
        }
        else if (resultsType == ResultsType.CLUSTERING){
            return retrieveDoubles(ClustererResults.GETTER_Accuracy);
        }
        else{
            throw new Exception("Invalid ResultType.");
        }
    }

    /**
     * Given a function that extracts information in the form of a double from a results object, 
     * returns a big array [split][estimator][dataset][fold] of that information from
     * every result object loaded 
     * 
     * todo make generic
     * 
     * @param getter function that takes a EstimatorResults object, and returns a Double
     * @return Array [split][estimator][dataset][fold] of doubles with info from each result
     */
    public double[][][][] retrieveDoubles(Function<EstimatorResults, Double> getter) {
        double[][][][] info = new double[numSplits][numEstimators][numDatasets][numFolds];
        for (int i = 0; i < numSplits; i++)
            for (int j = 0; j < numEstimators; j++)
                for (int k = 0; k < numDatasets; k++)
                    for (int l = 0; l < numFolds; l++) 
                        info[i][j][k][l] = getter.apply(allResults[i][j][k][l]);
        return info;
    }
    
    /**
     * Given a function that extracts information in the form of a String from a results object, 
     * returns a big array [split][estimator][dataset][fold] of that information from
     * every result object loaded 
     * 
     * todo make generic
     * 
     * @param getter function that takes a EstimatorResults object, and returns a String
     * @return Array [split][estimator][dataset][fold] of String with info from each result
     */
    public String[][][][] retrieveStrings(Function<EstimatorResults, String> getter) {
        String[][][][] info = new String[numSplits][numEstimators][numDatasets][numFolds];
        for (int i = 0; i < numSplits; i++)
            for (int j = 0; j < numEstimators; j++)
                for (int k = 0; k < numDatasets; k++)
                    for (int l = 0; l < numFolds; l++) 
                        info[i][j][k][l] = getter.apply(allResults[i][j][k][l]);
        return info;
    }
    
    /**
     * Simply get all of the results in their raw/complete form. If allowMissingResults was set to true when loading results,
     * one or more entries may be null, otherwise each should be complete (the loading would have failed
     * otherwise). If cleanResults was set to true when loading results, each results object will contain the 
     * evaluation statistics and meta info for that split/estimator/dataset/fold, but not the individual
     * predictions.
     * 
     * @return the big EstimatorResults[split][estimator][dataset][fold] arrays in its raw form
     */
    public EstimatorResults[][][][] retrieveResults() {
        return allResults;
    }
    
    
    
    
    
    
    
    
    
    public static void main(String[] args) throws Exception {
        EstimatorResultsCollection col = new EstimatorResultsCollection();
        col.addEstimators(new String[] { "Logistic", "SVML", "MLP" }, "C:/JamesLPHD/CAWPEExtension/Results/");
        col.setDatasets(Arrays.copyOfRange(DatasetLists.ReducedUCI, 0, 5));
        col.setFolds(10);
        col.setSplit_Test();

        EstimatorResults[][][][] res = col.load();
        System.out.println(res.length);
        System.out.println(res[0].length);
        System.out.println(res[0][0].length);
        System.out.println(res[0][0][0].length);        
        System.out.println(res[0][0][0][0].getAcc());      
        System.out.println("");
        
        double[][][][] accs = col.retrieveAccuracies();
        System.out.println(accs.length);
        System.out.println(accs[0].length);
        System.out.println(accs[0][0].length);
        System.out.println(accs[0][0][0].length);        
        System.out.println(accs[0][0][0][0]);  
        System.out.println("");
        
        EstimatorResultsCollection subcol = col.sliceEstimator("Logistic");
        EstimatorResults[][][][] subres = subcol.retrieveResults();
        System.out.println(subres.length);
        System.out.println(subres[0].length);
        System.out.println(subres[0][0].length);
        System.out.println(subres[0][0][0].length);        
        System.out.println(subres[0][0][0][0].getAcc());      
        System.out.println("");
        
        subcol = col.sliceDataset(DatasetLists.ReducedUCI[0]);
        subres = subcol.retrieveResults();
        System.out.println(subres.length);
        System.out.println(subres[0].length);
        System.out.println(subres[0][0].length);
        System.out.println(subres[0][0][0].length);        
        System.out.println(subres[0][0][0][0].getAcc());      
        System.out.println("");
        
        subcol = col.sliceFolds(new int[] { 0, 3 });
        subres = subcol.retrieveResults();
        System.out.println(subres.length);
        System.out.println(subres[0].length);
        System.out.println(subres[0][0].length);
        System.out.println(subres[0][0][0].length);        
        System.out.println(subres[0][0][0][0].getAcc());      
        System.out.println("");
    }
}
