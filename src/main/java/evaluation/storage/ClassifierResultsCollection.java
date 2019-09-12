/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package evaluation.storage;

import experiments.data.DatasetLists;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import utilities.DebugPrinting;
import utilities.ErrorReport;

/**
 * Essentially a loader for many results over a given set of classifiers, datasets, folds, and splits
 * 
 * This as been implemented as barebones arrays instead of large collections for speed (of execution and implementation) 
 * and memory efficiency, however depending on demand, use cases and time could be redone to be represented by e.g. maps underneath
 * 
 * Usage: 
 *      Construct the object
 *      Set classifiers, datasets, folds and splits to read at MINIMUM
 *      Set any optional settings on how to read and store the results
 *      Call LOAD()
 *          Either use the big old ClassifierResults[][][][] returned, or interact with the 
 *          collection via the SLICE or RETRIEVE methods 
 * 
 *      SLICE...() methods get subsets of the results already loaded into memory
 *      RETRIEVE...(...) methods get a particular stat or info from each results object
 *          retrieveAccuracies() wraps the accuracies getter as a shortcut/example 
 * 
 * todo integrate into multipleclassifierevaluation/classifierresultsanalysis
 * todo replace old DebugPrinting stuff with loggers if/when going full enterprise
 * todo proper missing results summaries, option to reduce to largest complete subset 
 *      of split/classifier/dataset/folds
 * todo maybe use this class for other things to, e.g. instead of loading results, just check 
 *      existence, large-scale zipping/copying/moving of results files, etc
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class ClassifierResultsCollection implements DebugPrinting {
    
    /**
     * ClassifierResults[split][classifier][dataset][fold]
     * Split taken to be first dimension for easy retrieval of a single split if only one is loaded
     * and you want results in the 3d form [classifier][dataset][fold]
     */
    private ClassifierResults[][][][] allResults;

    private int numDatasets;
    private String[] datasetNamesInStorage;
    private String[] datasetNamesInOutput;
    
    private int numClassifiers;
    private String[] classifierNamesInStorage;
    private String[] classifierNamesInOutput;
    
    private int numFolds;
    private int[] folds;
    
    private int numSplits;
    private String[] splits;
    
    private int numMissingResults;
    private HashSet<String> splitsWithMissingResults;
    private HashSet<String> classifiersWithMissingResults;
    private HashSet<String> datasetsWithMissingResults;
    private HashSet<Integer> foldsWithMissingResults;
    
    /**
     * Paths to directories containing all the classifierNamesInStorage directories 
     * with the results, in format {baseReadPath}/{classifiers}/Predictions/{datasets}/{split}Fold{folds}.csv
     * 
     * If readResultsFilesDirectories.length == 1, all classifier's results read from that one path
     * else, resultsPaths.length must equal classifiers.length, with each index aligning
     * to the path to read the classifier's results from.
     *
     * e.g to read 2 classifiers from one directory, and another 2 from 2 different directories:
     *
     *     Index |  Paths  | Classifier
     *     --------------------------
     *       0   |  pathA  |   c1
     *       1   |  pathA  |   c2
     *       2   |  pathB  |   c3
     *       3   |  pathC  |   c4
     *
     */
    private String resultsFilesDirectories[];
    
    
    /**
     * If true, will null the individual prediction info of each ClassifierResults object after stats are found for it 
     * 
     * Defaults to true
     */
    private boolean cleanResults = true;
    
    /**
     * If true, the returned lists are guaranteed to be of size numClassifiers*numDsets*numFolds*2,
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
    
    public ClassifierResultsCollection() {
        
    }
    
    /**
     * Creates complete copy of the other collection, but keeps the subResults instead of the 
     * other's results. Intended for use when slicing, and then manually edit the particular 
     * bit of meta info that was sliced 
     */
    private ClassifierResultsCollection(ClassifierResultsCollection other, ClassifierResults[][][][] subResults) {
        this.allResults = subResults;
        
        this.numDatasets = other.numDatasets;
        this.datasetNamesInStorage = other.datasetNamesInStorage;
        this.datasetNamesInOutput = other.datasetNamesInOutput;

        this.numClassifiers = other.numClassifiers;
        this.classifierNamesInStorage = other.classifierNamesInStorage;
        this.classifierNamesInOutput = other.classifierNamesInOutput;

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
     * Sets the number folds/resamples to read in for each classifier/dataset. 
     * Will create a range of fold ids from 0(inclusive) to maxFolds(exclusive)
     */
    public void setFolds(int maxFolds) { 
        setFolds(0, maxFolds);
    }
    
    /**
     * Sets the folds/resamples to read in for each classifier/dataset
     * Will create a range of fold ids from minFolds(inclusive) to maxFolds(exclusive)
     */
    public void setFolds(int minFolds, int maxFolds) { 
        setFolds(buildRange(minFolds, maxFolds));
    }
    
    /**
     * Sets the specific folds/resamples to read in for each classifier/dataset, 
     * to be used if the folds wanted to not lie in a continuous range for example
     */
    public void setFolds(int[] foldIds) {
        this.folds = foldIds;
        this.numFolds = foldIds.length;                
    }
        
    /**
     * Sets the classifiers to be read in from baseReadPath. Names must correspond to directory names
     * in which {classifiers}/Predictions/{datasets}/{split}Fold{folds}.csv directories/files exist
     */
    public void setClassifiers(String[] classifierNames, String[] baseReadPaths) {
        setClassifiers(classifierNames, classifierNames, baseReadPaths);
    }
    
    /**
     * Sets the classifiers to be read in from baseReadPath. Names must correspond to directory names
     * in which {classifiers}/Predictions/{datasets}/{split}Fold{folds}.csv directories/files exist
     */
    public void setClassifiers(String[] classifierNamesInStorage, String[] classifierNamesInOutput, String[] baseReadPaths) {        
        if (classifierNamesInStorage.length != classifierNamesInOutput.length 
                || classifierNamesInStorage.length != baseReadPaths.length)
            throw new IllegalArgumentException("Classifier names lengths and paths not equal, "
                    + "classifierNamesInStorage.length="+classifierNamesInStorage.length
                    + " classifierNamesInOutput.length="+classifierNamesInOutput.length
                    + " baseReadPaths.length="+baseReadPaths.length);
        
       this.classifierNamesInStorage = classifierNamesInStorage;
       this.classifierNamesInOutput = classifierNamesInOutput;
       this.resultsFilesDirectories = baseReadPaths;
       numClassifiers = classifierNamesInOutput.length;
    }

    
    /**
     * Adds classifiers to be read in from baseReadPath. Names must correspond to directory names
     * in which {classifiers}/Predictions/{datasets}/{split}Fold{folds}.csv directories/files exist
     */
    public void addClassifiers(String[] classifierNames, String baseReadPath) {
        addClassifiers(classifierNames, classifierNames, baseReadPath);
    }
    
    /**
     * Adds classifiers to be read in from baseReadPath. classifierNamesInStorage must correspond to directory names
     * in which {classifiers}/Predictions/{datasets}/{split}Fold{folds}.csv directories/files exist,
     * while classifierNamesInOutputs can be 'cleaner' names intended for image or spreadsheet
     * outputs. The two arrays should be parallel
     */
    public void addClassifiers(String[] classifierNamesInStorage, String[] classifierNamesInOutput, String baseReadPath) {
        if (classifierNamesInStorage.length != classifierNamesInOutput.length)
            throw new IllegalArgumentException("Classifier names lengths not equal, "
                    + "classifierNamesInStorage.length="+classifierNamesInStorage.length
                    + " classifierNamesInOutput.length="+classifierNamesInOutput.length);
        
        if (this.classifierNamesInOutput == null) { //nothing initialisd yet, just set them directly
            String[] t = new String[classifierNamesInOutput.length];
            for (int i = 0; i < classifierNamesInOutput.length; i++)
                t[i] = baseReadPath;
            setClassifiers(classifierNamesInStorage, classifierNamesInOutput, t);
            
            return;
        }
        
        //yay arrays 
        int origLength = this.classifierNamesInStorage.length;
        int addedLength = classifierNamesInStorage.length;
        
        this.classifierNamesInStorage = Arrays.copyOf(this.classifierNamesInStorage, origLength + addedLength);
        for (int i = 0; i < addedLength; i++)
            this.classifierNamesInStorage[origLength + i] = classifierNamesInStorage[i];
        
        this.classifierNamesInOutput = Arrays.copyOf(this.classifierNamesInOutput, origLength + addedLength);
        for (int i = 0; i < addedLength; i++)
            this.classifierNamesInOutput[origLength + i] = classifierNamesInOutput[i];
        
        baseReadPath.replace("\\", "/");
        if (baseReadPath.charAt(baseReadPath.length()-1) != '/')
            baseReadPath += "/";
        this.resultsFilesDirectories = Arrays.copyOf(this.resultsFilesDirectories, origLength + addedLength);
        for (int i = 0; i < addedLength; i++)
            this.resultsFilesDirectories[origLength + i] = baseReadPath;
        
        numClassifiers = origLength + addedLength;
    }
    
    
    
    
    /**
     * Sets the datasets to be read in for each classifier. Names must correspond to directory names
     * in which {datasets}/{split}Fold{folds}.csv directories/files exist for each classifier,
     */
    public void setDatasets(String[] datasetNames) {
        setDatasets(datasetNames, datasetNames);
    }
    
    /**
     * Sets the datasets to be read in for each classifier. datasetNamesInStorage must correspond to directory names
     * in which {datasets}/{split}Fold{folds}.csv directories/files exist for each classifier,
     * while classifierNamesInOutputs can be 'cleaner' names intended for image or spreadsheet
     * outputs. The two arrays should be parallel
     */
    public void setDatasets(String[] datasetNamesInStorage, String[] datasetNamesInOutput) {
        if (datasetNamesInStorage.length != datasetNamesInStorage.length)
            throw new IllegalArgumentException("Classifier datasetNamesInOutput lengths not equal, "
                    + "datasetNamesInStorage.length="+datasetNamesInStorage.length
                    + " datasetNamesInOutput.length="+datasetNamesInOutput.length);
        
        this.numDatasets = datasetNamesInStorage.length;
        this.datasetNamesInStorage = datasetNamesInStorage;
        this.datasetNamesInOutput = datasetNamesInOutput;
    }
    
    /**
     * Set to look for train fold files only for each classifier/dataset/fold
     */
    public void setSplit_Train() {
        setSplit("train");
    }

    /**
     * Sets to look for test fold files only for each classifier/dataset/fold
     */
    public void setSplit_Test() {
        setSplit("test");
    }
    
    /**
     * Sets to look for train AND test fold files for each classifier/dataset/fold
     */
    public void setSplit_TrainTest() {
        setSplits(new String[] { "train", "test" });
    }
    
    /**
     * Sets to look for a particular dataset split, test and test are currently 
     * the only options generated by e.g. Experiments.java. In the future, things 
     * like validation, cvFoldX, etc might be possible
     */
    public void setSplit(String split) {
        this.splits = new String[] { split };
        this.numSplits = 1;
    }
    
    /**
     * Sets to look for a particular dataset split, test and test are currently 
     * the only options generated by e.g. Experiments.java. In the future, things 
     * like validation, cvFoldX, etc might be possible
     */
    public void setSplits(String[] splits) {
        this.splits = splits;
        this.numSplits = splits.length;
    }
    
    /**
     * if true, will null the individual prediction info of each ClassifierResults object after stats are found for it 
     * 
     * defaults to true
     */
    public void setCleanResults(boolean cleanResults) {
        this.cleanResults = cleanResults;
    }

    /**
     * if true, the returned lists are guaranteed to be of size numClassifiers*numDsets*numFolds*2,
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

    public int getNumClassifiers() {
        return numClassifiers;
    }

    public String[] getClassifierNamesInStorage() {
        return classifierNamesInStorage;
    }

    public String[] getClassifierNamesInOutput() {
        return classifierNamesInOutput;
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
     * If true, will null the individual prediction info of each ClassifierResults object after stats are found for it 
     * 
     * Defaults to true
     */
    public boolean getCleanResults() {
        return cleanResults;
    }

    /**
     * If true, the returned lists are guaranteed to be of size numClassifiers*numDsets*numFolds*2,
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
        return (numSplits * numClassifiers * numDatasets * numFolds);
    }
        
    public int getTotalNumResults() { 
        return getTotalNumResultsIgnoreMissing() - numMissingResults;
    }
    
    @Override
    public String toString() { 
        StringBuilder sb = new StringBuilder("ClassifierResultsCollection: " + getTotalNumResults() + " total, " + numMissingResults + " missing");
        sb.append("\n\tSplits: ").append(Arrays.toString(splits));
        sb.append("\n\tClassifiers: ").append(Arrays.toString(classifierNamesInOutput));
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
            if (resultsFilesDirectories.length != classifierNamesInOutput.length) {
                err.log("\tEither need to specify a single read path, or a read path for each classifier. Read paths given: " + resultsFilesDirectories.length
                        + ", classifiers given: " + classifierNamesInOutput.length + "\n");
            }
            
            for (String dir : resultsFilesDirectories) {
                if (!(new File(dir).exists())) {
                    err.log("\tA base path to read results from cannot be found: " + dir + "\n");
                }
            }
        }
                    
        if (classifierNamesInStorage == null || classifierNamesInStorage.length == 0)
            err.log("\tClassifiers to read not set\n");
        
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
    
    private static String buildFileName(String baseReadPath, String classifier, String dataset, String split, int fold) { 
        return baseReadPath + classifier + "/Predictions/" + dataset + "/" + split + "Fold" + fold + ".csv";
    }
    
    private static void throwSliceError(String str, String[] arr, String key) throws Exception {
        throw new Exception("SLICE ERROR: Attempted to slice " + str + " by " + key + " but that does not exist in " + Arrays.toString(arr));
    }
    
    private static void throwSliceError(String str, int[] arr, int key) throws Exception {
        throw new Exception("SLICE ERROR: Attempted to slice " + str + " by " + key + " but that does not exist in " + Arrays.toString(arr));
    }
    
    
    /**
     * Loads the splits, classifiers, datasets, and folds specified from disk into memory
     * subject to the options set. 
     * 
     * @return the ClassifierResults[splits][classifiers][datasets][folds] loaded in, also accessible after the call with retrieveResults()
     * @throws Exception on any number of missing file if allowMissingResults is false
     */
    public ClassifierResults[][][][] load() throws Exception { 
        confirmMinimalInfoGivenAndValid();
        
        ErrorReport masterError = new ErrorReport("Results files not found:\n");

        allResults = new ClassifierResults[numSplits][numClassifiers][numDatasets][numFolds];
        numMissingResults = 0;
        
        //train files may be produced via TrainAccuracyEstimate, older code
        //while test files likely by experiments, but still might be a very old file
        //so having separate checks for each.
        boolean ignoringDistsFirstTime = true;
        
        splitsWithMissingResults = new HashSet<>(splits.length);
        classifiersWithMissingResults = new HashSet<>(classifierNamesInOutput.length);
        datasetsWithMissingResults = new HashSet<>(datasetNamesInOutput.length);
        foldsWithMissingResults = new HashSet<>(folds.length);
        
        for (int c = 0; c < numClassifiers; c++) {
            String classifierStorage = classifierNamesInStorage[c];
            String classifierOutput = classifierNamesInOutput[c];
            printlnDebug(classifierStorage + "(" + classifierOutput + ") reading");
            
            int classifierFnfs = 0;
            try {
                ErrorReport perClassifierError = new ErrorReport("FileNotFoundExceptions thrown:\n");

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
                            String fileName = buildFileName(readPath, classifierStorage, datasetStorage, split, fold); 
                            try {
                                allResults[s][c][d][f] = new ClassifierResults(fileName);
                                if (ignoreMissingDistributions) {
                                    boolean wasMissing = allResults[s][c][d][f].populateMissingDists();
                                    if (wasMissing && ignoringDistsFirstTime) {
                                        System.out.println("---------Probability distributions missing, but ignored: " 
                                                + classifierStorage + " - " + datasetStorage + " - " + f + " - train");
                                        ignoringDistsFirstTime = false;
                                    }
                                }
                                allResults[s][c][d][f].findAllStatsOnce();
                                if (cleanResults)
                                    allResults[s][c][d][f].cleanPredictionInfo();
                            } catch (FileNotFoundException ex) {
                                if (allowMissingResults) {
                                    allResults[s][c][d][f] = null;
                                    System.out.println("Failed to load " + fileName);
                                }
                                else {
                                    perClassifierError.log(fileName + "\n");
                                }
                                
                                classifierFnfs++;
                                
                                splitsWithMissingResults.add(split);
                                classifiersWithMissingResults.add(classifierStorage);
                                datasetsWithMissingResults.add(datasetStorage);
                                foldsWithMissingResults.add(fold);
                                
                            }

                            printlnDebug("\t\t\t" + split + " successfully read in");
                        }
                        printlnDebug("\t\t" + fold + " successfully read in");
                    }
                    printlnDebug("\t" + datasetStorage + "(" + datasetOutput + ") successfully read in");
                }

                if (!perClassifierError.isEmpty())
                    perClassifierError.log("Total num errors for " + classifierStorage + ": " + classifierFnfs);
                perClassifierError.throwIfErrors();
                printlnDebug(classifierStorage + "(" + classifierOutput + ") successfully read in");
            } catch (Exception e) {
                masterError.log("Classifier Errors: " + classifierNamesInStorage[c] + "\n" + e);
            }
            
            numMissingResults += classifierFnfs;
        }
        
        masterError.throwIfErrors();
        
        return allResults;
    }
    
    
    /**
     * Returns a ClassifierResultsCollection that contains the same classifier, dataset and fold
     * sets, but only the SPLITS for which all results exist for all classifiers, datasets and folds.
     */
    public ClassifierResultsCollection reduceToMinimalCompleteResults_splits() throws Exception { 
        if (!allowMissingResults || splitsWithMissingResults.size() == 0)
            return new ClassifierResultsCollection(this, this.allResults);
        else {
            List<String> completeSplits = new ArrayList<>(Arrays.asList(splits));
            completeSplits.removeAll(splitsWithMissingResults);
            
            ClassifierResultsCollection reducedCol = sliceSplits(completeSplits.toArray(new String[] { }));
            reductionSummary("SPLITS", completeSplits, splitsWithMissingResults);
                    
            return reducedCol;
        }
    }
    
    /**
     * Returns a ClassifierResultsCollection that contains the same split, dataset and fold
     * sets, but only the CLASSIFIERS for which all results exist for all splits, datasets and folds.
     */
    public ClassifierResultsCollection reduceToMinimalCompleteResults_classifiers() throws Exception { 
        if (!allowMissingResults)
            return new ClassifierResultsCollection(this, this.allResults); //should be all populated anyway
        else {
            List<String> completeClassifiers = new ArrayList<>(Arrays.asList(classifierNamesInStorage));
            completeClassifiers.removeAll(classifiersWithMissingResults);
            
            ClassifierResultsCollection reducedCol = sliceClassifiers(completeClassifiers.toArray(new String[] { }));
            reductionSummary("CLASSIFIERS", completeClassifiers, classifiersWithMissingResults);
                    
            return reducedCol;
        }
    }
    
    /**
     * Returns a ClassifierResultsCollection that contains the same split, classifier and fold
     * sets, but only the DATASETS for which all results exist for all splits, classifiers and folds.
     * 
     * Mainly for use with MultipleClassifierEvaluation, where for prototyping etc we only want 
     * to compare over completed datasets for a fair comparison. 
     */
    public ClassifierResultsCollection reduceToMinimalCompleteResults_datasets() throws Exception { 
        if (!allowMissingResults)
            return new ClassifierResultsCollection(this, this.allResults); //should be all populated anyway
        else {
            List<String> completeDsets = new ArrayList<>(Arrays.asList(datasetNamesInStorage));
            completeDsets.removeAll(datasetsWithMissingResults);
            
            ClassifierResultsCollection reducedCol = sliceDatasets(completeDsets.toArray(new String[] { }));
            reductionSummary("DATASETS", completeDsets, datasetsWithMissingResults);
                    
            return reducedCol;
        }
    }
    
    /**
     * Returns a ClassifierResultsCollection that contains the same split, classifier and dataset
     * sets, but only the FOLDS for which all results exist for all splits, classifiers and datasets.
     */
    public ClassifierResultsCollection reduceToMinimalCompleteResults_folds() throws Exception { 
        if (!allowMissingResults)
            return new ClassifierResultsCollection(this, this.allResults); //should be all populated anyway
        else {
            // ayy java 8
            List<Integer> completeFolds = new ArrayList<>(Arrays.stream(folds).boxed().collect(Collectors.toList()));
            completeFolds.removeAll(foldsWithMissingResults);
            
            ClassifierResultsCollection reducedCol = sliceFolds(completeFolds.stream().mapToInt(Integer::intValue).toArray());
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
     * Returns a new ClassifierResultsCollection that is identical to this one (in terms of
     * settings etc) aside from only the results of the provided split is returned for each classifier/dataset/fold
     * 
     * @param split split to keep
     * @return new ClassifierResultsCollection with results for all classifiers/datasets/folds, but only the split given
     * @throws java.lang.Exception if the split searched for was not loaded into this collection
     */
    public ClassifierResultsCollection sliceSplit(String split) throws Exception { 
        return sliceSplits(new String[] { split });
    }
       
    /**
     * Returns a new ClassifierResultsCollection that is identical to this one (in terms of
     * settings etc) aside from only the results of the provided splits are returned for each classifier/dataset/fold
     * 
     * @param splitsToSlice splits to keep
     * @return new ClassifierResultsCollection with results for all classifiers/datasets/folds, but only the splits given
     * @throws java.lang.Exception if any of the splits were not loaded into this collection
     */
    public ClassifierResultsCollection sliceSplits(String[] splitsToSlice) throws Exception {
        //perform existence checks before allocating the mem
        for (String split : splitsToSlice)
            if (find(splits, split) == -1)
                throwSliceError("splits", splits, split);
        
        //copy across the results, for splits it's nice and easy
        ClassifierResults[][][][] subResults = new ClassifierResults[splitsToSlice.length][][][];
        for (int sts = 0; sts < splitsToSlice.length; sts++) {
            int sidOrig = find(splits, splitsToSlice[sts]); //know it exists, did checks above
            subResults[sts] = this.allResults[sidOrig];
        }
        
        //copy across the meta info to new collection object
        ClassifierResultsCollection newCol = new ClassifierResultsCollection(this, subResults);
        newCol.setSplits(splitsToSlice); //setting the particular meta info sliced
        return newCol;
    }
       
    /**
     * Returns a new ClassifierResultsCollection that is identical to this one (in terms of
     * settings etc) aside from only the results of the provided classifier is returned for each split/dataset/fold
     * 
     * If different names were provided for storage and output, the name in storage should be provided
     * 
     * @param classifier to keep
     * @return new ClassifierResultsCollection with results for all split/datasets/folds, but only the classifier given
     * @throws java.lang.Exception if the classifier searched for was not loaded into this collection
     */
    public ClassifierResultsCollection sliceClassifier(String classifier) throws Exception { 
        return sliceClassifiers(new String[] { classifier });
    }
    
    /**
     * Returns a new ClassifierResultsCollection that is identical to this one (in terms of
     * settings etc) aside from only the results of the provided classifiers are returned for each split/dataset/fold
     * 
     * If different names were provided for storage and output, the name in storage should be provided
     * 
     * @param classifiersToSlice classifiers to keep
     * @return new ClassifierResultsCollection with results for all split/datasets/folds, but only the classifiers given
     * @throws java.lang.Exception if the classifiers searched for were not loaded into this collection
     */
    public ClassifierResultsCollection sliceClassifiers(String[] classifiersToSlice) throws Exception { 
        int[] origClassifierIds = new int[classifiersToSlice.length];
        String[] keptNamesStorage = new String[classifiersToSlice.length];
        String[] keptNamesOutput = new String[classifiersToSlice.length];
        String[] keptReadPaths = new String[classifiersToSlice.length];
        
        //perform existence checks before allocating the mem
        for (int i = 0; i < classifiersToSlice.length; i++) {
            String classifier = classifiersToSlice[i];
            origClassifierIds[i] = find(classifierNamesInStorage, classifier);
            if (origClassifierIds[i] == -1)
                throwSliceError("classifiers", classifierNamesInStorage, classifier);
            else {
                keptNamesStorage[i] = classifierNamesInStorage[origClassifierIds[i]];
                keptNamesOutput[i] = classifierNamesInOutput[origClassifierIds[i]];
                keptReadPaths[i] = resultsFilesDirectories[origClassifierIds[i]];
            }
        }
                
        //copy across the results
        ClassifierResults[][][][] subResults = new ClassifierResults[numSplits][classifiersToSlice.length][][];
        for (int s = 0; s < numSplits; s++)
            for (int cts = 0; cts < classifiersToSlice.length; cts++)
                subResults[s][cts] = this.allResults[s][origClassifierIds[cts]];
        
        //copy across the meta info to new collection object
        ClassifierResultsCollection newCol = new ClassifierResultsCollection(this, subResults);
        newCol.setClassifiers(keptNamesStorage, keptNamesOutput, keptReadPaths); //setting the particular meta info sliced
        return newCol;
    }
    
    /**
     * Returns a new ClassifierResultsCollection that is identical to this one (in terms of
     * settings etc) aside from only the results of the provided dataset is returned for each split/classifier/fold
     * 
     * If different names were provided for storage and output, the name in storage should be provided
     * 
     * @param dataset dataset to keep
     * @return new ClassifierResultsCollection with results for all split/classifier/folds, but only the dataset given
     * @throws java.lang.Exception if the dataset searched for was not loaded into this collection
     */
    public ClassifierResultsCollection sliceDataset(String dataset) throws Exception { 
        return sliceDatasets(new String[] { dataset });
    }
    
     /**
     * Returns a new ClassifierResultsCollection that is identical to this one (in terms of
     * settings etc) aside from only the results of the provided datasets are returned for each split/classifier/fold
     * 
     * If different names were provided for storage and output, the name in storage should be provided
     * 
     * @param datasetsToSlice datasets to keep
     * @return new ClassifierResultsCollection with results for all split/classifier/folds, but only the datasets given
     * @throws java.lang.Exception if the datasets searched for were not loaded into this collection
     */
    public ClassifierResultsCollection sliceDatasets(String[] datasetsToSlice) throws Exception { 
        //perform existence checks before allocating the mem
        for (String dataset : datasetsToSlice)
            if (find(datasetNamesInStorage, dataset) == -1)
                throwSliceError("datasets", datasetNamesInStorage, dataset);
                
        //copy across the results
        ClassifierResults[][][][] subResults = new ClassifierResults[numSplits][numClassifiers][datasetsToSlice.length][];
        for (int s = 0; s < numSplits; s++) {
            for (int c = 0; c < numClassifiers; c++) {
                for (int dts = 0; dts < datasetsToSlice.length; dts++) {
                    int didOrig = find(datasetNamesInStorage, datasetsToSlice[dts]); //know it exists, did checks above
                    subResults[s][c][dts] = this.allResults[s][c][didOrig];
                }
            }
        }
        
        //copy across the meta info to new collection object
        ClassifierResultsCollection newCol = new ClassifierResultsCollection(this, subResults);
        newCol.setDatasets(datasetsToSlice); //setting the particular meta info sliced
        return newCol;
    }
    
    /**
     * Returns a new ClassifierResultsCollection that is identical to this one (in terms of
     * settings etc) aside from only the results of the provided fold is returned for each split/classifier/dataset
     * 
     * @param fold fold to keep
     * @return new ClassifierResultsCollection with results for all split/classifier/dataset, but only the fold given
     * @throws java.lang.Exception if the fold searched for was not loaded into this collection
     */
    public ClassifierResultsCollection sliceFold(int fold) throws Exception { 
        return sliceFolds(new int[] { fold });
    }
    
    /**
     * Returns a new ClassifierResultsCollection that is identical to this one (in terms of
     * settings etc) aside from only the results of the folds in the provided range are returned for each split/classifier/dataset
     * 
     * @param minFolds bottom of range, inclusive
     * @param maxFolds top of range, exclusive
     * @return new ClassifierResultsCollection with results for all split/classifier/datasets, but only the fold range given
     * @throws java.lang.Exception if the fold range searched for was not loaded into this collection
     */
    public ClassifierResultsCollection sliceFolds(int minFolds, int maxFolds) throws Exception { 
        return sliceFolds(buildRange(minFolds, maxFolds));
    }
    
    /**
     * Returns a new ClassifierResultsCollection that is identical to this one (in terms of
     * settings etc) aside from only the results of the folds provided are returned for each split/classifier/dataset
     * 
     * @param foldsToSlice individual fold ids to keep
     * @return new ClassifierResultsCollection with results for all split/classifier/datasets, but only the folds given
     * @throws java.lang.Exception if any of the folds searched for was not loaded into this collection
     */
    public ClassifierResultsCollection sliceFolds(int[] foldsToSlice) throws Exception { 
        //perform existence checks before allocating the mem
        for (int fold : foldsToSlice)
            if (find(folds, fold) == -1)
                throwSliceError("folds", folds, fold);
                
        //copy across the results
        ClassifierResults[][][][] subResults = new ClassifierResults[numSplits][numClassifiers][numDatasets][foldsToSlice.length];
        for (int s = 0; s < numSplits; s++) {
            for (int c = 0; c < numClassifiers; c++) {
                for (int d = 0; d < numDatasets; d++) {
                    for (int fts = 0; fts < foldsToSlice.length; fts++) {
                        int fidOrig = find(folds, foldsToSlice[fts]); //know it exists, did checks above
                        subResults[s][c][d][fts] = this.allResults[s][c][d][fidOrig];
                    }
                }
            }
        }
        
        //copy across the meta info to new collection object
        ClassifierResultsCollection newCol = new ClassifierResultsCollection(this, subResults);
        newCol.setFolds(foldsToSlice); //setting the particular meta info sliced
        return newCol;
    }

    
    

    
    
    
    
    /**
     * Returns the accuracy of each result object loaded in as a large array 
 double[split][classifier][dataset][fold]
 
 Wrapper retrieveDoubles for accuracies
     
     * @return Array [split][classifier][dataset][fold] of doubles with accuracy from each result
     */
    public double[][][][] retrieveAccuracies() {
        return retrieveDoubles(ClassifierResults.GETTER_Accuracy);
    }

    /**
     * Given a function that extracts information in the form of a double from a results object, 
     * returns a big array [split][classifier][dataset][fold] of that information from 
     * every result object loaded 
     * 
     * todo make generic
     * 
     * @param getter function that takes a ClassifierResults object, and returns a Double
     * @return Array [split][classifier][dataset][fold] of doubles with info from each result
     */
    public double[][][][] retrieveDoubles(Function<ClassifierResults, Double> getter) {
        double[][][][] info = new double[numSplits][numClassifiers][numDatasets][numFolds];
        for (int i = 0; i < numSplits; i++)
            for (int j = 0; j < numClassifiers; j++)
                for (int k = 0; k < numDatasets; k++)
                    for (int l = 0; l < numFolds; l++) 
                        info[i][j][k][l] = getter.apply(allResults[i][j][k][l]);
        return info;
    }
    
    /**
     * Given a function that extracts information in the form of a String from a results object, 
     * returns a big array [split][classifier][dataset][fold] of that information from 
     * every result object loaded 
     * 
     * todo make generic
     * 
     * @param getter function that takes a ClassifierResults object, and returns a String
     * @return Array [split][classifier][dataset][fold] of String with info from each result
     */
    public String[][][][] retrieveStrings(Function<ClassifierResults, String> getter) {
        String[][][][] info = new String[numSplits][numClassifiers][numDatasets][numFolds];
        for (int i = 0; i < numSplits; i++)
            for (int j = 0; j < numClassifiers; j++)
                for (int k = 0; k < numDatasets; k++)
                    for (int l = 0; l < numFolds; l++) 
                        info[i][j][k][l] = getter.apply(allResults[i][j][k][l]);
        return info;
    }
    
    /**
     * Simply get all of the results in their raw/complete form. If allowMissingResults was set to true when loading results,
     * one or more entries may be null, otherwise each should be complete (the loading would have failed
     * otherwise). If cleanResults was set to true when loading results, each results object will contain the 
     * evaluation statistics and meta info for that split/classifier/dataset/fold, but not the individual
     * predictions.
     * 
     * @return the big ClassifierResults[split][classifier][dataset][fold] arrays in its raw form
     */
    public ClassifierResults[][][][] retrieveResults() { 
        return allResults;
    }
    
    
    
    
    
    
    
    
    
    public static void main(String[] args) throws Exception {
        ClassifierResultsCollection col = new ClassifierResultsCollection();
        col.addClassifiers(new String[] { "Logistic", "SVML", "MLP" }, "C:/JamesLPHD/CAWPEExtension/Results/");
        col.setDatasets(Arrays.copyOfRange(DatasetLists.ReducedUCI, 0, 5));
        col.setFolds(10);
        col.setSplit_Test();
        
        ClassifierResults[][][][] res = col.load();
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
        
        ClassifierResultsCollection subcol = col.sliceClassifier("Logistic");
        ClassifierResults[][][][] subres = subcol.retrieveResults();
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
