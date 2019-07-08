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

import java.io.File;
import java.io.FileNotFoundException;
import java.nio.file.Path;
import java.nio.file.Paths;
import utilities.DebugPrinting;
import utilities.ErrorReport;

/**
 * Essentially a loader for many results over a given set of classifiers, datasets, folds, and splits
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class Collection_New implements DebugPrinting {
    
    /**
     * ClassifierResults[split][classifier][dataset][fold]
     * Split taken to be first dimension for easy retrieval of a single split if only one is loaded
     * and you want results in the 3d form [classifier][dataset][fold]
     */
    public ClassifierResults[][][][] allResults;

    public int numDatasets;
    public String[] datasetNamesInStorage;
    public String[] datasetNamesInOutput;
    
    public int numClassifiers;
    public String[] classifierNamesInStorage;
    public String[] classifierNamesInOutput;
    
    public int numFolds;
    public int[] folds;
    
    public int numSplits;
    public String[] splits;
    
    
    /**
     * A path to a directory containing all the classifierNamesInStorage directories 
     * with the results, in format {baseReadPath}/{classifiers}/Predictions/{datasets}/{split}Fold{folds}.csv
     */
    public Path baseReadPath;
    
    
    
    /**
     * if true, will null the individual prediction info of each ClassifierResults object after stats are found for it 
     * 
     * defaults to true
     */
    private boolean cleanResults = true;

    /**
     * if true, will not attempt to load trainFold results, and will not produce stats for train or traintestdiffs results
     * 
     * defaults to true
     */
    private boolean testResultsOnly = true;
    
    /**
     * if true, the returned lists are guaranteed to be of size numClassifiers*numDsets*numFolds*2,
     * but entries may be null;
     * 
     * defaults to false
     */
    public boolean allowMissingResults = false;

    /**
     * if true, will fill in missing probability distributions with one-hot vectors
     * for files read in that are missing them. intended for very old files, where you still 
     * want to calc auroc etc (metrics that need dists) for all the other classifiers 
     * that DO provide them, but also want to compare e.g accuracy with classifier that don't
     * 
     * defaults to false
     */
    private boolean ignoreMissingDistributions = false;
    
    public Collection_New() throws Exception {
        
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
        numFolds = maxFolds - minFolds;
        folds = new int[numFolds];
        
        int c = minFolds;
        for (int i = 0; i < numFolds; i++, c++)
            folds[i] = c;
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
     * Sets the classifiers to be read in. Names must correspond to directory names
     * in which {classifiers}/Predictions/{datasets}/{split}Fold{folds}.csv directories/files exist
     */
    public void setClassifiers(String[] classifierNames) {
        setClassifiers(classifierNames, classifierNames);
    }
    
    /**
     * Sets the classifiers to be read in. classifierNamesInStorage must correspond to directory names
     * in which {classifiers}/Predictions/{datasets}/{split}Fold{folds}.csv directories/files exist,
     * while classifierNamesInOutputs can be 'cleaner' names intended for image or spreadsheet
     * outputs. The two arrays should be parallel
     */
    public void setClassifiers(String[] classifierNamesInStorage, String[] classifierNamesInOutput) {
        if (classifierNamesInStorage.length != classifierNamesInOutput.length)
            throw new IllegalArgumentException("Classifier names lengths not equal, "
                    + "classifierNamesInStorage.length="+classifierNamesInStorage.length
                    + " classifierNamesInOutput.length="+classifierNamesInOutput.length);
        
        this.numClassifiers = classifierNamesInOutput.length;
        this.classifierNamesInStorage = classifierNamesInStorage;
        this.classifierNamesInOutput = classifierNamesInOutput;
        
    }
    
    /**
     * Sets the datasets to be read in for each classifier. Names must correspond to directory names
     * in which {datasets}/{split}Fold{folds}.csv directories/files exist for each classifier,
     */
    public void setDatasets(String[] datasetNames) {
        setClassifiers(datasetNames, datasetNames);
    }
    
    /**
     * Sets the datasets to be read in for each classifier. datasetNamesInStorage must correspond to directory names
     * in which {datasets}/{split}Fold{folds}.csv directories/files exist for each classifier,
     * while classifierNamesInOutputs can be 'cleaner' names intended for image or spreadsheet
     * outputs. The two arrays should be parallel
     */
    public void setDatasets(String[] datasetNamesInStorage, String[] datasetNamesInOutput) {
        if (datasetNamesInStorage.length != classifierNamesInOutput.length)
            throw new IllegalArgumentException("Classifier datasetNamesInOutput lengths not equal, "
                    + "datasetNamesInStorage.length="+datasetNamesInStorage.length
                    + " datasetNamesInOutput.length="+datasetNamesInOutput.length);
        
        this.datasetNamesInStorage = datasetNamesInStorage;
        this.datasetNamesInOutput = datasetNamesInOutput;
    }
    
    /**
     * Set to look for train fold files only for each classifier/dataset/fold
     */
    public void setSplit_Train() {
        this.splits = new String[] { "train" };
    }

    /**
     * Sets to look for test fold files only for each classifier/dataset/fold
     */
    public void setSplit_Test() {
        this.splits = new String[] { "test" };
    }
    
    /**
     * Sets to look for train AND test fold files for each classifier/dataset/fold
     */
    public void setSplit_TrainTest() {
        this.splits = new String[] { "train", "test" };
    }
    
    /**
     * Sets the path to a directory containing all the classifierNamesInStorage directories 
     * with the results, in format {baseReadPath}/{classifiers}/Predictions/{datasets}/{split}Fold{folds}.csv
     */
    public void setBaseReadPath(String baseReadPath) {
        this.baseReadPath = Paths.get(baseReadPath);
    }
    
    /**
     * Sets the path to a directory containing all the classifierNamesInStorage directories 
     * with the results, in format {baseReadPath}/{classifiers}/Predictions/{datasets}/{split}Fold{folds}.csv
     */
    public void setBaseReadPath(File baseReadPath) {
        this.baseReadPath = Paths.get(baseReadPath.getAbsolutePath());
    }
    
    public boolean confirmMinimalInfoGivenAndValid() {
        return false;
    }
    
    
    private String buildFileName(Path baseReadPath, String classifier, String dataset, String split, int fold) { 
        return baseReadPath + classifier + "/Predictions/" + dataset + "/" + split + "Fold" + fold + ".csv";
    }
    
    public void load() throws Exception { 
        confirmMinimalInfoGivenAndValid();
        
        ErrorReport masterError = new ErrorReport("Results files not found:\n");

        allResults = new ClassifierResults[numSplits][numClassifiers][numDatasets][numFolds];
     
        //train files may be produced via TrainAccuracyEstimate, older code
        //while test files likely by experiments, but still might be a very old file
        //so having separate checks for each.
        boolean ignoringDistsFirstTime = true;
        
        for (int c = 0; c < numClassifiers; c++) {
            String classifierStorage = classifierNamesInStorage[c];
            String classifierOutput = classifierNamesInOutput[c];
            printlnDebug(classifierStorage + "(" + classifierOutput + ") reading");
            
            try {

                int totalFnfs = 0;
                ErrorReport perClassifierError = new ErrorReport("FileNotFoundExceptions thrown (### total):\n");

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

                            String fileName = buildFileName(baseReadPath, classifierStorage, datasetStorage, split, fold); 
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
                                perClassifierError.log(fileName + "\n");
                                totalFnfs++;
                            }

                            printlnDebug("\t\t\t" + split + " successfully read in");
                        }
                        printlnDebug("\t\t" + fold + " successfully read in");
                    }
                    printlnDebug("\t" + datasetStorage + "(" + datasetOutput + ") successfully read in");
                }

                perClassifierError.getLog().replace("###", totalFnfs+"");
                perClassifierError.throwIfErrors();
                printlnDebug(classifierStorage + "(" + classifierOutput + ") successfully read in");
            } catch (Exception e) {
                masterError.log("Classifier Errors: " + classifierNamesInStorage[c] + "\n" + e);
            }
        }
        
        masterError.throwIfErrors();
    }
    
}
