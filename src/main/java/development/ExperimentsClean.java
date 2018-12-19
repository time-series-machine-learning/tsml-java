/**
 * The main experimental class of the timeseriesclassification codebase. The 'main' method to run is 
 * setupAndRunSingleClassifierAndFoldTrainTestSplit(ExperimentalArguments expSettings)
 * 
 * An execution of this will evaluate a single classifier on a single resample of a single dataset. 
 * 
 * Given an ExperimentalArguments object, which may be parsed from command line arguments
 * or constructed in code, (and in the future, perhaps other methods such as JSON files etc),
 * will load the classifier and dataset specified, prep the location to write results to, 
 * train the classifier - potentially generating an error estimate via cross validation on the train set 
 * as well - and then predict the cases of the test set. 
 * 
 * The primary outputs are the train and/or 'testFoldX.csv' files, in the so-called ClassifierResults format,
 * (see the class of the same name under utilities). 
 * 
 * @author ajb (anthony.bagnall@uea.ac.uk), james (james.large@uea.ac.uk)
 */
package development;

import fileIO.OutFile;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.TreeMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import timeseriesweka.classifiers.ParameterSplittable;
import utilities.ClassifierTools;
import utilities.CrossValidator;
import utilities.InstanceTools;
import utilities.SaveParameterInfo;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.Classifier;
import utilities.ClassifierResults;
import timeseriesweka.classifiers.ensembles.SaveableEnsemble;
import utilities.multivariate_tools.MultivariateInstanceTools;
import vector_classifiers.*;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class ExperimentsClean  {

    private final static Logger LOGGER = Logger.getLogger(ExperimentsClean.class.getName());

    public static boolean testing = true;
    public static boolean debug = false;
    
    //A few 'should be final but leaving them not final just in case' public static settings 
    public static String LOXO_ATT_ID = "experimentsSplitAttribute";
    public static int numCVFolds = 10;
    public static double proportionKeptForTraining = 0.5;

    /**
     * TODO Eventually, this should be replaced/fed with something like argparse4j. This will do for now,
     * avoids adding a new dependence until we really want it
     */
    public static class ExperimentalArguments implements Runnable {
        public String dataReadLocation = null;
        public String resultsWriteLocation = null;
        public boolean generateErrorEstimateOnTrainSet = false;
        public String classifierName = null;
        public String datasetName = null;
        public int foldId = 0;
        public boolean checkpointing = false;
        public Integer singleParameterID = null;

        public ExperimentalArguments() {
            
        }
        
        public ExperimentalArguments(String[] args) throws Exception {
            parseArguments(args);
        }
        
        @Override
        public void run() {
            try {
                setupAndRunSingleClassifierAndFoldTrainTestSplit(this);
            } catch (Exception ex) {
                System.out.println("Threaded Experiment Failed: " + ex);
            }
        }
        
        /**
         * This is a bit of a bolt-on method for now. It assumes that the object on which 
         * this method is being called has all the other parameters not passed to it set already 
         * (e.g data location, results location) and these will be replicated across all experiments.
         * The current value of this.classifierName, this.datasetName, and this.foldId are ignored within 
         * this method. 
         * 
         * 
         * @param minFold inclusive
         * @param maxFold exclusive, i.e will make folds [ for (int f = minFold; f < maxFold; ++f) ]
         * @return a list of unique experimental arguments, covering all combinations of classifier, datasets, and folds passed, with the same meta info as 'this' currently stores
         */
        public List<ExperimentalArguments> generateExperiments(String[] classifierNames, String[] datasetNames, int minFold, int maxFold) { 
            
            if (minFold > maxFold) {
                int t = minFold;
                minFold = maxFold;
                maxFold = t;
            }
            
            ArrayList<ExperimentalArguments> exps = new ArrayList<>(classifierNames.length * datasetNames.length * (maxFold - minFold));
            
            for (String classifier : classifierNames) {
                for (String dataset : datasetNames) {
                    for (int fold = minFold; fold < maxFold; fold++) {
                        ExperimentalArguments exp = new ExperimentalArguments();
                        exp.classifierName = classifier;
                        exp.datasetName = dataset;
                        exp.foldId = fold;
                        
                        exp.dataReadLocation = this.dataReadLocation;
                        exp.resultsWriteLocation = this.resultsWriteLocation;
                        exp.generateErrorEstimateOnTrainSet = this.generateErrorEstimateOnTrainSet;
                        exp.checkpointing = this.checkpointing;
                        exp.singleParameterID = this.singleParameterID;
                        
                        exps.add(exp);
                    }
                }
            }
            
            return exps;
        }
        
        
        /**      
         * If on cluster (or generally from the command line), the arguments are: 
         * 
         * REQUIRED ARGUMENTS:
         * args[0]: (String)  Directory containing datasets
         *              such that args[0] + "/" + args[4] + "/" + args[4] is an arff file (see sampleDataset(...) for precise arff formats readable).
         * args[1]: (String)  Directory to write results to
         *              such that train and/or testFoldX.csv are written to args[1] + "/" + args[3] + "/Predictions/" + args[4] + "/"
         * args[2]: (boolean) Defines whether to CV on the train set to generate trainFoldX.csv files (true/false)
         * args[3]: (String)  Name of the classifier, defined in ClassifierLists
         * args[4]: (String)  Name of the dataset 
         *              such that args[0] + "/" + args[4] + "/" + args[4] is an arff file (see sampleDataset(...) for precise arff formats readable).
         * args[5]: (int)     Fold number used for resampling and rng seeds
         *              INDEXED FROM ONE, to conform to cluster job indexing. i.e foldId = Integer.parseInt(args[5]) - 1;
         * 
         * OPTIONAL ARGUMENTS:
         * args[6]: (boolean) Defines whether to checkpoint a parameter search for relevant classifiers  (true/false)
         * args[7]: (Integer) If present and not null, defines a specific unique parameter id for a parameter search to evaluate (null indicates ignore this) 
         */
        private void parseArguments(String[] args) throws Exception {
            //REQUIRED ARGUMENTS
            dataReadLocation = args[0]; // Arg0: where's the datasetName? 
            resultsWriteLocation = args[1]; // Arg1: where are we writing results to? 
            //todo, format file separators into os-independent format and add a final "/" if there isnt one
            
            generateErrorEstimateOnTrainSet = Boolean.parseBoolean(args[2]); // Arg2: shall we perform cross validation on the train set to get an error estimate? 
            
            File f = new File(resultsWriteLocation);
            if (!f.isDirectory()) {
                f.mkdirs();
                f.setWritable(true, false);
            }

            classifierName = args[3];
            datasetName = args[4];
            foldId = Integer.parseInt(args[5]) - 1;

            //OPTIONAL ARGUMENTS
            //  Arg 7:  whether to checkpoint        
            checkpointing = false;
            if (args.length >= 7)
                checkpointing = Boolean.parseBoolean(args[6]);
            
            //Arg 8: if present, do a single parameter split
            if (args.length >= 8)
                singleParameterID = Integer.parseInt(args[7]);
            else 
                singleParameterID = null;
        }
        
        public String toShortString() { 
            return "["+classifierName+","+datasetName+","+foldId+"]";
        }
        
        @Override
        public String toString() { 
            StringBuilder sb = new StringBuilder();
            
            sb.append("EXPERIMENT SETTINGS "+ this.toShortString());
            sb.append("\ndataReadLocation: ").append(dataReadLocation);
            sb.append("\nresultsWriteLocation: ").append(resultsWriteLocation);
            sb.append("\ngenerateErrorEstimateOnTrainSet: ").append(generateErrorEstimateOnTrainSet);
            sb.append("\nclassifierName: ").append(classifierName);
            sb.append("\ndatasetName: ").append(datasetName);
            sb.append("\nfoldId: ").append(foldId);
            sb.append("\ncheckpoint: ").append(checkpointing);
            sb.append("\nsingleParameterID: ").append(singleParameterID);
            
            return sb.toString();
        }        
    }
    
    /** 
     * Parses args into an ExperimentalArguments object, then calls setupAndRunSingleClassifierAndFoldTrainTestSplit(ExperimentalArguments expSettings)
     * 
     * If on cluster (or generally from the command line), the arguments are: 
     * 
     * REQUIRED ARGUMENTS:
     * args[0]: (String)  Directory containing datasets
     *              such that args[0] + "/" + args[4] + "/" + args[4] is an arff file (see sampleDataset(...) for precise arff formats readable).
     * args[1]: (String)  Directory to write results to
     *              such that train and/or testFoldX.csv are written to args[1] + "/" + args[3] + "/Predictions/" + args[4] + "/"
     * args[2]: (boolean) Defines whether to CV on the train set to generate trainFoldX.csv files (true/false)
     * args[3]: (String)  Name of the classifier, defined in ClassifierLists
     * args[4]: (String)  Name of the dataset 
     *              such that args[0] + "/" + args[4] + "/" + args[4] is an arff file (see sampleDataset(...) for precise arff formats readable).
     * args[5]: (int)     Fold number used for resampling and rng seeds
     *              INDEXED FROM ONE, to conform to cluster job indexing. i.e foldId = Integer.parseInt(args[5]) - 1;
     * 
     * OPTIONAL ARGUMENTS:
     * args[6]: (boolean) Defines whether to checkpoint a parameter search for relevant classifiers  (true/false)
     * args[7]: (Integer) If present and not null, defines a specific unique parameter id for a parameter search to evaluate (null indicates ignore this) 
     * 
     * If running locally, easier to build the ExperimentalArguments object yourself and call the above method, 
     * instead of building the String[] args and calling main like a lot of legacy code does.
     * 
     */
    public static void main(String[] args) throws Exception {
        if (testing) {
            test_correctnessTestsComparingOldExpsAndNew();
            return;
        }
        
        //even if all else fails, print the args as a sanity check for cluster.
        for (String str : args)
            System.out.println(str);
        
        if (args.length > 0) {
            ExperimentalArguments expSettings = new ExperimentalArguments(args);
            setupAndRunSingleClassifierAndFoldTrainTestSplit(expSettings);
        }
        
    }

    /**
     * Runs an experiment with the given settings. For the more direct method in case e.g 
     * you have a bespoke classifier not handled by ClassifierList or dataset that 
     * is sampled in a bespoke way, use singleClassifierAndFoldTrainTestSplit
     * 
     * 1) Sets up the logger. 
     * 2) Sets up the results write path
     * 3) Checks whether this experiments results already exist. If so, exit
     * 4) Constructs the classifier
     * 5) Samples the dataset.
     * 6) If we're good to go, starts the experiment.
     */
    public static void setupAndRunSingleClassifierAndFoldTrainTestSplit(ExperimentalArguments expSettings) throws Exception {
        LOGGER.addHandler(new ConsoleHandler());
        //todo: when we convert to e.g argparse4j for parameter passing, add a para 
        //for location to log to file as well. for now, assuming console output is good enough
        //for local running, and cluster output files are good enough on there. 
        if (debug)
            LOGGER.setLevel(Level.FINE);
        else 
            LOGGER.setLevel(Level.SEVERE);
        LOGGER.log(Level.INFO, expSettings.toString());
        
        //TODO still setting these for now, since maybe certain classfiiers still use these "global" 
        //paths. would rather just use the expSettings to do it all though 
        DataSets.resultsPath = expSettings.resultsWriteLocation;
        DataSets.problemPath = expSettings.dataReadLocation;
        
        String fullWriteLocation = expSettings.resultsWriteLocation + expSettings.classifierName + "/Predictions/" + expSettings.datasetName;
        File f = new File(fullWriteLocation);
        if (!f.exists()) {
            f.mkdirs();
        }

        String targetFileName = fullWriteLocation + "/testFold" + expSettings.foldId + ".csv";
        
        //Check whether fold already exists, if so, dont do it, just quit
        if (CollateResults.validateSingleFoldFile(targetFileName)) {
            LOGGER.log(Level.INFO, expSettings.toShortString() + " already exists at "+targetFileName+", exiting.");
            return;
        }
        else {           
            Classifier classifier = ClassifierLists.setClassifierClassic(expSettings.classifierName, expSettings.foldId);
            Instances[] data = sampleDataset(expSettings.dataReadLocation, expSettings.datasetName, expSettings.foldId);
            
            //If this is to be a single _parameter_ evaluation of a fold, check whether this exists, and again quit if it does.
            if (expSettings.singleParameterID != null && classifier instanceof ParameterSplittable) {
                expSettings.checkpointing = false; //Just to tie up loose ends in case user defines both checkpointing AND para splitting
                
                targetFileName = fullWriteLocation + "/fold" + expSettings.foldId + "_" + expSettings.singleParameterID + ".csv";
                if (CollateResults.validateSingleFoldFile(targetFileName)) { 
                    LOGGER.log(Level.INFO, expSettings.toShortString() + ", parameter " + expSettings.singleParameterID +", already exists at "+targetFileName+", exiting.");
                    return;
                }
            }

            double acc = singleClassifierAndFoldTrainTestSplit(expSettings, data[0], data[1], classifier, fullWriteLocation);
            LOGGER.log(Level.INFO, "Experiment finished " + expSettings.toShortString() + ", Test Acc:" + acc);
            System.out.println("Classifier="+expSettings.classifierName+", Problem="+expSettings.datasetName+", Fold="+expSettings.foldId+", Test Acc,"+acc);
        }
    }
    
    /**
     * This method will return a train/test split of the problem, resampled with the fold ID given. 
     * 
     * Currently, there are four ways to load datasets. These will be attempted from 
     * top to bottom, in an order designed to make the fewest assumptions 
     * possible about the nature of the split, in terms of potential differences in class distributions,
     * train and test set sizes, etc. 
     * 
     * 1) if predefined splits are found at the specified location, in the form dataLocation/dsetName/dsetName0_TRAIN and TEST,
     *      these will be loaded and used as they are, OTHERWISE...
     * 2) if a predefined fold0 split is given as in the UCR archive, and fold0 is being experimented on, the split exactly as it is defined will be used. 
     *      For fold != 0, the fold0 split is combined and resampled, maintaining the original train and test distributions. OTHERWISE...
     * 3) if only a single file is found containing all the data, this dataset is  stratified randomly resampled with proportionKeptForTraining (default=0.5)
     *      instances reserved for the _TRAIN_ set. OTHERWISE...
     * 4) if the dataset loaded has a first attribute whose name _contains_ the string "experimentsSplitAttribute".toLowerCase() 
     *      then it will be assumed that we want to perform a leave out one X cross validation. Instances are sampled such that fold N is comprised of 
     *      a test set with all instances with first-attribute equal to the Nth unique value in a sorted list of first-attributes. The train
     *      set would be all other instances. The first attribute would then be removed from all instances, so that they are not given
     *      to the classifier to potentially learn from. It is up to the user to ensure the the foldID requested is within the range of possible 
     *      values 1 to numUniqueFirstAttValues OTHERWISE...
     * 5) error
     * 
     * TODO: potentially just move to development.DataSets once we clean up that
     * 
     * @return new Instances[] { trainSet, testSet };
     */
    public static Instances[] sampleDataset(String parentFolder, String problem, int fold) throws Exception {
        Instances[] data = new Instances[2];

        File trainFile = new File(parentFolder + problem + "/" + problem + fold + "_TRAIN.arff");
        File testFile = new File(parentFolder + problem + "/" + problem + fold + "_TEST.arff");
        
        boolean predefinedSplitsExist = (trainFile.exists() && testFile.exists());
        if (predefinedSplitsExist) {
            // CASE 1) 
            data[0] = ClassifierTools.loadData(trainFile);
            data[1] = ClassifierTools.loadData(testFile);
            LOGGER.log(Level.INFO, problem + " loaded from predfined folds.");
        } else {   
            trainFile = new File(parentFolder + problem + "/" + problem + "_TRAIN.arff");
            testFile = new File(parentFolder + problem + "/" + problem + "_TEST.arff");
            boolean predefinedFold0Exists = (trainFile.exists() && testFile.exists());
            if (predefinedFold0Exists) {
                // CASE 2) 
                data[0] = ClassifierTools.loadData(trainFile);
                data[1] = ClassifierTools.loadData(testFile);
                if (data[0].checkForAttributeType(Attribute.RELATIONAL))
                    data = MultivariateInstanceTools.resampleMultivariateTrainAndTestInstances(data[0], data[1], fold);
                else
                    data = InstanceTools.resampleTrainAndTestInstances(data[0], data[1], fold);
                
                LOGGER.log(Level.INFO, problem + " resampled from predfined fold0 split.");
            }
            else { 
                // We only have a single file with all the data
                Instances all = null;
                try {
                    all = ClassifierTools.loadDataThrowable(parentFolder + problem + "/" + problem);
                } catch (IOException io) {
                    String msg = "Could not find the dataset \"" + problem + "\" in any form at the path\n"+
                            parentFolder+"\n"
                            +"The IOException: " + io;
                    LOGGER.log(Level.SEVERE, msg, io);
                }
                 
                
                boolean needToDefineLeaveOutOneXFold = all.attribute(0).name().toLowerCase().contains(LOXO_ATT_ID.toLowerCase());
                if (needToDefineLeaveOutOneXFold) {
                    // CASE 4)
                    data = splitDatasetByFirstAttribute(all, fold);
                    LOGGER.log(Level.INFO, problem + " resampled from full data file.");
                }
                else { 
                    // CASE 3) 
                    if (all.checkForAttributeType(Attribute.RELATIONAL))
                        data = MultivariateInstanceTools.resampleMultivariateInstances(all, fold, proportionKeptForTraining);
                    else
                        data = InstanceTools.resampleInstances(all, fold, proportionKeptForTraining);
                    LOGGER.log(Level.INFO, problem + " resampled from full data file.");
                }
            }
        }
        return data;
    }

    /**
     * If the dataset loaded has a first attribute whose name _contains_ the string "experimentsSplitAttribute".toLowerCase() 
     * then it will be assumed that we want to perform a leave out one X cross validation. Instances are sampled such that fold N is comprised of 
     * a test set with all instances with first-attribute equal to the Nth unique value in a sorted list of first-attributes. The train
     * set would be all other instances. The first attribute would then be removed from all instances, so that they are not given
     * to the classifier to potentially learn from. It is up to the user to ensure the the foldID requested is within the range of possible 
     * values 1 to numUniqueFirstAttValues
     * 
     * TODO: potentially just move to development.DataSets once we clean up that
     * 
     * @return new Instances[] { trainSet, testSet };
     */
    public static Instances[] splitDatasetByFirstAttribute(Instances all, int foldId) {        
        TreeMap<Double, Integer> splitVariables = new TreeMap<>();
        for (int i = 0; i < all.numInstances(); i++) {
            //even if it's a string attribute, this val corresponds to the index into the array of possible strings for this att
            double key= all.instance(i).value(0);
            Integer val = splitVariables.get(key);
            if (val == null)
                val = 0;
            splitVariables.put(key, ++val); 
        }

        //find the split attribute value to keep for testing this fold
        double idToReserveForTestSet = -1;
        int testSize = -1;
        int c = 0;
        for (Map.Entry<Double, Integer> splitVariable : splitVariables.entrySet()) {
            if (c++ == foldId) {
                idToReserveForTestSet = splitVariable.getKey();
                testSize = splitVariable.getValue();
            }
        }

        //make the split
        Instances train = new Instances(all, all.size() - testSize);
        Instances test  = new Instances(all, testSize);
        for (int i = 0; i < all.numInstances(); i++)
            if (all.instance(i).value(0) == idToReserveForTestSet)
                test.add(all.instance(i));
        train.addAll(all);

        //delete the split attribute
        train.deleteAttributeAt(0);
        test.deleteAttributeAt(0);
        
        return new Instances[] { train, test };
    }
    
        
    /**
     * Perform an actual experiment, using the loaded classifier and resampled dataset given, writing to the specified results location. 
     * 
     * 1) If needed, set up file paths and flags related to a single parameter evaluation and/or the classifier's internal parameter saving things
     * 2) If we want to be performing cv to find an estimate of the error on the train set, either do that here or set up the classifier to do it internally 
     *          during buildClassifier()
     * 3) Do the actual training, i.e buildClassifier()
     * 4) Save any train cv results
     * 5) Evaluate on the test set
     * 6) Save test results
     * 7) Done
     * 
     * NOTES: 1. If the classifier is a SaveableEnsemble, then we save the
     * internal cross validation accuracy and the internal test predictions 2.
     * The output of the file testFold+fold+.csv is Line 1:
     * ProblemName,ClassifierName, train/test Line 2: parameter information for
     * final classifierName, if it is available Line 3: test accuracy then each line
     * is Actual Class, Predicted Class, Class probabilities
     * 
     * @param resultsPath The exact folder in which to write the train and/or testFoldX.csv files
     * @return the accuracy of c on fold for problem given in train/test
     */
    public static double singleClassifierAndFoldTrainTestSplit(ExperimentalArguments expSettings, Instances trainSet, Instances testSet, Classifier classifier, String resultsPath) {
        
        //if this is a parameter split run, train file name is defined by this
        //otherwise generally if the classifier wants to save parameter info itnerally, set that up here too
        String trainFoldFilename = setupParameterSavingInfo(expSettings, classifier, trainSet, resultsPath);
        if (trainFoldFilename == null) 
            //otherwise, defined by this as default
            trainFoldFilename = "/trainFold" + expSettings.foldId + ".csv";
        String testFoldFilename = "/testFold" + expSettings.foldId + ".csv";       
        
        ClassifierResults trainResults = null;
        ClassifierResults testResults = null;
        
        LOGGER.log(Level.INFO, "Preamble complete, real experiment starting.");
        
        try {
            //Setup train results
            if (expSettings.generateErrorEstimateOnTrainSet) 
                trainResults = findOrSetUpTrainEstimate(classifier, trainSet, expSettings.foldId, resultsPath + trainFoldFilename);
            LOGGER.log(Level.INFO, "Train estimate ready.");


            //Build on the full train data here
            long buildTime = System.currentTimeMillis();
            classifier.buildClassifier(trainSet);
            buildTime = System.currentTimeMillis() - buildTime;
            LOGGER.log(Level.INFO, "Training complete");


            //Write train results
            if (expSettings.generateErrorEstimateOnTrainSet) 
                writeTrainEstimate(classifier, trainSet, trainResults, resultsPath + trainFoldFilename, buildTime);
            LOGGER.log(Level.INFO, "Train estimate written");


            //And now evaluate on the test set, if this wasn't a single parameter fold
            if (expSettings.singleParameterID == null) {
                //This is checked before the buildClassifier also, but 
                //a) another process may have been doign the same experiment 
                //b) we have a special case for the file builder that copies the results over in buildClassifier (apparently?)
                //No harm in checking again!
                if (!CollateResults.validateSingleFoldFile(resultsPath + testFoldFilename)) {
                    testResults = evaluateClassifier(classifier, testSet);
                    LOGGER.log(Level.INFO, "Testing complete");
                    writeTestResults(classifier, testSet, testResults, resultsPath + testFoldFilename);
                    LOGGER.log(Level.INFO, "Testing written");
                } 
                else {
                    LOGGER.log(Level.INFO, "Test file already found, written by another process.");
                    testResults = new ClassifierResults(resultsPath + testFoldFilename);
                }
                return testResults.acc;
            } 
            else {
                return 0; //not error, but we dont have a test acc. just returning 0 for now
            }
        } 
        catch (Exception e) {
            //todo expand..
            LOGGER.log(Level.SEVERE, "Experiment failed. Settings: " + expSettings + "\n\nERROR: " + e.toString(), e);
            return -1; //error state
        }
    }
    
    private static String setupParameterSavingInfo(ExperimentalArguments expSettings, Classifier classifier, Instances train, String resultsPath) { 
        String parameterFileName = null;
        if (expSettings.singleParameterID != null && classifier instanceof ParameterSplittable)//Single parameter fold
        {
            if (classifier instanceof TunedRandomForest)
                ((TunedRandomForest) classifier).setNumFeaturesInProblem(train.numAttributes() - 1);
            
            expSettings.checkpointing = false;
            ((ParameterSplittable) classifier).setParametersFromIndex(expSettings.singleParameterID);
            parameterFileName = "/fold" + expSettings.foldId + "_" + expSettings.singleParameterID + ".csv";
            expSettings.generateErrorEstimateOnTrainSet = true;
        } 
        else {
            //Only do all this if not an internal _single parameter_ experiment
            // Save internal info for ensembles
            if (classifier instanceof SaveableEnsemble) {
                ((SaveableEnsemble) classifier).saveResults(resultsPath + "/internalCV_" + expSettings.foldId + ".csv", resultsPath + "/internalTestPreds_" + expSettings.foldId + ".csv");
            }
            if (expSettings.checkpointing && classifier instanceof SaveEachParameter) {
                ((SaveEachParameter) classifier).setPathToSaveParameters(resultsPath + "/fold" + expSettings.foldId + "_");
            }
        }
        
        return parameterFileName;
    }
    
    private static ClassifierResults findOrSetUpTrainEstimate(Classifier classifier, Instances train, int fold, String fullTrainWritingPath) throws Exception { 
        ClassifierResults trainResults = null;
        
        if (classifier instanceof TrainAccuracyEstimate) { 
            //Classifier will perform cv internally while building, probably as part of a parameter search
            ((TrainAccuracyEstimate) classifier).writeCVTrainToFile(fullTrainWritingPath);
            File f = new File(fullTrainWritingPath);
            if (f.exists())
                f.setWritable(true, false);
        } 
        else { 
            // Need to cross validate separately here
            if (classifier instanceof RiseV2 && ((RiseV2) classifier).getBuildFromSavedData()) {
                //TODO Write some internal crossvalidation that can deal with read from files.
            } 
            else {
                CrossValidator cv = new CrossValidator();
                cv.setSeed(fold);
                int numFolds = Math.min(train.numInstances(), numCVFolds);
                cv.setNumFolds(numFolds);
                trainResults = cv.crossValidateWithStats(classifier, train);
            }
        }
        
        return trainResults;
    }
    
    public static void writeTrainEstimate(Classifier classifier, Instances train, ClassifierResults results, String fullTrainWritingPath, long buildTime) {
        if (classifier instanceof TrainAccuracyEstimate) {
            //the classifier will have written it's own train estimate internally
            return;
        }
        else {
            //Write the results
            OutFile trainOut = new OutFile(fullTrainWritingPath);
            trainOut.writeLine(train.relationName() + "," + classifier.getClass().getName() + ",train");
            if (classifier instanceof SaveParameterInfo) {
                trainOut.writeLine(((SaveParameterInfo) classifier).getParameters()); //assumes build time is in it's param info, is for tunedsvm
            } else {
                trainOut.writeLine("BuildTime," + buildTime + ",No Parameter Info");
            }
            trainOut.writeLine(results.acc + "");
            trainOut.writeLine(results.writeInstancePredictions());
            //not simply calling trainResults.writeResultsFileToString() since it looks like those that extend SaveParameterInfo will store buildtimes
            //as part of their params, and so would be written twice
            trainOut.closeFile();
            
            File f = new File(fullTrainWritingPath);
            if (f.exists()) {
                f.setWritable(true, false);
            }

        }
    }
    
    public static ClassifierResults evaluateClassifier(Classifier classifier, Instances testSet) throws Exception {
        ClassifierResults results = new ClassifierResults(testSet.numClasses());
        double[] trueClassValues = testSet.attributeToDoubleArray(testSet.classIndex());

        for (Instance instance : testSet) {
            instance.setClassMissing(); //just to be sure of no funny business 
            double[] probs = classifier.distributionForInstance(instance);
            results.storeSingleResult(probs);
        }
        
        results.finaliseResults(trueClassValues);
        return results;
    }
    
    public static void writeTestResults(Classifier classifier, Instances dataset, ClassifierResults results, String fullTestWritingPath) {
        OutFile testOut = new OutFile(fullTestWritingPath);
        testOut.writeLine(dataset.relationName() + "," + classifier.getClass().getName() + ",test");
        if (classifier instanceof SaveParameterInfo) {
            testOut.writeLine(((SaveParameterInfo) classifier).getParameters());
        } else {
            testOut.writeLine("No parameter info");
        }
        testOut.writeLine(results.acc + "");
        testOut.writeString(results.writeInstancePredictions());
        testOut.closeFile();
        File f = new File(fullTestWritingPath);
        if (f.exists()) {
            f.setWritable(true, false);
        }
    }
    
    /**
     * Will run through all combinations of classifiers*datasets*folds provided, using the meta experimental info stored in the 
     * standardArgs. Will by default set numThreads = numCores
     */
    public static void setupAndRunMultipleExperimentsThreaded(ExperimentalArguments standardArgs, String[] classifierNames, String[] datasetNames, int minFolds, int maxFolds) throws Exception{
        setupAndRunMultipleExperimentsThreaded(standardArgs, classifierNames, datasetNames, minFolds, maxFolds, 0);
    }
    
    /**
     * Will run through all combinations of classifiers*datasets*folds provided, using the meta experimental info stored in the 
     * standardArgs. If numThreads > 0, will spawn that many threads. If numThreads == 0, will use as many threads as there are cores, 
     * else if numThreads == -1, will spawn as many threads as there are cores minus 1, to aid usability of the machine. 
     */
    public static void setupAndRunMultipleExperimentsThreaded(ExperimentalArguments standardArgs, String[] classifierNames, String[] datasetNames, int minFolds, int maxFolds, int numThreads) throws Exception{
        int numCores = Runtime.getRuntime().availableProcessors();        
        if (numThreads == 0)
            numThreads = numCores;
        else if (numThreads < 0)
            numThreads = Math.max(1, numCores-1);
        
        System.out.println("# cores ="+numCores);
        System.out.println("# threads ="+numThreads);
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        
        List<ExperimentalArguments> exps = standardArgs.generateExperiments(classifierNames, datasetNames, minFolds, maxFolds);
        for (ExperimentalArguments exp : exps)
            executor.execute(exp);
        
        executor.shutdown();
        while (!executor.isTerminated()) {
        }
        System.out.println("Finished all threads");            
    }

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    public static void test_correctnessTestsComparingOldExpsAndNew() throws Exception {
        ExperimentsClean.testing = false;
        
        String baseResLoc = "C:/Temp/ExpsCorrectnessTests/";
        
        String[] classifiers = {
            "C45", //simple, pure weka
            "RandF", //untuned tsc wrapper of weka classifier
            "DTW", //pure tsc
            "TunedSVMLinear17", // implements TrainAccuracyEstimate,SaveEachParameter,ParameterSplittable
            "HESCA", // varied ensemble, has plenty of it's own shit going on internally
        };
        
        String separator = "____";
        
        String[] datasets = {
            "Z:/Data/UCIDelgado/"+separator+"hayes-roth", //single file for data
            "Z:/Data/TSCProblems/"+separator+"ItalyPowerDemand",//fold0 defined
            "Z:/Data/TSCProblems2017_Folds/"+separator+"SonyAIBORobotSurface1",//all folds predefined
        };
        
        Method[] mains = { 
            Experiments.class.getMethod("main", String[].class), 
            ExperimentsClean.class.getMethod("main", String[].class) 
        };
        
        String[] mainNames = { "ExperimentsClassic", "ExperimentsClean" };
        
        
        //for a bunch of different experimental parameters
        for (String d : datasets) {
            String[] split = d.split(separator);
            String dataLoc = split[0];
            String dataset = split[1];
                
            for (String classifier : classifiers) {
                for (boolean genTrainFiles : new boolean[] { false, true }) {
                    for (boolean checkpoint : new boolean[] { false, true }) {
                        //just doing fold 0 and 1 resample, to test the resampling methods
                        for (int fold = 1; fold < 3; fold++) {
                            String[] trainFileNames = new String[mains.length];
                            String[] testFileNames = new String[mains.length];
                            String[] trainRes = new String[mains.length];
                            String[] testRes = new String[mains.length];
                            
                            //run the different mains/collect results
                            for (int mainInd = 0; mainInd < mains.length; mainInd++) {
                                String resultLoc = baseResLoc + mainNames[mainInd] + "/Train=" + genTrainFiles + "/Checkpoint=" + checkpoint + "/";
                                System.out.println(mainNames[mainInd]);
                                mains[mainInd].invoke(null, (Object) new String[] { dataLoc, resultLoc, genTrainFiles+"", classifier, dataset, fold+"", checkpoint+"" });
                                System.out.println("\n\n");
                                
                                if (genTrainFiles) {
                                    trainFileNames[mainInd] = resultLoc + classifier + "/Predictions/" + dataset + "/trainFold" + (fold-1) + ".csv";
                                    StringBuilder sb = new StringBuilder();
                                    Scanner in = new Scanner(new File(trainFileNames[mainInd]));
                                    while (in.hasNext())
                                        sb.append(in.next());
                                    trainRes[mainInd] = sb.toString();
                                }
                                
                                testFileNames[mainInd] = resultLoc + classifier + "/Predictions/" + dataset + "/testFold" + (fold-1) + ".csv";
                                StringBuilder sb = new StringBuilder();
                                Scanner in = new Scanner(new File(testFileNames[mainInd]));
                                while (in.hasNext())
                                    sb.append(in.next());
                                testRes[mainInd] = sb.toString();
                            }
                            
                            //compare results for equality
                            for (int i = 1; i < mains.length; i++) {
                                //should all be the same, so jsut compare all to the first
                                if (genTrainFiles && !trainRes[0].equals(testRes[i])) {
                                    System.out.println("*******Difference in train files:*******");
                                    System.out.println("\t"+trainFileNames[0]);
                                    System.out.println("\t"+trainFileNames[i]);
                                }
                                else { 
                                    
                                }
                                
                                if (!testRes[0].equals(testRes[i])) {    
                                    System.out.println("*******Difference in test files:******");
                                    System.out.println("\t"+testFileNames[0]);
                                    System.out.println("\t"+testFileNames[i]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }    
    
}
