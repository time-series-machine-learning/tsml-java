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
package experiments;

import com.google.common.testing.GcFinalization;
import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.SingleSampleEvaluator;
import evaluation.evaluators.SingleTestSetEvaluator;
import evaluation.evaluators.StratifiedResamplesEvaluator;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import machine_learning.classifiers.SaveEachParameter;
import machine_learning.classifiers.ensembles.SaveableEnsemble;
import machine_learning.classifiers.tuned.TunedRandomForest;
import tsml.classifiers.*;
import tsml.classifiers.early_classification.AbstractEarlyClassifier;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;

import static utilities.GenericTools.indexOfMax;
import static utilities.InstanceTools.truncateInstance;
import static utilities.InstanceTools.truncateInstances;

/**
 * The main experimental class of the timeseriesclassification codebase. The 'main' method to run is
 setupAndRunExperiment(ExperimentalArguments expSettings)

 An execution of this will evaluate a single classifier on a single resample of a single dataset.

 Given an ExperimentalArguments object, which may be parsed from command line arguments
 or constructed in code, (and in the future, perhaps other methods such as JSON files etc),
 will load the classifier and dataset specified, prep the location to write results to,
 train the classifier - potentially generating an error estimate via cross validation on the train set
 as well - and then predict the cases of the test set.

 The primary outputs are the train and/or 'testFoldX.csv' files, in the so-called ClassifierResults format,
 (see the class of the same name under utilities).
*
 * @author James Large (james.large@uea.ac.uk), Tony Bagnall (anthony.bagnall@uea.ac.uk)
 */
public class ClassifierExperiments {

    private final static Logger LOGGER = Logger.getLogger(ClassifierExperiments.class.getName());

    public static boolean debug = false;

    private static boolean testFoldExists;
    private static boolean trainFoldExists;

    /**
     * If true, experiments will not print or log to stdout/err anything other that exceptions (SEVERE)
     */
    public static boolean beQuiet = false;

    //A few 'should be final but leaving them not final just in case' public static settings
    public static int numCVFolds = 10;

    private static String WORKSPACE_DIR = "Workspace";
    private static String PREDICTIONS_DIR = "Predictions";

    /**
     * Parses args into an ExperimentalArguments object, then calls setupAndRunExperiment(ExperimentalArguments expSettings).
     * Calling with the --help argument, or calling with un-parsable parameters, will print a summary of the possible parameters.

     Argument key-value pairs are separated by '='. The 5 basic, always required, arguments are:
     Para name (short/long)  |    Example
     -dp --dataPath          |    --dataPath=C:/Datasets/
     -rp --resultsPath       |    --resultsPath=C:/Results/
     -cn --classifierName    |    --classifierName=RandF
     -dn --datasetName       |    --datasetName=ItalyPowerDemand
     -f  --fold              |    --fold=1

     Use --help to see all the optional parameters, and more information about each of them.

     If running locally, it may be easier to build the ExperimentalArguments object yourself and call setupAndRunExperiment(...)
     directly, instead of building the String[] args and calling main like a lot of legacy code does.
     */
    public static void main(String[] args) throws Exception {
        //even if all else fails, print the args as a sanity check for cluster.

        if (args.length > 0) {
            ExperimentalArguments expSettings = new ExperimentalArguments(args);
            ClassifierExperiments.setupAndRunExperiment(expSettings);
        }
        else {//Manually set args
            int start=1;
            int folds = 1;

            /*
             * Change these settings for your experiment:
             */
            //Experiment Parameters, see

            String[] classifier = {"1NN-DTW"};//"Arsenal", "TDE","DrCIF","RotF",Classifier name: See ClassifierLists for valid options
            ArrayList<String> parameters = new ArrayList<>();
            parameters.add("-dp=src\\main\\java\\experiments\\data\\tsc\\"); //Where to get datasets
            parameters.add("-rp=temp\\"); //Where to write results
            parameters.add("-gtf=true"); //Whether to generate train files or not
            parameters.add("-cn=" + classifier[0]); //Classifier name
            parameters.add("-dn="); //Problem name, don't change here as it is overwritten by probFiles
            parameters.add("-f=1"); //Fold number (fold number 1 is stored as testFold0.csv, its a cluster thing)
            parameters.add("-d=true"); //Debugging
            parameters.add("-ctr=1h"); //Whether to generate train files or not
            parameters.add("--force=true"); //Overwrites existing results if true, otherwise set to false
            //            parameters.add("-ctr=3m"); //contract time, default in hours
            String[] probFiles ={"UnitTest"};


            String[] settings = new String[parameters.size()];
            int count = 0;
            for (String str : parameters)
                settings[count++] = str;


            //            String[] probFiles= univariate; //Problem name(s)
            //            String[] probFiles= univariate; //{"ArrowHead"}; //Problem name(s)
//           String[] probFiles= {"ChinaTown"}; //Problem name(s)
            //           String[] probFiles = DatasetLists.equalLengthProblems;
            //            String[] probFiles= DatasetLists.fixedLengthMultivariate;
            /*
             * END OF SETTINGS
             */
            System.out.println("Manually set args:");
            for (String str : settings)
                System.out.println("\t" + str);
            System.out.println("");

            boolean threaded = false;
            if (threaded) {
                ExperimentalArguments expSettings = new ExperimentalArguments(settings);
                System.out.println("Threaded experiment with " + expSettings);
                //             setupAndRunMultipleExperimentsThreaded(expSettings, classifier,probFiles,0,folds);
                ClassifierExperiments.setupAndRunMultipleExperimentsThreaded(expSettings, classifier, null, probFiles, 0, folds);
            } else {//Local run without args, mainly for debugging
                for (String prob : probFiles) {
                    settings[4] = "-dn=" + prob;
                    for (int i = start; i <= folds; i++) {
                        settings[5] = "-f=" + i;
                        ExperimentalArguments expSettings = new ExperimentalArguments(settings);
                        //                      System.out.println("Sequential experiment with "+expSettings);
                        ClassifierExperiments.setupAndRunExperiment(expSettings);
                    }
                }
            }
        }
    }

    /**
     * Runs an experiment with the given settings. For the more direct method in case e.g
     * you have a bespoke classifier not handled by ClassifierList or dataset that
     * is sampled in a bespoke way, use runExperiment
     *
     * 1) Sets up the logger.
     * 2) Sets up the results write path
     * 3) Checks whether this experiments results already exist. If so, exit
     * 4) Constructs the classifier
     * 5) Samples the dataset.
     * 6) If we're good to go, runs the experiment.
     */
    public static ClassifierResults[] setupAndRunExperiment(ExperimentalArguments expSettings) throws Exception {
        if (beQuiet)
            LOGGER.setLevel(Level.SEVERE); // only print severe things
        else {
            if (debug) LOGGER.setLevel(Level.FINEST); // print everything
            else       LOGGER.setLevel(Level.INFO); // print warnings, useful info etc, but not simple progress messages, e.g. 'training started'

            DatasetLoading.setDebug(debug); //TODO when we go full enterprise and figure out how to properly do logging, clean this up
        }
        LOGGER.log(Level.FINE, expSettings.toString());

        // if a pre-instantiated classifier instance hasn't been supplied, generate one here
        if (expSettings.classifier == null) {
            // if a classifier-generating-function has been given (typically in the case of bespoke classifiers wanted in threaded exps),
            // instantiate the classifier from that
            if (expSettings.classifierGenerator != null)
                expSettings.classifier = expSettings.classifierGenerator.get();
            else {
                // else, use the classic setClassifier

                // Cases in the classifierlist can now change the classifier name to reflect particular parameters wanting to be
                // represented as different classifiers, e.g. ST_1day, ST_2day
                // The set classifier call is therefore made before defining paths that are dependent on the classifier name
                expSettings.classifier = ClassifierLists.setClassifier(expSettings);
            }
        }

        buildExperimentDirectoriesAndFilenames(expSettings, expSettings.classifier);
        //Check whether results already exists, if so and force evaluation is false: just quit
        if (quitEarlyDueToResultsExistence(expSettings))
            return null;

        Instances[] data = DatasetLoading.sampleDataset(expSettings.dataReadLocation, expSettings.datasetName, expSettings.foldId);
        setupClassifierExperimentalOptions(expSettings, expSettings.classifier, data[0]);
        ClassifierResults[] results = runExperiment(expSettings, data[0], data[1], expSettings.classifier);
        LOGGER.log(Level.INFO, "Experiment finished " + expSettings.toShortString() + ", Test Acc:" + results[1].getAcc());

        return results;
    }

    /**
     * Perform an actual experiment, using the loaded classifier and resampled dataset given, writing to the specified results location.
     *
     * 1) If needed, set up file paths and flags related to a single parameter evaluation and/or the classifier's internal parameter saving things
     * 2) If we want to be performing cv to find an estimate of the error on the train set, either do that here or set up the classifier to do it internally
     *          during buildClassifier()
     * 3) Do the actual training, i.e buildClassifier()
     * 4) Save information needed from the training, e.g. train estimates, serialising the classifier, etc.
     * 5) Evaluate the trained classifier on the test set
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
     * @return the classifierresults for this experiment, {train, test}
     */
    public static ClassifierResults[] runExperiment(ExperimentalArguments expSettings, Instances trainSet, Instances testSet, Classifier classifier) {
        ClassifierResults[] experimentResults = null; // the combined container, to hold { trainResults, testResults } on return

        LOGGER.log(Level.FINE, "Preamble complete, real experiment starting.");

        try {
            ClassifierResults trainResults = training(expSettings, classifier, trainSet);
            postTrainingOperations(expSettings, classifier);
            ClassifierResults testResults = testing(expSettings, classifier, testSet, trainResults);

            experimentResults = new ClassifierResults[] {trainResults, testResults};
        }
        catch (Exception e) {
            //todo expand..
            LOGGER.log(Level.SEVERE, "Experiment failed. Settings: " + expSettings + "\n\nERROR: " + e.toString(), e);
            e.printStackTrace();
            return null; //error state
        }

        return experimentResults;
    }








    /**
     * Performs all operations related to training the classifier, and returns a ClassifierResults object holding the results
     * of training.
     *
     * At minimum these results hold the hardware benchmark timing (if requested in expSettings), the memory used,
     * and the build time.
     *
     * If a train estimate is to be generated, the results also hold predictions and results from the train set, and these
     * results are written to file.
     */
    public static ClassifierResults training(ExperimentalArguments expSettings, Classifier classifier, Instances trainSet) throws Exception {
        ClassifierResults trainResults = new ClassifierResults();

        long benchmark = findBenchmarkTime(expSettings);

        MemoryMonitor memoryMonitor = new MemoryMonitor();
        memoryMonitor.installMonitor();

        if (expSettings.generateErrorEstimateOnTrainSet && (!trainFoldExists || expSettings.forceEvaluation || expSettings.forceEvaluationTrainFold)) {
            //Tell the classifier to generate train results if it can do it internally,
            //otherwise perform the evaluation externally here (e.g. cross validation on the
            //train data
            if (EnhancedAbstractClassifier.classifierAbleToEstimateOwnPerformance(classifier))
                ((EnhancedAbstractClassifier) classifier).setEstimateOwnPerformance(true);
            else
                trainResults = findExternalTrainEstimate(expSettings, classifier, trainSet, expSettings.foldId);
        }
        LOGGER.log(Level.FINE, "Train estimate ready.");

        //Build on the full train data here
        long buildTime = System.nanoTime();
        classifier.buildClassifier(trainSet);
        buildTime = System.nanoTime() - buildTime;
        LOGGER.log(Level.FINE, "Training complete");

        // Training done, collect memory monitor results
        // Need to wait for an update, otherwise very quick classifiers may not experience gc calls during training,
        // or the monitor may not update in time before collecting the max
        GcFinalization.awaitFullGc();
        long maxMemory = memoryMonitor.getMaxMemoryUsed();

        trainResults = finaliseTrainResults(expSettings, classifier, trainResults, buildTime, benchmark, TimeUnit.NANOSECONDS, maxMemory);

        //At this stage, regardless of whether the classifier is able to estimate it's
        //own accuracy or not, train results should contain either
        //    a) timings, if expSettings.generateErrorEstimateOnTrainSet == false
        //    b) full predictions, if expSettings.generateErrorEstimateOnTrainSet == true

        if (expSettings.generateErrorEstimateOnTrainSet && (!trainFoldExists || expSettings.forceEvaluation || expSettings.forceEvaluationTrainFold)) {
            writeResults(expSettings, trainResults, expSettings.trainFoldFileName, "train");
            LOGGER.log(Level.FINE, "Train estimate written");
        }

        return trainResults;
    }

    /**
     * Any operations aside from testing that we want to perform on the trained classifier. Performed after training, but before testing,
     * with exceptions caught and only severe warning logged instead of program failure; completion of testing is preferred instead
     * requiring retraining in a future execution
     */
    public static void postTrainingOperations(ExperimentalArguments expSettings, Classifier classifier)  {
        if (expSettings.serialiseTrainedClassifier) {
            if (classifier instanceof Serializable) {
                try {
                    serialiseClassifier(expSettings, classifier);
                } catch (Exception ex) {
                    LOGGER.log(Level.SEVERE, "Serialisation attempted but failed for classifier ("+classifier.getClass().getName()+")", ex);
                }
            }
            else
                LOGGER.log(Level.WARNING, "Serialisation requested, but the classifier ("+classifier.getClass().getName()+") does not extend Serializable.");
        }

        if (expSettings.visualise) {
            if (classifier instanceof Visualisable) {
                ((Visualisable) classifier).setVisualisationSavePath(expSettings.supportingFilePath);

                try {
                    ((Visualisable) classifier).createVisualisation();
                } catch (Exception ex) {
                    LOGGER.log(Level.SEVERE, "Visualisation attempted but failed for classifier ("+classifier.getClass().getName()+")", ex);
                }
            }
            else {
                expSettings.visualise = false;
                LOGGER.log(Level.WARNING, "Visualisation requested, but the classifier (" + classifier.getClass().getName() + ") does not extend Visualisable.");
            }
        }

        if (expSettings.interpret) {
            if (classifier instanceof Interpretable) {
                ((Interpretable) classifier).setInterpretabilitySavePath(expSettings.supportingFilePath);
            }
            else {
                expSettings.interpret = false;
                LOGGER.log(Level.WARNING, "Interpretability requested, but the classifier (" + classifier.getClass().getName() + ") does not extend Interpretable.");
            }
        }
    }

    /**
     * Performs all operations related to testing the classifier, and returns a ClassifierResults object holding the results
     * of testing.
     *
     * Computational resource costs of the training process are taken from the train results.
     */
    public static ClassifierResults testing(ExperimentalArguments expSettings, Classifier classifier, Instances testSet, ClassifierResults trainResults) throws Exception {
        ClassifierResults testResults = new ClassifierResults();

        //And now evaluate on the test set, if this wasn't a single parameter fold
        if (expSettings.singleParameterID == null) {
            //This is checked before the buildClassifier also, but
            //a) another process may have been doing the same experiment
            //b) we have a special case for the file builder that copies the results over in buildClassifier (apparently?)
            //no reason not to check again
            if (expSettings.forceEvaluation || expSettings.forceEvaluationTestFold || !CollateResults.validateSingleFoldFile(expSettings.testFoldFileName)) {
                if (classifier instanceof AbstractEarlyClassifier) testResults = evaluateEarlyClassifier(expSettings,
                        (AbstractEarlyClassifier) classifier, testSet);
                else testResults = evaluateClassifier(expSettings, classifier, testSet);
                testResults.setParas(trainResults.getParas());
                testResults.turnOffZeroTimingsErrors();
                testResults.setBenchmarkTime(testResults.getTimeUnit().convert(trainResults.getBenchmarkTime(), trainResults.getTimeUnit()));
                testResults.setBuildTime(testResults.getTimeUnit().convert(trainResults.getBuildTime(), trainResults.getTimeUnit()));
                testResults.turnOnZeroTimingsErrors();
                testResults.setMemory(trainResults.getMemory());
                LOGGER.log(Level.FINE, "Testing complete");

                writeResults(expSettings, testResults, expSettings.testFoldFileName, "test");
                LOGGER.log(Level.FINE, "Test results written");
            }
            else {
                LOGGER.log(Level.INFO, "Test file already found, written by another process.");
                testResults = new ClassifierResults(expSettings.testFoldFileName);
            }
        }
        else {
            LOGGER.log(Level.INFO, "This experiment evaluated a single training iteration or parameter set, skipping test phase.");
        }

        return testResults;
    }


    /**
     * Based on experimental parameters passed, defines the target results file and workspace locations for use in the
     * rest of the experiment
     */
    public static void buildExperimentDirectoriesAndFilenames(ExperimentalArguments expSettings, Classifier classifier) {
        //Build/make the directory to write the train and/or testFold files to
        // [writeLoc]/[classifier]/Predictions/[dataset]/
        String fullWriteLocation = expSettings.resultsWriteLocation + expSettings.estimatorName + "/"+PREDICTIONS_DIR+"/" + expSettings.datasetName + "/";
        File f = new File(fullWriteLocation);
        if (!f.exists())
            f.mkdirs();

        expSettings.testFoldFileName = fullWriteLocation + "testFold" + expSettings.foldId + ".csv";
        expSettings.trainFoldFileName = fullWriteLocation + "trainFold" + expSettings.foldId + ".csv";

        if (expSettings.singleParameterID != null && classifier instanceof ParameterSplittable)
            expSettings.testFoldFileName = expSettings.trainFoldFileName = fullWriteLocation + "fold" + expSettings.foldId + "_" + expSettings.singleParameterID + ".csv";

        testFoldExists = CollateResults.validateSingleFoldFile(expSettings.testFoldFileName);
        trainFoldExists = CollateResults.validateSingleFoldFile(expSettings.trainFoldFileName);

        // If needed, build/make the directory to write any supporting files to, e.g. checkpointing files
        // [writeLoc]/[classifier]/Workspace/[dataset]/[fold]/
        // todo foreseeable problems with threaded experiments:
        // user sets a supporting path for the 'master' exp, each generated exp to be run threaded inherits that path,
        // every classifier/dset/fold writes to same single location. For now, that's up to the user to recognise that's
        // going to be the case; supply a path and everything will be written there
        if (expSettings.supportingFilePath == null || expSettings.supportingFilePath.equals(""))
            expSettings.supportingFilePath = expSettings.resultsWriteLocation + expSettings.estimatorName + "/"+WORKSPACE_DIR+"/" + expSettings.datasetName + "/";

        f = new File(expSettings.supportingFilePath);
        if (!f.exists())
            f.mkdirs();
    }


    /**
     * Returns true if the work to be done in this experiment already exists at the locations defined by the experimental settings,
     * indicating that this execution can be skipped.
     */
    public static boolean quitEarlyDueToResultsExistence(ExperimentalArguments expSettings) {
        boolean quit = false;

        if (!expSettings.forceEvaluation && !expSettings.forceEvaluationTestFold && !expSettings.forceEvaluationTrainFold &&
                ((!expSettings.generateErrorEstimateOnTrainSet && testFoldExists) ||
                        (expSettings.generateErrorEstimateOnTrainSet && trainFoldExists  && testFoldExists))) {
            LOGGER.log(Level.INFO, expSettings.toShortString() + " already exists at " + expSettings.testFoldFileName + ", exiting.");
            quit = true;
        }

        return quit;
    }


    /**
     * This method cleans up and consolidates the information we have about the
     * build process into a ClassifierResults object, based on the capabilities of the
     * classifier and whether we want to be writing a train predictions file
     *
     * Regardless of whether the classifier is able to estimate it's own accuracy
     * or not, the returned train results should contain either
     *    a) timings, if expSettings.generateErrorEstimateOnTrainSet == false
     *    b) full predictions, if expSettings.generateErrorEstimateOnTrainSet == true
     *
     * @param exp
     * @param classifier
     * @param trainResults the results object so far which may be empty, contain the recorded
     *      timings of the particular classifier, or contain the results of a previous executed
     *      external estimation process,
     * @param buildTime as recorded by experiments.java, but which may not be used if the classifier
     *      records it's own build time more accurately
     * @return the finalised train results object
     * @throws Exception
     */
    public static ClassifierResults finaliseTrainResults(ExperimentalArguments exp, Classifier classifier, ClassifierResults trainResults, long buildTime, long benchmarkTime, TimeUnit expTimeUnit, long maxMemory) throws Exception {

        /*
        if estimateacc { //want full predictions
            timingToUpdateWith = buildTime (the one passed to this func)
            if is EnhancedAbstractClassifier {
                if able to estimate own acc
                    just return getTrainResults()
                else
                    timingToUpdateWith = getTrainResults().getBuildTime()
            }
            trainResults.setBuildTime(timingToUpdateWith)
            return trainResults
        }
        else not estimating acc { //just want timings
            if is EnhancedAbstractClassifier
                just return getTrainResults(), contains the timings and other maybe useful metainfo
            else
                trainResults passed are empty
                trainResults.setBuildTime(buildTime)
                return trainResults
        */

        //todo just enforce nanos everywhere, this is ridiculous. this needs overhaul

        long estimateToUpdateWith = 0; // no estimate by default
        long timingToUpdateWith = buildTime; //the timing that experiments measured by default

        if (exp.generateErrorEstimateOnTrainSet) { //want timings and full predictions
            TimeUnit timeUnitToUpdateWith = expTimeUnit;
            String paras = "No parameter info";

            if (classifier instanceof EnhancedAbstractClassifier) {
                EnhancedAbstractClassifier eac = ((EnhancedAbstractClassifier)classifier);
                if (eac.getEstimateOwnPerformance()) {
                    ClassifierResults res = eac.getTrainResults(); //classifier internally estimateed/recorded itself, just return that directly
                    res.setBenchmarkTime(res.getTimeUnit().convert(benchmarkTime, expTimeUnit));
                    res.setMemory(maxMemory);

                    return res;
                }
                else {
                    timingToUpdateWith = eac.getTrainResults().getBuildTime(); //update with classifier's own timings instead
                    timeUnitToUpdateWith = eac.getTrainResults().getTimeUnit();
                    paras = eac.getParameters();
                }
            }

            timingToUpdateWith = trainResults.getTimeUnit().convert(timingToUpdateWith, timeUnitToUpdateWith);
            estimateToUpdateWith = trainResults.getTimeUnit().convert(trainResults.getErrorEstimateTime(), timeUnitToUpdateWith);

            //update the externally produced results with the appropriate timing
            trainResults.setBuildTime(timingToUpdateWith);
            trainResults.setBuildPlusEstimateTime(timingToUpdateWith + estimateToUpdateWith);

            trainResults.setParas(paras);
        }
        else { // just want the timings
            if (classifier instanceof EnhancedAbstractClassifier) {
                trainResults = ((EnhancedAbstractClassifier) classifier).getTrainResults();
            }
            else {
                trainResults.setBuildTime(trainResults.getTimeUnit().convert(buildTime, expTimeUnit));
            }
        }

        trainResults.setBenchmarkTime(trainResults.getTimeUnit().convert(benchmarkTime, expTimeUnit));
        trainResults.setMemory(maxMemory);

        return trainResults;
    }

    /**
     * Based on the experimental settings passed, make any classifier interface calls that modify how the classifier is TRAINED here,
     * e.g. give checkpointable classifiers the location to save, give contractable classifiers their contract, etc.
     *
     * @return If the classifier is set up to evaluate a single parameter set on the train data, a new trainfilename shall be returned,
     *      otherwise null.
     *
     */
    private static String setupClassifierExperimentalOptions(ExperimentalArguments expSettings, Classifier classifier, Instances train) throws Exception {
        String parameterFileName = null;

        if (classifier instanceof Randomizable && expSettings.useSeed)
            if (expSettings.seed > Integer.MIN_VALUE)
                ((Randomizable)classifier).setSeed(expSettings.seed);
            else
                ((Randomizable)classifier).setSeed(expSettings.foldId);

        if (classifier instanceof MultiThreadable && expSettings.numberOfThreads != 1)
            if (expSettings.numberOfThreads < 1)
                ((MultiThreadable)classifier).enableMultiThreading();
            else
                ((MultiThreadable)classifier).enableMultiThreading(expSettings.numberOfThreads);

        if (classifier instanceof AbstractClassifier && expSettings.classifierOptions != null)
            ((AbstractClassifier)classifier).setOptions(expSettings.classifierOptions);

        // Parameter/thread/job splitting and checkpointing are treated as mutually exclusive, thus if/else
        if (expSettings.singleParameterID != null && classifier instanceof ParameterSplittable)//Single parameter fold
        {
            if (expSettings.checkpointing)
                LOGGER.log(Level.WARNING, "Parameter splitting AND checkpointing requested, but cannot do both. Parameter splitting turned on, checkpointing not.");

            if (classifier instanceof TunedRandomForest)
                ((TunedRandomForest) classifier).setNumFeaturesInProblem(train.numAttributes() - 1);

            expSettings.checkpointing = false;
            ((ParameterSplittable) classifier).setParametersFromIndex(expSettings.singleParameterID);
            parameterFileName = "fold" + expSettings.foldId + "_" + expSettings.singleParameterID + ".csv";
            expSettings.generateErrorEstimateOnTrainSet = true;

        }
        else {
            // Only do all this if not an internal _single parameter_ experiment
            // Save internal info for ensembles
            if (classifier instanceof SaveableEnsemble) { // mostly legacy, original hivecote code afaik
                ((SaveableEnsemble) classifier).saveResults(expSettings.supportingFilePath + "internalCV_" + expSettings.foldId + ".csv", expSettings.supportingFilePath + "internalTestPreds_" + expSettings.foldId + ".csv");
            }
            if (expSettings.checkpointing && classifier instanceof SaveEachParameter) { // for legacy things. mostly tuned classifiers
                ((SaveEachParameter) classifier).setPathToSaveParameters(expSettings.supportingFilePath + "fold" + expSettings.foldId + "_");
            }

            // Main thing to set:
            if (expSettings.checkpointing && classifier instanceof Checkpointable) {
                ((Checkpointable) classifier).setCheckpointPath(expSettings.supportingFilePath);

                if (expSettings.checkpointInterval > 0) {
                    // want to checkpoint at regular timings
                    // todo setCheckpointTimeHours expects int hours only, review
                    ((Checkpointable) classifier).setCheckpointTimeHours((int) TimeUnit.HOURS.convert(expSettings.checkpointInterval, TimeUnit.NANOSECONDS));
                }
                //else, as default
                    // want to checkpoint at classifier's discretion
            }
        }

        if(classifier instanceof TrainTimeContractable && expSettings.contractTrainTimeNanos>0)
            ((TrainTimeContractable) classifier).setTrainTimeLimit(TimeUnit.NANOSECONDS,expSettings.contractTrainTimeNanos);
        if(classifier instanceof TestTimeContractable && expSettings.contractTestTimeNanos >0)
            ((TestTimeContractable) classifier).setTestTimeLimit(TimeUnit.NANOSECONDS,expSettings.contractTestTimeNanos);

        return parameterFileName;
    }

    private static ClassifierResults findExternalTrainEstimate(ExperimentalArguments exp, Classifier classifier, Instances train, int fold) throws Exception {
        ClassifierResults trainResults = null;
        long trainBenchmark = findBenchmarkTime(exp);

        //todo clean up this hack. default is cv_10, as with all old trainFold results pre 2019/07/19
        String[] parts = exp.trainEstimateMethod.split("_");
        String method = parts[0];

        String para1 = null;
        if (parts.length > 1)
            para1 = parts[1];

        String para2 = null;
        if (parts.length > 2)
            para2 = parts[2];

        switch (method) {
            case "cv":
            case "CV":
            case "CrossValidationEvaluator":
                int numCVFolds = ClassifierExperiments.numCVFolds;
                if (para1 != null)
                    numCVFolds = Integer.parseInt(para1);
                numCVFolds = Math.min(train.numInstances(), numCVFolds);

                CrossValidationEvaluator cv = new CrossValidationEvaluator();
                cv.setSeed(fold);
                cv.setNumFolds(numCVFolds);
                trainResults = cv.crossValidateWithStats(classifier, train);
                break;

            case "hov":
            case "HOV":
            case "SingleTestSetEvaluator":
                double trainPropHov = DatasetLoading.getProportionKeptForTraining();
                if (para1 != null)
                    trainPropHov = Double.parseDouble(para1);

                SingleSampleEvaluator hov = new SingleSampleEvaluator();
                hov.setSeed(fold);
                hov.setPropInstancesInTrain(trainPropHov);
                trainResults = hov.evaluate(classifier, train);
                break;

            case "sr":
            case "SR":
            case "StratifiedResamplesEvaluator":
                int numSRFolds = 30;
                if (para1 != null)
                    numSRFolds = Integer.parseInt(para1);

                double trainPropSRR = DatasetLoading.getProportionKeptForTraining();
                if (para2 != null)
                    trainPropSRR = Double.parseDouble(para2);

                StratifiedResamplesEvaluator srr = new StratifiedResamplesEvaluator();
                srr.setSeed(fold);
                srr.setNumFolds(numSRFolds);
                srr.setUseEachResampleIdAsSeed(true);
                srr.setPropInstancesInTrain(trainPropSRR);
                trainResults = srr.evaluate(classifier, train);
                break;

            default:
                throw new Exception("Unrecognised method to estimate error on the train given: " + exp.trainEstimateMethod);
        }

        trainResults.setErrorEstimateMethod(exp.trainEstimateMethod);
        trainResults.setBenchmarkTime(trainBenchmark);

        return trainResults;
    }

    public static void serialiseClassifier(ExperimentalArguments expSettings, Classifier classifier) throws FileNotFoundException, IOException {
        String filename = expSettings.supportingFilePath + expSettings.estimatorName + "_" + expSettings.datasetName + "_" + expSettings.foldId + ".ser";

        LOGGER.log(Level.FINE, "Attempting classifier serialisation, to " + filename);

        FileOutputStream fos = new FileOutputStream(filename);
        try (ObjectOutputStream out = new ObjectOutputStream(fos)) {
            out.writeObject(classifier);
            fos.close();
            out.close();
        }

        LOGGER.log(Level.FINE, "Classifier serialised successfully");
    }

    /**
     * Meta info shall be set by writeResults(...), just generating the prediction info and
     * any info directly calculable from that here
     */
    public static ClassifierResults evaluateClassifier(ExperimentalArguments exp, Classifier classifier, Instances testSet) throws Exception {
        SingleTestSetEvaluator eval = new SingleTestSetEvaluator(exp.foldId, false, true, exp.interpret); //DONT clone data, DO set the class to be missing for each inst

        return eval.evaluate(classifier, testSet);
    }

    /**
     * Mimics SingleTestSetEvaluator but for early classification classifiers.
     * Earliness for each test instance is written to the description.
     * Normalisation for experimental purposes should be handled by the individual classifiers/decision makers.
     */
    public static ClassifierResults evaluateEarlyClassifier(ExperimentalArguments exp, AbstractEarlyClassifier classifier, Instances testSet) throws Exception {
        ClassifierResults res = new ClassifierResults(testSet.numClasses());
        res.setTimeUnit(TimeUnit.NANOSECONDS);
        res.setClassifierName(classifier.getClass().getSimpleName());
        res.setDatasetName(testSet.relationName());
        res.setFoldID(exp.foldId);
        res.setSplit("test");

        int length = testSet.numAttributes()-1;
        int[] thresholds = classifier.getThresholds();
        Instances[] truncatedInstances = new Instances[thresholds.length];
        truncatedInstances[thresholds.length-1] = new Instances(testSet, 0);
        for (int i = 0; i < thresholds.length-1; i++) {
            truncatedInstances[i] = truncateInstances(truncatedInstances[thresholds.length-1], length, thresholds[i]);
        }

        res.turnOffZeroTimingsErrors();
        for (Instance testinst : testSet) {
            double trueClassVal = testinst.classValue();
            testinst.setClassMissing();

            long startTime = System.nanoTime();

            double[] dist = null;
            double earliness = 0;
            for (int i = 0; i < thresholds.length; i++){
                Instance newInst = truncateInstance(testinst, length, thresholds[i]);
                newInst.setDataset(truncatedInstances[i]);

                dist = classifier.distributionForInstance(newInst);

                if (dist != null) {
                    earliness = thresholds[i]/(double)length;
                    break;
                }
            }

            long predTime = System.nanoTime() - startTime;

            res.addPrediction(trueClassVal, dist, indexOfMax(dist), predTime, Double.toString(earliness));
        }

        res.turnOnZeroTimingsErrors();

        res.finaliseResults();
        res.findAllStatsOnce();

        return res;
    }

    /**
     * If exp.performTimingBenchmark = true, this will return the total time to
     * sort 1,000 arrays of size 10,000
     *
     * Expected time on Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz is ~0.8 seconds
     *
     * This can still anecdotally vary between 0.75 to 1.05 on my windows machine, however.
     */
    public static long findBenchmarkTime(ExperimentalArguments exp) {
        if (!exp.performTimingBenchmark)
            return -1; //the default in classifierresults, i.e no benchmark

        // else calc benchmark

        int arrSize = 10000;
        int repeats = 1000;
        long[] times = new long[repeats];
        long total = 0L;
        for (int i = 0; i < repeats; i++) {
            times[i] = atomicBenchmark(arrSize);
            total+=times[i];
        }

        if (debug) {
            long mean = 0L, max = Long.MIN_VALUE, min = Long.MAX_VALUE;
            for (long time : times) {
                mean += time;
                if (time < min)
                    min = time;
                if (time > max)
                    max = time;
            }
            mean/=repeats;

            int halfR = repeats/2;
            long median = repeats % 2 == 0 ?
                    (times[halfR] + times[halfR+1]) / 2 :
                    times[halfR];

            double d = 1000000000;
            StringBuilder sb = new StringBuilder("BENCHMARK TIMINGS, summary of times to "
                    + "sort "+repeats+" random int arrays of size "+arrSize+" - in seconds\n");
            sb.append("total = ").append(total/d).append("\n");
            sb.append("min = ").append(min/d).append("\n");
            sb.append("max = ").append(max/d).append("\n");
            sb.append("mean = ").append(mean/d).append("\n");
            sb.append("median = ").append(median/d).append("\n");

            LOGGER.log(Level.FINE, sb.toString());
        }

        return total;
    }

    private static long atomicBenchmark(int arrSize) {
        long startTime = System.nanoTime();
        int[] arr = new int[arrSize];
        Random rng = new Random(0);
        for (int j = 0; j < arrSize; j++)
            arr[j] = rng.nextInt();

        Arrays.sort(arr);
        return System.nanoTime() - startTime;
    }

    public static String buildExperimentDescription() {
        //TODO get system information, e.g. cpu clock-speed. generic across os too
        Date date = new Date();
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

        StringBuilder sb = new StringBuilder("Generated by ClassifierExperiments.java on " + formatter.format(date) + ".");

        sb.append("    SYSTEMPROPERTIES:{");
        sb.append("user.name:").append(System.getProperty("user.name", "unknown"));
        sb.append(",os.arch:").append(System.getProperty("os.arch", "unknown"));
        sb.append(",os.name:").append(System.getProperty("os.name", "unknown"));
        sb.append("},ENDSYSTEMPROPERTIES");

        return sb.toString().replace("\n", "NEW_LINE");
    }

    public static void writeResults(ExperimentalArguments exp, ClassifierResults results, String fullTestWritingPath, String split) throws Exception {
        results.setClassifierName(exp.estimatorName);
        results.setDatasetName(exp.datasetName);
        results.setFoldID(exp.foldId);
        results.setSplit(split);
        results.setDescription(buildExperimentDescription());

        //todo, need to make design decisions with the classifierresults enum to clean this switch up
        switch (exp.classifierResultsFileFormat) {
            case 0: //PREDICTIONS
                results.writeFullResultsToFile(fullTestWritingPath);
                break;
            case 1: //METRICS
                results.writeSummaryResultsToFile(fullTestWritingPath);
                break;
            case 2: //COMPACT
                results.writeCompactResultsToFile(fullTestWritingPath);
                break;
            default: {
                System.err.println("Classifier Results file writing format not recognised, "+exp.classifierResultsFileFormat+", just writing the full predictions.");
                results.writeFullResultsToFile(fullTestWritingPath);
                break;
            }

        }

        File f = new File(fullTestWritingPath);
        if (f.exists()) {
            f.setWritable(true, false);
        }
    }

    /**
     * Will run through all combinations of classifiers*datasets*folds provided, using the meta experimental info stored in the
     * standardArgs. Will by default set numThreads = numCores
     *
     * If using bespoke classifiers (not found in setClassifier), e.g. different parameterisations, bespoke ensembles etc,
     * provide a generator function for each classifier, in a list that is parallel with classifierNames. Assuming the
     * classifier is Randomizable, the seed shall be set equal to the expSettings foldId
     *
     * If simply using setClassifier to instantiate classifiers, classifierGenerators itself or fields within it can be null
     *
     * For e.g. classifierNames = { "TSF" }, these methods of classifier instance generation are all equivalent
     *       -  classifierGenerators = null                                          // uses setClassifier("TSF")
     *       -  classifierGenerators = Arrays.asList(null);                          // uses setClassifier("TSF")
     *       -  classifierGenerators = Arrays.asList(() -> {return new TSF();});     // be careful with rng seeding though
     *       -  classifierGenerators = Arrays.asList(() -> {return setClassifierClassic("TSF",0)});
     */
    public static void setupAndRunMultipleExperimentsThreaded(ExperimentalArguments standardArgs, String[] classifierNames, List<Supplier<Classifier>> classifierGenerators, String[] datasetNames, int minFolds, int maxFolds) throws Exception{
        setupAndRunMultipleExperimentsThreaded(standardArgs, classifierNames, classifierGenerators, datasetNames, minFolds, maxFolds, 0);


        /*
        bespoke classifier example usage:

        ClassifierExperiments.ExperimentalArguments standardArgs = new ClassifierExperiments.ExperimentalArguments();
        standardArgs.dataReadLocation = "src/main/java/experiments/data/uci/";
        standardArgs.resultsWriteLocation = "C:/Temp/tests/";

        String[] classifierNames = { "ED", "RandF", "BespokeEnsemble" };

        Supplier<Classifier> ensembleSupplier = () -> {
            CAWPE cawpe = new CAWPE();
            cawpe.setClassifiersForBuildingInMemory(new Classifier[] { new ED1NN(), new RandomForest() });
            return cawpe;
        };

        List<Supplier<Classifier>> classifierGenerators = Arrays.asList(
            () -> {return new ED1NN();},
            () -> {return new RandomForest();},
            ensembleSupplier
        );
        String[] datasets = { "hayes-roth", "iris", "teaching" };
        int numFolds = 3;

        ClassifierExperiments.setupAndRunMultipleExperimentsThreaded(standardArgs, classifierNames, classifierGenerators, datasets, 0, numFolds);

         */
    }

    /**
     * Will run through all combinations of classifiers*datasets*folds provided, using the meta experimental info stored in the
     * standardArgs. If numThreads > 0, will spawn that many threads. If numThreads == 0, will use as many threads as there are cores,
     * else if numThreads == -1, will spawn as many threads as there are cores minus 1, to aid usability of the machine.
     *
     * If using bespoke classifiers (not found in setClassifier), e.g. different parameterisations, bespoke ensembles etc,
     * provide a generator function for each classifier, in a list that is parallel with classifierNames. Assuming the
     * classifier is Randomizable, the seed shall be set equal to the expSettings foldId
     *
     * If simply using setClassifier to instantiate classifiers, classifierGenerators itself or fields within it can be null
     *
     * For e.g. classifierNames = { "TSF" }, these methods of classifier instance generation are all equivalent
     *       -  classifierGenerators = null                                          // uses setClassifier("TSF")
     *       -  classifierGenerators = Arrays.asList(null);                          // uses setClassifier("TSF")
     *       -  classifierGenerators = Arrays.asList(() -> {return new TSF();});     // be careful with rng seeding though
     *       -  classifierGenerators = Arrays.asList(() -> {return setClassifierClassic("TSF",0)});
     */
    public static void setupAndRunMultipleExperimentsThreaded(ExperimentalArguments standardArgs, String[] classifierNames, List<Supplier<Classifier>> classifierGenerators, String[] datasetNames, int minFolds, int maxFolds, int numThreads) throws Exception{
        int numCores = Runtime.getRuntime().availableProcessors();
        if (numThreads == 0)
            numThreads = numCores;
        else if (numThreads < 0)
            numThreads = Math.max(1, numCores-1);

        System.out.println("# cores ="+numCores);
        System.out.println("# threads ="+numThreads);
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);

        List<ExperimentalArguments> exps = standardArgs.generateExperiments(classifierNames, classifierGenerators, datasetNames, minFolds, maxFolds);
        for (ExperimentalArguments exp : exps)
            executor.execute(exp);

        executor.shutdown();
        while (!executor.isTerminated()) {
        }
        System.out.println("Finished all threads");
    }


}
