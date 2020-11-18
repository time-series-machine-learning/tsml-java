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
package experiments;

import com.google.common.testing.GcFinalization;
import machine_learning.classifiers.SaveEachParameter;
import machine_learning.classifiers.tuned.TunedRandomForest;
import experiments.data.DatasetLists;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.JCommander.Builder;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Level;
import java.util.logging.Logger;

import tsml.classifiers.*;
import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.SingleSampleEvaluator;
import tsml.classifiers.distance_based.utils.strings.StrUtils;
import weka.classifiers.Classifier;
import evaluation.storage.ClassifierResults;
import evaluation.evaluators.SingleTestSetEvaluator;
import evaluation.evaluators.StratifiedResamplesEvaluator;
import experiments.data.DatasetLoading;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import machine_learning.classifiers.ensembles.SaveableEnsemble;
import weka.core.Instances;

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

 main(String[] args) info:
 Parses args into an ExperimentalArguments object, then calls setupAndRunExperiment(ExperimentalArguments expSettings).
 Calling with the --help argument, or calling with un-parsable parameters, will print a summary of the possible parameters.

 Argument key-value pairs are separated by '='. The 5 basic, always required, arguments are:
 Para name (short/long)  |    Example
 -dp --dataPath          |    --dataPath=C:/Datasets/
 -rp --resultsPath       |    --resultsPath=C:/Results/
 -cn --classifierName    |    --classifierName=RandF
 -dn --datasetName       |    --datasetName=ItalyPowerDemand
 -f  --fold              |    --fold=1

 Use --help to see all the optional parameters, and more information about each of them.

 If running locally, it may be easier to build the ExperimentalArguments object yourself and call setupAndRunExperiment(...)
 directly.
 *
 * @author James Large (james.large@uea.ac.uk), Tony Bagnall (anthony.bagnall@uea.ac.uk)
 */
public class Experiments  {

    private final static Logger LOGGER = Logger.getLogger(Experiments.class.getName());

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
        if (!beQuiet) {
            System.out.println("Raw args:");
            for (String str : args)
                System.out.println("\t"+str);
            System.out.println("");
        }

        if (args.length > 0) {
            ExperimentalArguments expSettings = new ExperimentalArguments(args);
            setupAndRunExperiment(expSettings);
        }
        else {//Manually set args
            int folds=1;
            String[] settings=new String[9];

            /*
             * Change these settings for your experiment:
             */
//            String[] classifiers={"TSF_I","RISE_I","STC_I","CBOSS_I","HIVE-COTEn_I"};
//            String classifier=classifiers[2];
            String classifier="STC";//Classifier name: See ClassifierLists for valid options

            settings[0]="-dp=C:\\Data Working Area\\Datasets"; //Where to get datasets
            settings[1]="-rp=C:\\Experiments\\Results\\"; //Where to write results
            settings[2]="-gtf=false"; //Whether to generate train files or not
            settings[3]="-cn="+classifier; //Classifier name
            settings[4]="-dn="; //Problem name, don't change here as it is overwritten by probFiles
            settings[5]="-f=1"; //Fold number (fold number 1 is stored as testFold0.csv, its a cluster thing)
            settings[6]="-ctr=600s"; //Time contract
            settings[7]="-d=true"; //Debugging
            settings[8]="--force=true"; //Overwrites existing results if true, otherwise set to false

            String[] probFiles= {"ItalyPowerDemand"}; //Problem name(s)
//            String[] probFiles= DatasetLists.fixedLengthMultivariate;
            /*
             * END OF SETTINGS
             */

            System.out.println("Manually set args:");
            for (String str : settings)
                System.out.println("\t"+str);
            System.out.println("");

            boolean threaded=true;
            if (threaded) {
                ExperimentalArguments expSettings = new ExperimentalArguments(settings);
                System.out.println("Threaded experiment with "+expSettings);
//              setupAndRunMultipleExperimentsThreaded(expSettings, classifiers,probFiles,0,folds);
                setupAndRunMultipleExperimentsThreaded(expSettings, new String[]{classifier},probFiles,0,folds);
            }
            else {//Local run without args, mainly for debugging
                for (String prob:probFiles) {
                    settings[4]="-dn="+prob;

                    for(int i=1;i<=folds;i++) {
                        settings[5]="-f="+i;
                        ExperimentalArguments expSettings = new ExperimentalArguments(settings);
//                      System.out.println("Sequential experiment with "+expSettings);
                        setupAndRunExperiment(expSettings);
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

        // Cases in the classifierlist can now change the classifier name to reflect particular parameters wanting to be
        // represented as different classifiers, e.g. ST_1day, ST_2day
        // The set classifier call is therefore made before defining paths that are dependent on the classifier name
        Classifier classifier = ClassifierLists.setClassifier(expSettings);

        buildExperimentDirectoriesAndFilenames(expSettings, classifier);
        //Check whether results already exists, if so and force evaluation is false: just quit
        if (quitEarlyDueToResultsExistence(expSettings))
            return null;

        Instances[] data = DatasetLoading.sampleDataset(expSettings.dataReadLocation, expSettings.datasetName, expSettings.foldId);
        setupClassifierExperimentalOptions(expSettings, classifier, data[0]);
        ClassifierResults[] results = runExperiment(expSettings, data[0], data[1], classifier);
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

        if (expSettings.generateErrorEstimateOnTrainSet && (!trainFoldExists || expSettings.forceEvaluation)) {
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

        if (expSettings.generateErrorEstimateOnTrainSet && (!trainFoldExists || expSettings.forceEvaluation)) {
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
            if (expSettings.forceEvaluation || !CollateResults.validateSingleFoldFile(expSettings.testFoldFileName)) {
                testResults = evaluateClassifier(expSettings, classifier, testSet);
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
        String fullWriteLocation = expSettings.resultsWriteLocation + expSettings.classifierName + "/"+PREDICTIONS_DIR+"/" + expSettings.datasetName + "/";
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
            expSettings.supportingFilePath = expSettings.resultsWriteLocation + expSettings.classifierName + "/"+WORKSPACE_DIR+"/" + expSettings.datasetName + "/";

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

        if (!expSettings.forceEvaluation &&
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

        if (exp.generateErrorEstimateOnTrainSet) { //want timings and full predictions
            long timingToUpdateWith = buildTime; //the timing that experiments measured by default
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
            long estimateToUpdateWith = trainResults.getTimeUnit().convert(trainResults.getErrorEstimateTime(), timeUnitToUpdateWith);

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
    private static String setupClassifierExperimentalOptions(ExperimentalArguments expSettings, Classifier classifier, Instances train) {
        String parameterFileName = null;


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
                int numCVFolds = Experiments.numCVFolds;
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
        String filename = expSettings.supportingFilePath + expSettings.classifierName + "_" + expSettings.datasetName + "_" + expSettings.foldId + ".ser";

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

        StringBuilder sb = new StringBuilder("Generated by Experiments.java on " + formatter.format(date) + ".");

        sb.append("    SYSTEMPROPERTIES:{");
        sb.append("user.name:").append(System.getProperty("user.name", "unknown"));
        sb.append(",os.arch:").append(System.getProperty("os.arch", "unknown"));
        sb.append(",os.name:").append(System.getProperty("os.name", "unknown"));
        sb.append("},ENDSYSTEMPROPERTIES");

        return sb.toString().replace("\n", "NEW_LINE");
    }

    public static void writeResults(ExperimentalArguments exp, ClassifierResults results, String fullTestWritingPath, String split) throws Exception {
        results.setClassifierName(exp.classifierName);
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

    @Parameters(separators = "=")
    public static class ExperimentalArguments implements Runnable {

        //REQUIRED PARAMETERS
        @Parameter(names={"-dp","--dataPath"}, required=true, order=0, description = "(String) The directory that contains the dataset to be evaluated on, in the form "
                + "[--dataPath]/[--datasetName]/[--datasetname].arff (the actual arff file(s) may be in different forms, see Experiments.sampleDataset(...).")
        public String dataReadLocation = null;

        @Parameter(names={"-rp","--resultsPath"}, required=true, order=1, description = "(String) The parent directory to write the results of the evaluation to, in the form "
                + "[--resultsPath]/[--classifierName]/Predictions/[--datasetName]/...")
        public String resultsWriteLocation = null;

        @Parameter(names={"-cn","--classifierName"}, required=true, order=2, description = "(String) The name of the classifier to evaluate. A case matching this value should exist within the ClassifierLists")
        public String classifierName = null;

        @Parameter(names={"-dn","--datasetName"}, required=true, order=3, description = "(String) The name of the dataset to be evaluated on, which resides within the dataPath in the form "
                + "[--dataPath]/[--datasetName]/[--datasetname].arff (the actual arff file(s) may be of different forms, see Experiments.sampleDataset(...).")
        public String datasetName = null;

        @Parameter(names={"-f","--fold"}, required=true, order=4, description = "(int) The fold index for dataset resampling, also used as the rng seed. *Indexed from 1* to conform with cluster array "
                + "job indices. The fold id pass will be automatically decremented to be zero-indexed internally.")
        public int foldId = 0;

        //OPTIONAL PARAMETERS
        @Parameter(names={"--help"}, hidden=true) //hidden from usage() printout
        private boolean help = false;

        //todo separate verbosity into it own thing
        @Parameter(names={"-d","--debug"}, arity=1, description = "(boolean) Increases verbosity and turns on the printing of debug statements")
        public boolean debug = false;

        @Parameter(names={"-gtf","--genTrainFiles"}, arity=1, description = "(boolean) Turns on the production of trainFold[fold].csv files, the results of which are calculate either via a cross validation of "
                + "the train data, or if a classifier implements the TrainAccuracyEstimate interface, the classifier will write its own estimate via its own means of evaluation.")
        public boolean generateErrorEstimateOnTrainSet = false;

        @Parameter(names={"-cp","--checkpointing"}, arity=1, description = "(boolean or String) Turns on the usage of checkpointing, if the classifier implements the SaveParameterInfo and/or CheckpointClassifier interfaces. "
                + "Default is false/0, for no checkpointing. if -cp = true, checkpointing is turned on and checkpointing frequency is determined by the classifier. if -cp is a timing of the form [int][char], e.g. 1h, "
                + "checkpoints shall be made at that frequency (as close as possible according to the atomic unit of learning for the classifier). Possible units, in order: n (nanoseconds), u, m, s, M, h, d (days)."
                + "Lastly, if -cp is of the the [int] only, it is assumed to be a timing in hours."
                + "The classifier by default will write its checkpointing files to workspace path parallel to the --resultsPath, unless another path is optionally supplied to --supportingFilePath.")
        private String checkpointingStr = null;
        public boolean checkpointing = false;
        public long checkpointInterval = 0;

        @Parameter(names={"-vis","--visualisation"}, description = "(boolean) Turns on the production of visualisation files, if the classifier implements the Visualisable interface. "
                + "Figures are created using Python. Exact requirements are to be determined, but a a Python 3.7 installation is the current recommendation with the numpy and matplotlib packages installed on the global environment. "
                + "The classifier by default will write its visualisation files to workspace path parallel to the --resultsPath, unless another path is optionally supplied to --supportingFilePath.")
        public boolean visualise = false;

        @Parameter(names={"-int","--interpretability"}, description = "(boolean) Turns on the production of interpretability files, if the classifier implements the Interpretable interface. "
                + "The classifier by default will write its interpretability files to workspace path parallel to the --resultsPath, unless another path is optionally supplied to --supportingFilePath.")
        public boolean interpret = false;

        @Parameter(names={"-sp","--supportingFilePath"}, description = "(String) Specifies the directory to write any files that may be produced by the classifier if it is a FileProducer. This includes but may not be "
                + "limited to: parameter evaluations, checkpoints, and logs. By default, these files are written to a generated subdirectory in the same location that the train and testFold[fold] files are written, relative"
                + "the --resultsPath. If a path is supplied via this parameter however, the files shall be written to that precisely that directory, as opposed to e.g. [-sp]/[--classifierName]/Predictions... "
                + "THIS IS A PLACEHOLDER PARAMETER. TO BE FULLY IMPLEMENTED WHEN INTERFACES AND SETCLASSIFIER ARE UPDATED.")
        public String supportingFilePath = null;

        @Parameter(names={"-pid","--parameterSplitIndex"}, description = "(Integer) If supplied and the classifier implements the ParameterSplittable interface, this execution of experiments will be set up to evaluate "
                + "the parameter set -pid within the parameter space used by the classifier (whether that be a supplied space or default). How the integer -pid maps onto the parameter space is up to the classifier.")
        public Integer singleParameterID = null;

        @Parameter(names={"-tb","--timingBenchmark"}, arity=1, description = "(boolean) Turns on the computation of a standard operation to act as a simple benchmark for the speed of computation on this hardware, which may "
                + "optionally be used to normalise build/test/predictions times across hardware in later analysis. Expected time on Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz is ~0.8 seconds. For experiments that are likely to be very "
                + "short, it is recommended to leave this off, as it will proportionally increase the total time to perform all your experiments by a great deal, and for short evaluation time the proportional affect of "
                + "any processing noise may make any benchmark normalisation process unreliable anyway.")
        public boolean performTimingBenchmark = false;

        //todo expose the filetype enum in some way, currently just using an unconnected if statement, if e.g the order of the enum values changes in the classifierresults, which we have no knowledge
        //of here, the ifs will call the wrong things. decide on the design of this
        @Parameter(names={"-ff","--fileFormat"}, description = "(int) Specifies the format for the classifier results file to be written in, accepted values = { 0, 1, 2 }, default = 0. 0 writes the first 3 lines of meta information "
                + "as well as the full prediction information, and requires the most disk space. 1 writes the first three lines and a list of the performance metrics calculated from the prediction info. 2 writes the first three lines only, and "
                + "requires the least space. Use options other than 0 if generating too many files with too much prediction information for the disk space available, however be aware that there is of course a loss of information.")
        public int classifierResultsFileFormat = 0;

        @Parameter(names={"-ctr","--contractTrain"}, description = "(String) Defines a time limit for the training of the classifier if it implements the TrainTimeContractClassifier interface. Defaults to "
                + "no contract time. If an integral value is given, it is assumed to be in HOURS. Otherwise, a string of the form [int][char] can be supplied, with the [char] defining the time unit. "
                + "e.g.1 10s = 10 seconds,   e.g.2 1h = 60M = 3600s. Possible units, in order: n (nanoseconds), u, m, s, M, h, d (days).")
        private String contractTrainTimeString = null;
        public long contractTrainTimeNanos = 0;

        @Parameter(names={"-cte","--contractTest"}, description = "(String) Defines a time limit for the testing of the classifier if it implements the TestTimeContractable interface. Defaults to "
                + "no contract time. If an integral value is given, it is assumed to be in HOURS. Otherwise, a string of the form [int][char] can be supplied, with the [char] defining the time unit. "
                + "e.g.1 10s = 10 seconds,   e.g.2 1h = 60M = 3600s. Possible units, in order: n (nanoseconds), u, m, s, M, h, d (days).")
        private String contractTestTimeString = null;
        public long contractTestTimeNanos = 0;

        @Parameter(names={"-sc","--serialiseClassifier"}, arity=1, description = "(boolean) If true, and the classifier is serialisable, the classifier will be serialised to the --supportingFilesPath after training, but before testing.")
        public boolean serialiseTrainedClassifier = false;

        @Parameter(names={"--force"}, arity=1, description = "(boolean) If true, the evaluation will occur even if what would be the resulting file already exists. The old file will be overwritten with the new evaluation results.")
        public boolean forceEvaluation = false;

        @Parameter(names={"-tem", "--trainEstimateMethod"}, arity=1, description = "(String) Defines the method and parameters of the evaluation method used to estimate error on the train set, if --genTrainFiles == true. Current implementation is a hack to get the option in for"
                + " experiment running in the short term. Give one of 'cv' and 'hov' for cross validation and hold-out validation set respectively, and a number of folds (e.g. cv_10) or train set proportion (e.g. hov_0.7) respectively. Default is a 10 fold cv, i.e. cv_10.")
        public String trainEstimateMethod = "cv_10";

        @Parameter(names={"--conTrain"}, arity = 2, description = "todo")
        private List<String> trainContracts = new ArrayList<>();

        @Parameter(names={"--contractInName"}, arity = 1, description = "todo")
        private boolean appendTrainContractToClassifierName = true;

        @Parameter(names={"-l", "--logLevel"}, description = "log level")
        private String logLevelStr = null;

        private Level logLevel = null;

        public boolean hasTrainContracts() {
            return trainContracts.size() > 0;
        }


        // calculated/set during experiment setup, indirectly using the parameters passed
        public String trainFoldFileName = null;
        public String testFoldFileName = null;

        public ExperimentalArguments() {

        }

        public ExperimentalArguments(String[] args) throws Exception {
            parseArguments(args);
        }

        @Override //Runnable
        public void run() {
            try {
                setupAndRunExperiment(this);
            } catch (Exception ex) {
                ex.printStackTrace();
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

                        // copying fields via reflection now to avoid cases of forgetting to account for newly added paras
                        for (Field field : ExperimentalArguments.class.getFields()) {

                            // these are the ones being set individually per exp, skip the copying over
                            if (field.getName().equals("classifierName") ||
                                    field.getName().equals("datasetName") ||
                                    field.getName().equals("foldId"))
                                continue;

                            try {
                                field.set(exp, field.get(this));
                            } catch (IllegalAccessException ex) {
                                System.out.println("Fatal, should-be-unreachable exception thrown while copying across exp args");
                                System.out.println(ex);
                                ex.printStackTrace();
                                System.exit(0);
                            }
                        }

                        exps.add(exp);
                    }
                }
            }

            return exps;
        }

        private void parseArguments(String[] args) throws Exception {
            Builder b = JCommander.newBuilder();
            b.addObject(this);
            JCommander jc = b.build();
            jc.setProgramName("Experiments.java");  //todo maybe add copyright etcetc
            try {
                jc.parse(args);
            } catch (Exception e) {
                if (!help) {
                    //we actually errored, instead of the program simply being called with the --help flag
                    System.err.println("Parsing of arguments failed, parameter information follows after the error. Parameters that require values should have the flag and value separated by '='.");
                    System.err.println("For example: java -jar TimeSeriesClassification.jar -dp=data/path/ -rp=results/path/ -cn=someClassifier -dn=someDataset -f=0");
                    System.err.println("Parameters prefixed by a * are REQUIRED. These are the first five parameters, which are needed to run a basic experiment.");
                    System.err.println("Error: \n\t"+e+"\n\n");
                }
                jc.usage();
//                Thread.sleep(1000); //usage can take a second to print for some reason?... no idea what it's actually doing
//                System.exit(1);
            }

            foldId -= 1; //go from one-indexed to zero-indexed
            Experiments.debug = this.debug;

            resultsWriteLocation = StrUtils.asDirPath(resultsWriteLocation);
            dataReadLocation = StrUtils.asDirPath(dataReadLocation);
            if (checkpointingStr != null) {
                //some kind of checkpointing is wanted

                // is it simply "true"?

                checkpointing = Boolean.parseBoolean(checkpointingStr.toLowerCase());
                if(!checkpointing){
                    //it's not. must be a timing string
                    checkpointing = true;
                    checkpointInterval = parseTiming(checkpointingStr);

                }
          }

            //populating the contract times if present
            if (contractTrainTimeString != null)
                contractTrainTimeNanos = parseTiming(contractTrainTimeString);
            if (contractTestTimeString != null)
                contractTestTimeNanos = parseTiming(contractTestTimeString);

            if(contractTrainTimeNanos > 0) {
                trainContracts.add(String.valueOf(contractTrainTimeNanos));
                trainContracts.add(TimeUnit.NANOSECONDS.toString());
            }

            // check the contracts are in ascending order // todo sort them
            for(int i = 1; i < trainContracts.size(); i += 2) {
                trainContracts.set(i, trainContracts.get(i).toUpperCase());
            }
            long prev = -1;
            for(int i = 0; i < trainContracts.size(); i += 2) {
                long nanos = TimeUnit.NANOSECONDS.convert(Long.parseLong(trainContracts.get(i)),
                        TimeUnit.valueOf(trainContracts.get(i + 1)));
                if(prev > nanos) {
                    throw new IllegalArgumentException("contracts not in asc order");
                }
                prev = nanos;
            }

            if(trainContracts.size() % 2 != 0) {
                throw new IllegalStateException("illegal number of args for time");
            }

            if(logLevelStr != null) {
                logLevel = Level.parse(logLevelStr);
            }
        }

        /**
         * Helper func to parse a timing string of the form [int][char], e.g. 10s = 10 seconds = 10,000,000,000 nanosecs.
         * 1h = 60M = 3600s = 3600,000,000,000n
         *
         * todo Alternatively, string can be of form [int][TimeUnit.toString()], e.g. 10SECONDS
         *
         * If just a number is given without a time unit character, HOURS is assumed to be the time unit
         *
         * Possible time unit chars:
         * n - nanoseconds
         * u - microseconds
         * m - milliseconds
         * s - seconds
         * M - minutes
         * h - hours
         * d - days
         * w - weeks
         *
         * todo learn/use java built in timing things if really wanted, e.g. TemporalAmount
         *
         * @return long number of nanoseconds the input string represents
         */
        private long parseTiming(String timeStr) throws IllegalArgumentException{
            try {
                // check if it's just a number, in which case return it under assumption that it's in hours
                int val = Integer.parseInt(timeStr);
                return TimeUnit.NANOSECONDS.convert(val, TimeUnit.HOURS);
            } catch (Exception e) {
                //pass
            }

            // convert it
            char unit = timeStr.charAt(timeStr.length()-1);
            int amount = Integer.parseInt(timeStr.substring(0, timeStr.length()-1));

            long nanoAmount = 0;

            switch (unit) {
                case 'n': nanoAmount = amount; break;
                case 'u': nanoAmount = TimeUnit.NANOSECONDS.convert(amount, TimeUnit.MICROSECONDS); break;
                case 'm': nanoAmount = TimeUnit.NANOSECONDS.convert(amount, TimeUnit.MILLISECONDS); break;
                case 's': nanoAmount = TimeUnit.NANOSECONDS.convert(amount, TimeUnit.SECONDS); break;
                case 'M': nanoAmount = TimeUnit.NANOSECONDS.convert(amount, TimeUnit.MINUTES); break;
                case 'h': nanoAmount = TimeUnit.NANOSECONDS.convert(amount, TimeUnit.HOURS); break;
                case 'd': nanoAmount = TimeUnit.NANOSECONDS.convert(amount, TimeUnit.DAYS); break;
                default:
                    throw new IllegalArgumentException("Unrecognised time unit string conversion requested, was given " + timeStr);
            }

            return nanoAmount;
        }

        public String toShortString() {
            return "["+classifierName+","+datasetName+","+foldId+"]";
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();

            sb.append("EXPERIMENT SETTINGS "+ this.toShortString());

            // printing fields via reflection now to avoid cases of forgetting to account for newly added  paras
            for (Field field : ExperimentalArguments.class.getFields()) {
                try {
                    sb.append("\n").append(field.getName()).append(": ").append(field.get(this));
                } catch (IllegalAccessException ex) {
                    System.out.println("Fatal, should-be-unreachable exception thrown while printing exp args");
                    System.out.println(ex);
                    ex.printStackTrace();
                    System.exit(0);
                }
            }

            return sb.toString();
        }
    }


}
