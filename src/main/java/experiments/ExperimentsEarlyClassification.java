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

import weka.core.Instance;
import weka_extras.classifiers.SaveEachParameter;
import weka_extras.classifiers.tuned.TunedRandomForest;
import experiments.data.DatasetLists;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.JCommander.Builder;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.logging.Level;
import java.util.logging.Logger;
import timeseriesweka.classifiers.ParameterSplittable;
import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.SingleSampleEvaluator;
import timeseriesweka.classifiers.SaveParameterInfo;
import weka.classifiers.Classifier;
import evaluation.storage.ClassifierResults;
import evaluation.evaluators.SingleTestSetEvaluator;
import evaluation.evaluators.StratifiedResamplesEvaluator;
import experiments.data.DatasetLoading;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import utilities.InstanceTools;
import weka_extras.classifiers.ensembles.SaveableEnsemble;
import weka.core.Instances;
import timeseriesweka.classifiers.TrainAccuracyEstimator;

import experiments.Experiments.ExperimentalArguments;

import static utilities.GenericTools.indexOfMax;
import static utilities.InstanceTools.truncateInstance;
import static utilities.InstanceTools.truncateInstances;
import static utilities.Utilities.argMax;

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
 directly, instead of building the String[] args and calling main like a lot of legacy code does.
 *
 * @author James Large (james.large@uea.ac.uk), Tony Bagnall (anthony.bagnall@uea.ac.uk)
 */
public class ExperimentsEarlyClassification  {

    private final static Logger LOGGER = Logger.getLogger(Experiments.class.getName());

    public static boolean debug = false;

    //A few 'should be final but leaving them not final just in case' public static settings
    public static int numCVFolds = 10;

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
        System.out.println("Raw args:");
        for (String str : args)
            System.out.println("\t"+str);
        System.out.println("");

        if (args.length > 0) {
            ExperimentalArguments expSettings = new ExperimentalArguments(args);
            setupAndRunExperiment(expSettings);
        }else{
            int folds=1;
            boolean threaded=false;
            if(threaded){
                String[] settings=new String[6];
                settings[0]="-dp=E:/Data/TSCProblems2018/";//Where to get data
                settings[1]="-rp=E:/Results/";//Where to write results
                settings[2]="-gtf=true"; //Whether to generate train files or not
                settings[3]="-cn=RISE"; //Classifier name
                settings[5]="1";
                settings[4]="-dn="+"ItalyPowerDemand"; //Problem file
                settings[5]="-f=1";//Fold number (fold number 1 is stored as testFold0.csv, its a cluster thing)
                ExperimentalArguments expSettings = new ExperimentalArguments(settings);
                setupAndRunMultipleExperimentsThreaded(expSettings, new String[]{settings[3]},DatasetLists.tscProblems78,0,folds);
            }else{//Local run without args, mainly for debugging
                String[] settings=new String[6];
//Location of data set
                settings[0]="-dp=E:/Data/TSCProblems2018/";//Where to get data
                settings[1]="-rp=E:/Results/";//Where to write results
                settings[2]="-gtf=false"; //Whether to generate train files or not
                settings[3]="-cn=TunedTSF"; //Classifier name
//                for(String str:DatasetLists.tscProblems78){
                settings[4]="-dn="+"ItalyPowerDemand"; //Problem file
                settings[5]="-f=2";//Fold number (fold number 1 is stored as testFold0.csv, its a cluster thing)
                System.out.println("Manually set args:");
                for (String str : settings)
                    System.out.println("\t"+settings);
                System.out.println("");

                ExperimentalArguments expSettings = new ExperimentalArguments(settings);
                setupAndRunExperiment(expSettings);
//                }
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
    public static void setupAndRunExperiment(ExperimentalArguments expSettings) throws Exception {
        //todo: when we convert to e.g argparse4j for parameter passing, add a para
        //for location to log to file as well. for now, assuming console output is good enough
        //for local running, and cluster output files are good enough on there.
//        LOGGER.addHandler(new FileHandler());
        if (debug)
            LOGGER.setLevel(Level.FINEST);
        else
            LOGGER.setLevel(Level.INFO);
        DatasetLoading.setDebug(false); //TODO when we got full enterprise and figure out how to properly do logging, clean this up
        LOGGER.log(Level.FINE, expSettings.toString());

        //TODO still setting these for now, since maybe certain classfiiers still use these "global"
        //paths. would rather just use the expSettings to do it all though
        DatasetLists.resultsPath = expSettings.resultsWriteLocation;
        experiments.data.DatasetLists.problemPath = expSettings.dataReadLocation;

        //2019_06_03: cases in the classifier can now change the classifier name to reflect
        //paritcular parameters wanting to be represented as different classifiers
        //e.g. a case ShapletsContracted might take a contract time (e.g. 1 day) from the args and set up the
        //shapelet transform, but also change the classifier name stored in the experimentalargs to e.g. Shapelets_1day
        //such that if the experimenter is looping over contract times, they need only create one case
        //in the setclassifier switch and pass one classifier name, but loop over contract time directly
        //
        //so, the setClassifier has been moved to up here, previously only done after the check for
        //whether we abort due to the results file already existing. the instantiation of a classifier
        //shouldn't be too much work, so despite it looking a little ugly, the call is
        //moved to here before the first proper usage of classifiername, such that it can
        //be updated first if need be
        Classifier classifier = ClassifierLists.setClassifier(expSettings);

        //Build/make the directory to write the train and/or testFold files to
        String fullWriteLocation = expSettings.resultsWriteLocation + expSettings.classifierName + "/Predictions/" + expSettings.datasetName + "/";
        File f = new File(fullWriteLocation);
        if (!f.exists())
            f.mkdirs();

        String targetFileName = fullWriteLocation + "testFold" + expSettings.foldId + ".csv";

        //Check whether fold already exists, if so, dont do it, just quit
        if (!expSettings.forceEvaluation && experiments.CollateResults.validateSingleFoldFile(targetFileName)) {
            LOGGER.log(Level.INFO, expSettings.toShortString() + " already exists at "+targetFileName+", exiting.");
            return;
        }
        else {
            Instances[] data = DatasetLoading.sampleDataset(expSettings.dataReadLocation, expSettings.datasetName, expSettings.foldId);

            //If needed, build/make the directory to write the train and/or testFold files to
            if (expSettings.supportingFilePath == null || expSettings.supportingFilePath.equals(""))
                expSettings.supportingFilePath = fullWriteLocation;

            ///////////// 02/04/2019 jamesl to be put back in in place of above when interface redesign finished.
            // default builds a foldx/ dir in normal write dir
//            if (expSettings.supportingFilePath == null || expSettings.supportingFilePath.equals(""))
//                expSettings.supportingFilePath = fullWriteLocation + "fold" + expSettings.foldId + "/";
//            if (classifier instanceof FileProducer) {
//                f = new File(expSettings.supportingFilePath);
//                if (!f.exists())
//                    f.mkdirs();
//            }

            //If this is to be a single _parameter_ evaluation of a fold, check whether this exists, and again quit if it does.
            if (expSettings.singleParameterID != null && classifier instanceof ParameterSplittable) {
                expSettings.checkpointing = false; //Just to tie up loose ends in case user defines both checkpointing AND para splitting

                targetFileName = fullWriteLocation + "fold" + expSettings.foldId + "_" + expSettings.singleParameterID + ".csv";
                if (experiments.CollateResults.validateSingleFoldFile(targetFileName)) {
                    LOGGER.log(Level.INFO, expSettings.toShortString() + ", parameter " + expSettings.singleParameterID +", already exists at "+targetFileName+", exiting.");
                    return;
                }
            }

            double acc = runExperiment(expSettings, data[0], data[1], classifier, fullWriteLocation);
            LOGGER.log(Level.INFO, "Experiment finished " + expSettings.toShortString() + ", Test Acc:" + acc);
        }
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
     * @return the accuracy of c on fold for problem given in train/test, or -1 on an error
     */
    public static double runExperiment(ExperimentalArguments expSettings, Instances trainSet, Instances testSet, Classifier classifier, String resultsPath) {

        //if this is a parameter split run, train file name is defined by this
        //otherwise generally if the classifier wants to save parameter info itnerally, set that up here too
        String trainFoldFilename = setupParameterSavingInfo(expSettings, classifier, trainSet, resultsPath);
        if (trainFoldFilename == null)
            //otherwise, defined by this as default
            trainFoldFilename = "trainFold" + expSettings.foldId + ".csv";
        String testFoldFilename = "testFold" + expSettings.foldId + ".csv";

        ClassifierResults trainResults = null;
        ClassifierResults testResults = null;

        LOGGER.log(Level.FINE, "Preamble complete, real experiment starting.");

        try {
            //Setup train results
            if (expSettings.generateErrorEstimateOnTrainSet)
                trainResults = findOrSetUpTrainEstimate(expSettings, classifier, trainSet, expSettings.foldId, resultsPath + trainFoldFilename);
            LOGGER.log(Level.FINE, "Train estimate ready.");


            //Build on the full train data here
            long buildTime = System.nanoTime();
            classifier.buildClassifier(trainSet);
            buildTime = System.nanoTime() - buildTime;
            LOGGER.log(Level.FINE, "Training complete");

            if (expSettings.serialiseTrainedClassifier && classifier instanceof Serializable)
                serialiseClassifier(expSettings, classifier);

            //Write train results
            if (expSettings.generateErrorEstimateOnTrainSet) {
                if (!(classifier instanceof TrainAccuracyEstimator)) {
                    assert(trainResults.getTimeUnit().equals(TimeUnit.NANOSECONDS)); //should have been set as nanos in the crossvalidation
                    trainResults.turnOffZeroTimingsErrors();
                    trainResults.setBuildTime(buildTime);
                    writeResults(expSettings, classifier, trainResults, resultsPath + trainFoldFilename, "train");
                }
                //else
                //   the classifier will have written it's own train estimate internally via TrainAccuracyEstimate
            }
            LOGGER.log(Level.FINE, "Train estimate written");


            //And now evaluate on the test set, if this wasn't a single parameter fold
            if (expSettings.singleParameterID == null) {
                //This is checked before the buildClassifier also, but
                //a) another process may have been doing the same experiment
                //b) we have a special case for the file builder that copies the results over in buildClassifier (apparently?)
                //no reason not to check again
                if (expSettings.forceEvaluation || !CollateResults.validateSingleFoldFile(resultsPath + testFoldFilename)) {
                    long testBenchmark = findBenchmarkTime(expSettings);

                    testResults = evaluateClassifier(expSettings, classifier, testSet);
                    assert(testResults.getTimeUnit().equals(TimeUnit.NANOSECONDS)); //should have been set as nanos in the evaluation

                    testResults.turnOffZeroTimingsErrors();
                    testResults.setBenchmarkTime(testBenchmark);

                    if (classifier instanceof TrainAccuracyEstimator) {
                        //if this classifier is recording it's own results, use the build time it found
                        //this is because e.g ensembles that read from file (e.g cawpe) will calculate their build time
                        //as the sum of their modules' buildtime plus the time to define the ensemble prediction forming
                        //schemes. that is more accurate than what experiments would measure, which would in fact be
                        //the i/o time for reading in the modules' results, + the ensemble scheme time
                        //therefore the general assumption here is that the classifier knows its own buildtime
                        //better than we do here
                        testResults.setBuildTime(((TrainAccuracyEstimator)classifier).getTrainResults().getBuildTime());
                    }
                    else {
                        //else use the buildtime calculated here in experiments
                        testResults.setBuildTime(buildTime);
                    }

                    LOGGER.log(Level.FINE, "Testing complete");

                    writeResults(expSettings, classifier, testResults, resultsPath + testFoldFilename, "test");
                    LOGGER.log(Level.FINE, "Testing written");
                }
                else {
                    LOGGER.log(Level.INFO, "Test file already found, written by another process.");
                    testResults = new ClassifierResults(resultsPath + testFoldFilename);
                }
                return testResults.getAcc();
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
            parameterFileName = "fold" + expSettings.foldId + "_" + expSettings.singleParameterID + ".csv";
            expSettings.generateErrorEstimateOnTrainSet = true;
        }
        else {
            //Only do all this if not an internal _single parameter_ experiment
            // Save internal info for ensembles
            if (classifier instanceof SaveableEnsemble) {
                ((SaveableEnsemble) classifier).saveResults(resultsPath + "internalCV_" + expSettings.foldId + ".csv", resultsPath + "internalTestPreds_" + expSettings.foldId + ".csv");
            }
            if (expSettings.checkpointing && classifier instanceof SaveEachParameter) {
                ((SaveEachParameter) classifier).setPathToSaveParameters(resultsPath + "fold" + expSettings.foldId + "_");
            }
        }

        return parameterFileName;
    }

    private static ClassifierResults findOrSetUpTrainEstimate(ExperimentalArguments exp, Classifier classifier, Instances train, int fold, String fullTrainWritingPath) throws Exception {
        ClassifierResults trainResults = null;

        if (classifier instanceof TrainAccuracyEstimator) {
            //Classifier will perform cv internally while building, probably as part of a parameter search
            ((TrainAccuracyEstimator) classifier).writeTrainEstimatesToFile(fullTrainWritingPath);
            File f = new File(fullTrainWritingPath);
            if (f.exists())
                f.setWritable(true, false);
        }
        else {
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
        }

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
        ClassifierResults res = new ClassifierResults(testSet.numClasses());
        res.setTimeUnit(TimeUnit.NANOSECONDS);
        res.setClassifierName(classifier.getClass().getSimpleName());
        res.setDatasetName(testSet.relationName());
        res.setFoldID(exp.foldId);
        res.setSplit("test");

        int length = testSet.numAttributes()-1;
        Instances[] truncatedInstances = new Instances[20];
        truncatedInstances[19] = new Instances(testSet, 0);
        for (int i = 0; i < 19; i++) {
            int newLength = (int) Math.round((i + 1) * 0.05 * length);
            truncatedInstances[i] = truncateInstances(truncatedInstances[19], length, newLength);
        }

        res.turnOffZeroTimingsErrors();
        for (Instance testinst : testSet) {
            double trueClassVal = testinst.classValue();
            testinst.setClassMissing();

            long startTime = System.nanoTime();

            double[] dist = null;
            double earliness = 0;
            for (int i = 0; i < 20; i++){
                int newLength = (int)Math.round((i+1)*0.05 * length);
                Instance newInst = truncateInstance(testinst, length, newLength);
                newInst.setDataset(truncatedInstances[i]);

                dist = classifier.distributionForInstance(newInst);

                if (dist != null) {
                    earliness = newLength/(double)length;
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

    public static void writeResults(ExperimentalArguments exp, Classifier classifier, ClassifierResults results, String fullTestWritingPath, String split) throws Exception {
        results.setClassifierName(exp.classifierName);
        results.setDatasetName(exp.datasetName);
        results.setFoldID(exp.foldId);
        results.setSplit(split);
        results.setDescription("Generated by Experiments.java");

        if (classifier instanceof SaveParameterInfo)
            results.setParas(((SaveParameterInfo) classifier).getParameters());

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

    public static int[] defaultTimeStamps(int length){
        int[] ts = new int[20];
        for (int i = 0; i < 20; i++){
            ts[i] = (int)Math.round((i+1)*0.05 * length);
        }
        return ts;
    }
}
