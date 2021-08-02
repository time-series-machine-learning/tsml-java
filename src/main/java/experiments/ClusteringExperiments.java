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
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import weka.clusterers.Clusterer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

import static utilities.GenericTools.indexOfMax;

/**
 * The clustering experimental class of the tsml codebase. The 'main' method to run is
 setupAndRunExperiment(ExperimentalArguments expSettings)

 An execution of this will evaluate a single clusterer on a single resample of a single dataset.

 Given an ExperimentalArguments object, which may be parsed from command line arguments
 or constructed in code, (and in the future, perhaps other methods such as JSON files etc),
 will load the classifier and dataset specified, prep the location to write results to,
 train the classifier - potentially generating an error estimate via cross validation on the train set
 as well - and then predict the cases of the test set.

 The primary outputs are the train and/or 'testResampleX.csv' files
 *
 * @author James Large (james.large@uea.ac.uk), Tony Bagnall (anthony.bagnall@uea.ac.uk)
 */
public class ClusteringExperiments {

    private final static Logger LOGGER = Logger.getLogger(ClusteringExperiments.class.getName());
    public static boolean debug = false;
    private static boolean testFoldExists;
    private static boolean trainFoldExists;
    /*If true, experiments will not print or log to stdout/err anything other that exceptions (SEVERE)*/
    public static boolean beQuiet = false;
    private static final String WORKSPACE_DIR = "Workspace";
    private static final String PREDICTIONS_DIR = "Predictions";

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
            System.out.print("\n");
        }
        if (args.length > 0) {
            ExperimentalArguments expSettings = new ExperimentalArguments(args);
            setupAndRunExperiment(expSettings);
        }
        else {//Manually set args
            int folds=1;
            /* Change these settings for your experiment:*/
            String clusterer="KMeans";//Classifier name: See ClassifierLists for valid options
            ArrayList<String> parameters= new ArrayList<>();
            parameters.add("-dp=src\\main\\java\\experiments\\data\\tsc\\"); //Where to get datasets
            parameters.add("-rp=C:\\temp\\"); //Where to write results
            parameters.add("-gtf=false"); //Whether to generate train files or not
            parameters.add("-cn="+clusterer); //Classifier name
            parameters.add("-dn="); //Problem name, don't change here as it is overwritten by probFiles
            parameters.add("-f=1"); //Fold number (fold number 1 is stored as testFold0.csv, its a cluster thing)
            parameters.add("-d=true"); //Debugging
            parameters.add("--force=true"); //Overwrites existing results if true, otherwise set to false

            String[] settings=new String[parameters.size()];
            int count=0;
            for(String str:parameters)
                settings[count++]=str;
            String[] probFiles= {"UnitTest"}; //Problem name(s)

//            String[] probFiles= DatasetLists.fixedLengthMultivariate;
            /*
             * END OF SETTINGS
             */

            System.out.println("Manually set args:");
            for (String str : settings)
                System.out.println("\t"+str);
            System.out.println("");
           for (String prob:probFiles) {
                settings[4]="-dn="+prob;

                for(int i=1;i<=folds;i++) {
                    settings[5]="-f="+i;
                    ExperimentalArguments expSettings = new ExperimentalArguments(settings);
                    System.out.println("Sequential experiment with "+expSettings);
                    setupAndRunExperiment(expSettings);
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
     * 4) Constructs the clusterer
     * 5) Samples the dataset.
     * 6) If we're good to go, runs the experiment.
     */
    public static ClassifierResults[] setupAndRunExperiment(ExperimentalArguments expSettings) throws Exception {
        if (beQuiet)
            LOGGER.setLevel(Level.SEVERE); // only print severe things
        else {
            if (debug) LOGGER.setLevel(Level.FINEST); // print everything
            else       LOGGER.setLevel(Level.INFO); // print warnings, useful info etc, but not simple progress messages, e.g. 'training started'

            DatasetLoading.setDebug(debug);
        }
        LOGGER.log(Level.FINE, expSettings.toString());
        // if a pre-instantiated clusterer instance hasn't been supplied, generate one here
        if (expSettings.clusterer == null) {
            // if a classifier-generating-function has been given (typically in the case of bespoke classifiers wanted in threaded exps),
            // instantiate the classifier from that
                expSettings.clusterer = ClustererLists.setClusterer(expSettings);
        }
        buildExperimentDirectoriesAndFilenames(expSettings);
        //Check whether results already exists, if so and force evaluation is false: just quit
        if (quitEarlyDueToResultsExistence(expSettings))
            return null;

        Instances[] data = DatasetLoading.sampleDataset(expSettings.dataReadLocation, expSettings.datasetName, expSettings.foldId);
        setupClassifierExperimentalOptions(expSettings, expSettings.clusterer, data[0]);
        ClassifierResults[] results = runExperiment(expSettings, data[0], data[1], expSettings.clusterer);
        LOGGER.log(Level.INFO, "Experiment finished " + expSettings.toShortString() + ", Test Acc:" + results[1].getAcc());

        return results;
    }

    /**
     * Perform an actual experiment, using the loaded clusterer and resampled dataset given, writing to the specified results location.
     * 1) If needed, set up file paths and flags related to a single parameter evaluation and/or the clusterer's internal parameter saving things
     * 2) If we want to be performing cv to find an estimate of the error on the train set, either do that here or set up the clusterer to do it internally
     *          during buildClassifier()
     * 3) Do the actual training, i.e buildClassifier()
     * 4) Save information needed from the training, e.g. train estimates, serialising the clusterer, etc.
     * 5) Evaluate the trained clusterer on the test set
     * 6) Save test results
     * @return the classifierresults for this experiment, {train, test}
     */
    public static ClassifierResults[] runExperiment(ExperimentalArguments expSettings, Instances trainSet, Instances testSet, Clusterer clusterer) {
        ClassifierResults[] experimentResults; // the combined container, to hold { trainResults, testResults } on return

        LOGGER.log(Level.FINE, "Preamble complete, real experiment starting.");

        try {
            ClassifierResults trainResults = training(expSettings, clusterer, trainSet);
            ClassifierResults testResults = testing(expSettings, clusterer, testSet, trainResults);

            experimentResults = new ClassifierResults[] {trainResults, testResults};
        }
        catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Experiment failed. Settings: " + expSettings + "\n\nERROR: " + e.toString(), e);
            e.printStackTrace();
            return null; //error state
        }

        return experimentResults;
    }

    /**
     * Performs all operations related to training the clusterer, and returns a ClassifierResults object holding the results
     * of training.
     *
     * At minimum these results hold the hardware benchmark timing (if requested in expSettings), the memory used,
     * and the build time.
     *
     * If a train estimate is to be generated, the results also hold predictions and results from the train set, and these
     * results are written to file.
     */
    public static ClassifierResults training(ExperimentalArguments expSettings, Clusterer clusterer, Instances trainSet) throws Exception {
        ClassifierResults trainResults = new ClassifierResults();


        MemoryMonitor memoryMonitor = new MemoryMonitor();
        memoryMonitor.installMonitor();

        //Build on the full train data here
        long buildTime = System.nanoTime();
        //For now, just cloning the data and removing the class label
        Instances clsTrain = new Instances(trainSet);
        clsTrain.setClassIndex(-1);
        clsTrain.deleteAttributeAt(trainSet.classIndex());
        clusterer.buildClusterer(clsTrain);
        buildTime = System.nanoTime() - buildTime;
        LOGGER.log(Level.FINE, "Training complete");
        // Training done, collect memory monitor results
        GcFinalization.awaitFullGc();
        long maxMemory = memoryMonitor.getMaxMemoryUsed();

        if (!trainFoldExists || expSettings.forceEvaluation) {
//            trainResults = findExternalTrainEstimate(expSettings, clusterer, trainSet, expSettings.foldId);
            trainResults = evaluateClusterer(expSettings, clusterer, clsTrain,trainSet,"train", trainSet.numClasses());
            trainResults.setParas(trainResults.getParas());
            trainResults.turnOffZeroTimingsErrors();
            trainResults.setBenchmarkTime(trainResults.getTimeUnit().convert(trainResults.getBenchmarkTime(), trainResults.getTimeUnit()));
            trainResults.setBuildTime(trainResults.getTimeUnit().convert(trainResults.getBuildTime(), trainResults.getTimeUnit()));
            trainResults.turnOnZeroTimingsErrors();
            trainResults.setMemory(trainResults.getMemory());

            writeResults(expSettings, trainResults, expSettings.trainFoldFileName, "train");
            LOGGER.log(Level.FINE, "Train estimate written");
        }
        return trainResults;
    }


    /**
     * Performs all operations related to testing the clusterer, and returns a ClassifierResults object holding the results
     * of testing.
     *
     * Computational resource costs of the training process are taken from the train results.
     */
    public static ClassifierResults testing(ExperimentalArguments expSettings, Clusterer clusterer, Instances testSet, ClassifierResults trainResults) throws Exception {
        ClassifierResults testResults;

        if (expSettings.forceEvaluation || !CollateResults.validateSingleFoldFile(expSettings.testFoldFileName)) {
            Instances clsTest = new Instances(testSet);
            clsTest.setClassIndex(-1);
            clsTest.deleteAttributeAt(testSet.classIndex());

            testResults = evaluateClusterer(expSettings, clusterer, clsTest,testSet,"test",testSet.numClasses());
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
        return testResults;
    }


    /**
     * Based on experimental parameters passed, defines the target results file and workspace locations for use in the
     * rest of the experiment
     */
    public static void buildExperimentDirectoriesAndFilenames(ExperimentalArguments expSettings) {
        //Build/make the directory to write the train and/or testFold files to
        // [writeLoc]/[classifier]/Predictions/[dataset]/
        String fullWriteLocation = expSettings.resultsWriteLocation + expSettings.estimatorName + "/"+PREDICTIONS_DIR+"/" + expSettings.datasetName + "/";
        File f = new File(fullWriteLocation);
        if (!f.exists())
            f.mkdirs();

        expSettings.testFoldFileName = fullWriteLocation + "testResmaple" + expSettings.foldId + ".csv";
        expSettings.trainFoldFileName = fullWriteLocation + "trainResample" + expSettings.foldId + ".csv";
        testFoldExists = CollateResults.validateSingleFoldFile(expSettings.testFoldFileName);
        trainFoldExists = CollateResults.validateSingleFoldFile(expSettings.trainFoldFileName);

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

        if (!expSettings.forceEvaluation &&
                ((!expSettings.generateErrorEstimateOnTrainSet && testFoldExists) ||
                        (expSettings.generateErrorEstimateOnTrainSet && trainFoldExists  && testFoldExists))) {
            LOGGER.log(Level.INFO, expSettings.toShortString() + " already exists at " + expSettings.testFoldFileName + ", exiting.");
            quit = true;
        }

        return quit;
    }


    /**
     * Based on the experimental settings passed, make any clusterer interface calls that modify how the clusterer is TRAINED here,
     * e.g. give checkpointable classifiers the location to save, give contractable classifiers their contract, etc.
     *
     * @return If the clusterer is set up to evaluate a single parameter set on the train data, a new trainfilename shall be returned,
     *      otherwise null.
     *
     */
    private static String setupClassifierExperimentalOptions(ExperimentalArguments expSettings, Clusterer clusterer, Instances train) {
        String parameterFileName = null;

        if (clusterer instanceof Randomizable)
            ((Randomizable)clusterer).setSeed(expSettings.foldId);
        return parameterFileName;
    }

    /**
     * Meta info shall be set by writeResults(...), just generating the prediction info and
     * any info directly calculable from that here
     */
    public static ClassifierResults evaluateClusterer(ExperimentalArguments exp, Clusterer clusterer, Instances clusterData, Instances fullData, String trainOrTest, int numClasses) throws Exception {
        ClassifierResults res = new ClassifierResults(numClasses);
        res.setTimeUnit(TimeUnit.NANOSECONDS);
        res.setClassifierName(clusterer.getClass().getSimpleName());
        res.setDatasetName(clusterData.relationName());
        res.setFoldID(exp.foldId);
        res.setSplit(trainOrTest);
        res.turnOffZeroTimingsErrors();
        for(int i=0;i<clusterData.numInstances();i++) {
            Instance testinst=clusterData.instance(i);
            double trueClassVal = fullData.instance(i).classValue();
            long startTime = System.nanoTime();
            double[] dist = clusterer.distributionForInstance(testinst);
            long predTime = System.nanoTime() - startTime;

            res.addPrediction(trueClassVal, dist, indexOfMax(dist), predTime, "");
        }

        res.turnOnZeroTimingsErrors();
        res.finaliseResults();
        res.findAllStatsOnce();

        return res;
    }

    public static String buildExperimentDescription() {
        Date date = new Date();
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

        StringBuilder sb = new StringBuilder("Generated by ClassificationExperiments.java on " + formatter.format(date) + ".");

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
}
