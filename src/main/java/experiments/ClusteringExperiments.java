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
import evaluation.storage.ClustererResults;
import evaluation.storage.ClustererResults;
import experiments.data.DatasetLoading;
import tsml.clusterers.EnhancedAbstractClusterer;
import utilities.ClusteringUtilities;
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
 will load the clusterer and dataset specified, prep the location to write results to,
 train the clusterer - potentially generating an error estimate via cross validation on the train set
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
            int folds=30;
            /* Change these settings for your experiment:*/
            String clusterer="KMeans";//Clusterer name: See ClustererLists for valid options
            ArrayList<String> parameters= new ArrayList<>();
            parameters.add("-dp=src\\main\\java\\experiments\\data\\tsc\\"); //Where to get datasets
            parameters.add("-rp=temp\\"); //Where to write results
            parameters.add("-gtf=false"); //Whether to generate train files or not
            parameters.add("-cn="+clusterer); //Clusterer name
            parameters.add("-dn="); //Problem name, don't change here as it is overwritten by probFiles
            parameters.add("-f=1"); //Resample number (resample number 1 is stored as testResample0.csv, its a cluster thing)
            parameters.add("-d=true"); //Debugging
            parameters.add("--force=true"); //Overwrites existing results if true, otherwise set to false

            String[] settings=new String[parameters.size()];
            int count=0;
            for(String str:parameters)
                settings[count++]=str;
            String[] probFiles= {"ChinaTown"}; //Problem name(s)

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
     * you have a bespoke clusterer not handled by ClustererList or dataset that
     * is sampled in a bespoke way, use runExperiment
     *
     * 1) Sets up the logger.
     * 2) Sets up the results write path
     * 3) Checks whether this experiments results already exist. If so, exit
     * 4) Constructs the clusterer
     * 5) Samples the dataset.
     * 6) If we're good to go, runs the experiment.
     */
    public static ClustererResults[] setupAndRunExperiment(ExperimentalArguments expSettings) throws Exception {
        if (beQuiet)
            LOGGER.setLevel(Level.SEVERE); // only print severe things
        else {
            if (debug) LOGGER.setLevel(Level.FINEST); // print everything
            else       LOGGER.setLevel(Level.INFO); // print warnings, useful info etc, but not simple progress messages, e.g. 'training started'

            DatasetLoading.setDebug(debug);
        }
        LOGGER.log(Level.FINE, expSettings.toString());

        buildExperimentDirectoriesAndFilenames(expSettings);
        //Check whether results already exists, if so and force evaluation is false: just quit
        if (quitEarlyDueToResultsExistence(expSettings))
            return null;

        Instances[] data = DatasetLoading.sampleDataset(expSettings.dataReadLocation, expSettings.datasetName, expSettings.foldId);
        expSettings.numClassValues = data[0].numClasses();

        // if a pre-instantiated clusterer instance hasn't been supplied, generate one here using setClusterer
        if (expSettings.clusterer == null) {
            expSettings.clusterer = ClustererLists.setClusterer(expSettings);
        }

        setupClustererExperimentalOptions(expSettings, expSettings.clusterer);
        ClustererResults[] results = runExperiment(expSettings, data[0], data[1], expSettings.clusterer);
        LOGGER.log(Level.INFO, "Experiment finished " + expSettings.toShortString());

        return results;
    }

    /**
     * Perform an actual experiment, using the loaded clusterer and resampled dataset given, writing to the specified results location.
     * 1) If needed, set up file paths and flags related to a single parameter evaluation and/or the clusterer's internal parameter saving things
     * 2) If we want to be performing cv to find an estimate of the error on the train set, either do that here or set up the clusterer to do it internally
     *          during buildClusterer()
     * 3) Do the actual training, i.e buildClusterer()
     * 4) Save information needed from the training, e.g. train estimates, serialising the clusterer, etc.
     * 5) Evaluate the trained clusterer on the test set
     * 6) Save test results
     * @return the ClustererResults for this experiment, {train, test}
     */
    public static ClustererResults[] runExperiment(ExperimentalArguments expSettings, Instances trainSet, Instances testSet, Clusterer clusterer) {
        ClustererResults[] experimentResults; // the combined container, to hold { trainResults, testResults } on return

        LOGGER.log(Level.FINE, "Preamble complete, real experiment starting.");

        try {
            //Since we are copying train and test data, no need to copy it again
            if (clusterer instanceof EnhancedAbstractClusterer)
                ((EnhancedAbstractClusterer)clusterer).setCopyInstances(false);

            ClustererResults trainResults = training(expSettings, clusterer, trainSet);
            ClustererResults testResults = testing(expSettings, clusterer, testSet, trainResults);

            experimentResults = new ClustererResults[] {trainResults, testResults};
        }
        catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Experiment failed. Settings: " + expSettings + "\n\nERROR: " + e, e);
            e.printStackTrace();
            return null; //error state
        }

        return experimentResults;
    }

    /**
     * Performs all operations related to training the clusterer, and returns a ClustererResults object holding the results
     * of training.
     *
     * At minimum these results hold the hardware benchmark timing (if requested in expSettings), the memory used,
     * and the build time.
     *
     * If a train estimate is to be generated, the results also hold predictions and results from the train set, and these
     * results are written to file.
     */
    public static ClustererResults training(ExperimentalArguments expSettings, Clusterer clusterer, Instances trainSet) throws Exception {
        //For now, just cloning the data and removing the class label
        Instances clsTrain = new Instances(trainSet);
        clsTrain.setClassIndex(-1);
        if (trainSet.classIndex() >= 0)
            clsTrain.deleteAttributeAt(trainSet.classIndex());

        long benchmark = ClassifierExperiments.findBenchmarkTime(expSettings);

        MemoryMonitor memoryMonitor = new MemoryMonitor();
        memoryMonitor.installMonitor();

        //Build on the full train data here
        long buildTime = System.nanoTime();
        clusterer.buildClusterer(clsTrain);
        buildTime = System.nanoTime() - buildTime;
        LOGGER.log(Level.FINE, "Training complete");

        // Training done, collect memory monitor results
        // Need to wait for an update, otherwise very quick clusterers may not experience gc calls during training,
        // or the monitor may not update in time before collecting the max
        GcFinalization.awaitFullGc();
        long maxMemory = memoryMonitor.getMaxMemoryUsed();

        ClustererResults trainResults;
        if (!trainFoldExists || expSettings.forceEvaluation) {
            if (clusterer instanceof EnhancedAbstractClusterer)
                trainResults = ClusteringUtilities.getClusteringResults((EnhancedAbstractClusterer)clusterer, trainSet);
            else
                trainResults = evaluateClusterer(clusterer, clsTrain, trainSet, trainSet.numClasses());

            trainResults.setBenchmarkTime(benchmark);
            trainResults.setBuildTime(buildTime);
            trainResults.setMemory(maxMemory);

            writeResults(expSettings, trainResults, expSettings.trainFoldFileName, "train");
            LOGGER.log(Level.FINE, "Train estimate written");
        }
        else{
            trainResults = new ClustererResults(trainSet.numClasses());
            trainResults.setBenchmarkTime(benchmark);
            trainResults.setBuildTime(buildTime);
            trainResults.setMemory(maxMemory);
        }

        return trainResults;
    }

    /**
     * Performs all operations related to testing the clusterer, and returns a ClustererResults object holding the results
     * of testing.
     *
     * Computational resource costs of the training process are taken from the train results.
     */
    public static ClustererResults testing(ExperimentalArguments expSettings, Clusterer clusterer, Instances testSet, ClustererResults trainResults) throws Exception {
        ClustererResults testResults;
        if (expSettings.forceEvaluation || !CollateResults.validateSingleFoldFile(expSettings.testFoldFileName)) {
            Instances clsTest = new Instances(testSet);
            clsTest.setClassIndex(-1);
            if (testSet.classIndex() >= 0)
                clsTest.deleteAttributeAt(testSet.classIndex());

            testResults = evaluateClusterer(clusterer, clsTest, testSet, testSet.numClasses());
            testResults.setBenchmarkTime(trainResults.getBenchmarkTime());
            testResults.setBuildTime(trainResults.getBuildTime());
            testResults.setMemory(trainResults.getMemory());

            LOGGER.log(Level.FINE, "Testing complete");

            writeResults(expSettings, testResults, expSettings.testFoldFileName, "test");
            LOGGER.log(Level.FINE, "Test results written");
        }
        else {
            LOGGER.log(Level.INFO, "Test file already found, written by another process.");
            testResults = new ClustererResults(expSettings.testFoldFileName);
        }

        return testResults;
    }


    /**
     * Based on experimental parameters passed, defines the target results file and workspace locations for use in the
     * rest of the experiment
     */
    public static void buildExperimentDirectoriesAndFilenames(ExperimentalArguments expSettings) {
        //Build/make the directory to write the train and/or testFold files to
        // [writeLoc]/[clusterer]/Predictions/[dataset]/
        String fullWriteLocation = expSettings.resultsWriteLocation + expSettings.estimatorName + "/"
                +PREDICTIONS_DIR+"/" + expSettings.datasetName + "/";
        File f = new File(fullWriteLocation);
        if (!f.exists())
            f.mkdirs();

        expSettings.testFoldFileName = fullWriteLocation + "testResample" + expSettings.foldId + ".csv";
        expSettings.trainFoldFileName = fullWriteLocation + "trainResample" + expSettings.foldId + ".csv";
        testFoldExists = CollateResults.validateSingleFoldFile(expSettings.testFoldFileName);
        trainFoldExists = CollateResults.validateSingleFoldFile(expSettings.trainFoldFileName);

        if (expSettings.supportingFilePath == null || expSettings.supportingFilePath.equals(""))
            expSettings.supportingFilePath = expSettings.resultsWriteLocation + expSettings.estimatorName + "/"
                    +WORKSPACE_DIR+"/" + expSettings.datasetName + "/";

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

        if (!expSettings.forceEvaluation && trainFoldExists && testFoldExists) {
            LOGGER.log(Level.INFO, expSettings.toShortString() + " already exists at write location, exiting.");
            quit = true;
        }

        return quit;
    }


    /**
     * Based on the experimental settings passed, make any clusterer interface calls that modify how the clusterer is TRAINED here,
     * e.g. give checkpointable clustererss the location to save, give contractable clustererss their contract, etc.
     */
    private static void setupClustererExperimentalOptions(ExperimentalArguments expSettings, Clusterer clusterer) {
        if (clusterer instanceof Randomizable)
            ((Randomizable)clusterer).setSeed(expSettings.foldId);
    }

    /**
     * Meta info shall be set by writeResults(...), just generating the prediction info and
     * any info directly calculable from that here
     */
    public static ClustererResults evaluateClusterer(Clusterer clusterer, Instances clusterData, Instances fullData, int numClasses) throws Exception {
        ClustererResults res = new ClustererResults(numClasses);

        for(int i = 0; i < clusterData.numInstances(); i++) {
            double trueClassVal = fullData.instance(i).classValue();
            long startTime = System.nanoTime();
            double[] dist = clusterer.distributionForInstance(clusterData.instance(i));
            long predTime = System.nanoTime() - startTime;
            res.addPrediction(trueClassVal, dist, indexOfMax(dist), predTime, "");
        }

        res.finaliseResults();
        return res;
    }

    public static String buildExperimentDescription() {
        Date date = new Date();
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

        StringBuilder sb = new StringBuilder("Generated by ClusteringExperiments.java on " + formatter.format(date) + ".");

        sb.append("    SYSTEMPROPERTIES:{");
        sb.append("user.name:").append(System.getProperty("user.name", "unknown"));
        sb.append(",os.arch:").append(System.getProperty("os.arch", "unknown"));
        sb.append(",os.name:").append(System.getProperty("os.name", "unknown"));
        sb.append("},ENDSYSTEMPROPERTIES");

        return sb.toString().replace("\n", "NEW_LINE");
    }

    public static void writeResults(ExperimentalArguments exp, ClustererResults results, String fullTestWritingPath, String split) throws Exception {
        results.setTimeUnit(TimeUnit.NANOSECONDS);
        results.setClustererName(exp.estimatorName);
        results.setDatasetName(exp.datasetName);
        results.setFoldID(exp.foldId);
        results.setSplit(split);
        results.setDescription(buildExperimentDescription());

        results.writeFullResultsToFile(fullTestWritingPath);

        File f = new File(fullTestWritingPath);
        if (f.exists()) {
            f.setWritable(true, false);
        }
    }
}
