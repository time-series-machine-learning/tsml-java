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

import blogspot.software_and_algorithms.stern_library.optimization.HungarianAlgorithm;
import fileIO.OutFile;
import utilities.DebugPrinting;
import utilities.GenericTools;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

import static org.apache.commons.math3.special.Gamma.logGamma;

/**
 * This is a container class for the storage of predictions and meta-info of a
 * clusterer on a single set of instances.
 * <p>
 * Predictions can be stored via addPrediction(...) or addAllPredictions(...)
 * Currently, the information stored about each prediction is:
 * - The true class value                            (double   getTrueClassValue(index))
 * - The predicted cluster                           (double   getPredClassValue(index))
 * - The probability distribution for this instance  (double[] getProbabilityDistribution(index))
 * - An optional description of the prediction       (String   getPredDescription(index))
 * <p>
 * The meta info stored is:
 * [LINE 1 OF FILE]
 * - get/setDatasetName(String)
 * - get/setClassifierName(String)
 * - get/setSplit(String)
 * - get/setFoldId(String)
 * - get/setTimeUnit(TimeUnit)
 * - get/setDescription(String)
 * [LINE 2 OF FILE]
 * - get/setParas(String)
 * [LINE 3 OF FILE]
 * - getAccuracy() (calculated from predictions, only settable with a suitably annoying message)
 * - get/setBuildTime(long)
 * - get/setTestTime(long)
 * - get/setBenchmarkTime(long)
 * - get/setMemory(long)
 * - get/setNumClasses(int)
 * - get/setNumClusters(int) (either set by user or indirectly found through predicted probability distributions)
 * <p>
 * [REMAINING LINES: PREDICTIONS]
 * - trueClassVal, predClusterVal, [empty], dist[0], dist[1] ... dist[c], [empty], predTime, [empty], predDescription
 * <p>
 * Supports reading/writing of results from/to file, in the 'ClustererResults file-format'
 * - loadResultsFromFile(String path)
 * - writeFullResultsToFile(String path)  (other writing formats also supported, write...ToFile(...)
 * <p>
 * Supports recording of timings in different time units. Nanoseconds is the default.
 * Also supports the calculation of various evaluative performance metrics  based on the predictions (accuracy,
 * rand index, mutual information etc.)
 * <p>
 * EXAMPLE USAGE:
 * ClustererResults res = new ClustererResults(numClasses);
 * //set a particular timeunit, if using something other than nanos. Nanos recommended
 * //set any meta info you want to keep, e.g classifiername, datasetname...
 * <p>
 * for (Instance inst : test) {
 *   res.addPrediction(inst.classValue(), clusterDist, clusterPred, 0, ""); //description is optional
 * }
 * <p>
 * res.finaliseResults(); //performs some basic validation, and calcs some relevant internal info
 * <p>
 * //can now find summary scores for these predictions
 * //stats stored in simple public members for now
 * res.findAllStats();
 * <p>
 * //and/or save to file
 * res.writeFullResultsToFile(path);
 * <p>
 * //and could then load them back in
 * ClassifierResults res2 = new ClassifierResults(path);
 * <p>
 * //the are automatically finalised, however the stats are not automatically found
 * res2.findAllStats();
 *
 * @author Matthew Middlehurst, adapted from ClassifierResults (James Large)
 */
public class ClustererResults implements DebugPrinting, Serializable {

    /**
     * Print a message with the filename to stdout when a file cannot be loaded.
     * Can get very tiresome if loading thousands of files with some expected failures,
     * and a higher level process already summarises them, thus this option to
     * turn off the messages
     */
    public static boolean printOnFailureToLoad = true;


    //LINE 1: meta info, set by user
    private String clustererName = "";
    private String datasetName = "";
    private String split = "";
    private int foldID = -1;
    private String description = ""; //human-friendly optional extra info if wanted.

//LINE 2: clusterer setup/info, parameters. precise format is up to user.

    /**
     * For now, user dependent on the formatting of this string, and really, the contents of it.
     * It is notionally intended to contain the parameters of the classifier used to produce the
     * attached predictions, but could also store other things as well.
     */
    private String paras = "No parameter info";

//LINE 3: rand, buildTime, memoryUsage
    //simple summarative performance stats.

    /**
     * Calculated from the stored cluster predictions, cannot be explicitly set by user
     */
    private double accuracy = -1;

    /**
     * Number of clusters, can be inferred from the number of distributions
     */
    private int numClusters = -1;

    /**
     * The time taken to complete buildClusterer(Instances), aka training. May be cumulative time over many parameter
     * set builds, etc It is assumed that the time given will be in the unit of measurement set by this object TimeUnit,
     * default nanoseconds. If no benchmark time is supplied, the default value is -1
     */
    private long buildTime = -1;

    /**
     * The cumulative prediction time, equal to the sum of the individual prediction times stored. Intended as a quick
     * helper/summary in case complete prediction information is not stored, and/or for a human reader to quickly
     * compare times.
     *
     * It is assumed that the time given will be in the unit of measurement set by this object TimeUnit,
     * default nanoseconds.
     * If no benchmark time is supplied, the default value is -1
     */
    private long testTime = -1;

    /**
     * The time taken to perform some standard benchmarking operation, to allow for a (not necessarily precise)
     * way to measure the general speed of the hardware that these results were made on, such that users
     * analysing the results may scale the timings in this file proportional to the benchmarks to get a consistent
     * relative scale across different results sets. It is up to the user what this benchmark operation is, and how
     * long it is (roughly) expected to take.
     * <p>
     * It is assumed that the time given will be in the unit of measurement set by this object TimeUnit, default
     * nanoseconds. If no benchmark time is supplied, the default value is -1
     */
    private long benchmarkTime = -1;

    /**
     * It is user dependent on exactly what this field means and how accurate it may be (because of Java's lazy gc).
     * Intended purpose would be the size of the model at the end of/after buildClusterer, aka the clusterer
     * has been trained.
     * <p>
     * The assumption, for now, is that this is measured in BYTES, but this is not enforced/ensured
     * If no memoryUsage value is supplied, the default value is -1
     */
    private long memoryUsage = -1;

    //REMAINDER OF THE FILE - 1 case per line
    //raw performance data. currently just five parallel arrays
    private ArrayList<Double> trueClassValues;
    private ArrayList<Double> clusterValues;
    private ArrayList<double[]> distributions;
    private ArrayList<Long> predTimes;
    private ArrayList<String> descriptions;

    //inferred/supplied dataset meta info
    private int numClasses;
    private int numInstances;

    //calculated performance metrics
    //accuracy can be re-calced, as well as stored on line three in files
    private double ri = -1;
    private double ari = -1;
    private double mi = -1;
    private double nmi = -1;
    private double ami = -1;


    /**
     * Consistent time unit ASSUMED across build times. Default to nanoseconds.
     * <p>
     * A long can contain 292 years worth of nanoseconds, which I assume to be enough for now.
     * Could be conceivable that the cumulative time of a large meta ensemble that is run
     * multi-threaded on a large dataset might exceed this.
     */
    private TimeUnit timeUnit = TimeUnit.NANOSECONDS;


    //self-management flags
    /**
     * essentially controls whether a ClustererResults object can have finaliseResults(trueClassVals)
     * called upon it. In theory, every class using the ClustererResults object should make new
     * instantiations of it each time a set of results is being computed, and so this is not needed
     */
    private boolean finalised = false;
    private boolean allStatsFound = false;

    /**
     * System.nanoTime() can STILL return zero on some tiny datasets with simple classifiers,
     * because it does not have enough precision. This flag, if true, will allow timings
     * of zero, under the partial assumption/understanding from the user that times under
     * ~200 nanoseconds can be equated to 0.
     */
    private boolean errorOnTimingOfZero = false;

    //functional getters to retrieve info from a clustererresults object, initialised/stored here for conveniance
    public static final Function<ClustererResults, Double> GETTER_Accuracy = (ClustererResults cr) -> cr.accuracy;
    public static final Function<ClustererResults, Double> GETTER_RandIndex = (ClustererResults cr) -> cr.ri;
    public static final Function<ClustererResults, Double> GETTER_AdjustedRandIndex = (ClustererResults cr) -> cr.ari;
    public static final Function<ClustererResults, Double> GETTER_MutualInformation = (ClustererResults cr) -> cr.mi;
    public static final Function<ClustererResults, Double> GETTER_NormalizedMutualInformation = (ClustererResults cr)
            -> cr.nmi;
    public static final Function<ClustererResults, Double> GETTER_AdjustedMutualInformation = (ClustererResults cr)
            -> cr.ami;

    public static final Function<ClustererResults, Long> GETTER_MemoryMB = (ClustererResults cr) ->
            cr.memoryUsage / 1000000L;
    public static final Function<ClustererResults, Long> GETTER_buildTime = (ClustererResults cr) -> cr.buildTime;
    public static final Function<ClustererResults, Long> GETTER_benchmarkTime = (ClustererResults cr) ->
            cr.benchmarkTime;
    public static final Function<ClustererResults, Long> GETTER_buildTimeBenchmarked =
            (ClustererResults cr) -> cr.benchmarkTime <= 0 ? cr.buildTime : cr.buildTime / cr.benchmarkTime;


    /*********************************
     *
     *       CONSTRUCTORS
     *
     */

    /**
     * Create an empty ClustererResults object.
     * <p>
     * If number of classes is known when making the object, it is safer to use this constructor
     * and supply the number of classes directly.
     * <p>
     * In some extreme use cases, predictions on dataset splits that a particular classifier results represents
     * may not have examples of each class that actually exists in the full dataset. If it is left
     * to infer the number of classes, some may be missing.
     */
    public ClustererResults(int numClasses) {
        trueClassValues = new ArrayList<>();
        clusterValues = new ArrayList<>();
        distributions = new ArrayList<>();
        predTimes = new ArrayList<>();
        descriptions = new ArrayList<>();

        this.numClasses = numClasses;
        finalised = false;
    }

    /**
     * Load a ClustererResults object from the file at the specified path
     */
    public ClustererResults(String filePathAndName) throws Exception {
        loadResultsFromFile(filePathAndName);
    }

    /**
     * Create a clusterer results object with complete predictions (equivalent to addAllPredictions()). The results are
     * FINALISED after initialisation. Meta info such as clusterer name, datasetname... can still be set after
     * construction.
     * <p>
     * The descriptions array argument may be null, in which case the descriptions are stored as empty strings.
     * <p>
     * All other arguments are required in full, however
     */
    public ClustererResults(int numClasses, double[] trueClassVals, double[] predictions, double[][] distributions,
                            long[] predTimes, String[] descriptions) throws Exception {
        this.trueClassValues = new ArrayList<>();
        this.clusterValues = new ArrayList<>();
        this.distributions = new ArrayList<>();
        this.predTimes = new ArrayList<>();
        this.descriptions = new ArrayList<>();

        this.numClasses = numClasses;

        addAllPredictions(trueClassVals, predictions, distributions, predTimes, descriptions);
        finaliseResults();
    }


    /***********************
     *
     *      DATASET META INFO
     *
     *
     */

    public int getNumClasses() {
        return numClasses;
    }

    public void setNumClasses(int numClasses) {
        this.numClasses = numClasses;
    }

    public int numInstances() {
        if (numInstances <= 0)
            inferNumInstances();
        return numInstances;
    }

    private void inferNumInstances() {
        this.numInstances = clusterValues.size();
    }

    public void turnOffZeroTimingsErrors() {
        errorOnTimingOfZero = false;
    }

    public void turnOnZeroTimingsErrors() {
        errorOnTimingOfZero = true;
    }


    /***************************
     *
     *   LINE 1 GETS/SETS
     *
     *  Just basic descriptive stuff, nothing fancy going on here
     *
     */

    public String getClustererName() {
        return clustererName;
    }

    public void setClustererName(String clustererName) {
        this.clustererName = clustererName;
    }

    public String getDatasetName() {
        return datasetName;
    }

    public void setDatasetName(String datasetName) {
        this.datasetName = datasetName;
    }

    public int getFoldID() {
        return foldID;
    }

    public void setFoldID(int foldID) {
        this.foldID = foldID;
    }

    /**
     * e.g "train", "test", "validation"
     */
    public String getSplit() { return split; }

    /**
     * e.g "train", "test", "validation"
     */
    public void setSplit(String split) { this.split = split; }


    public TimeUnit getTimeUnit() {
        return timeUnit;
    }

    public void setTimeUnit(TimeUnit timeUnit) {
        this.timeUnit = timeUnit;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }


    /*****************************
     *
     *     LINE 2 GETS/SETS
     *
     */


    public String getParas() {
        return paras;
    }

    public void setParas(String paras) {
        this.paras = paras;
    }


    /*****************************
     *
     *     LINE 3 GETS/SETS
     *
     */

    public double getAccuracy() {
        if (accuracy < 0)
            calculateAcc();
        return accuracy;
    }

    private void calculateAcc() {
        if (trueClassValues == null || trueClassValues.isEmpty() || trueClassValues.get(0) == -1) {
            System.out.println("**getAcc():calculateAcc() no true class values supplied yet, cannot calculate accuracy");
            return;
        }

        int d = Math.max(numClasses, numClusters);
        double[][] w = new double[d][d];
        for (int i = 0; i < numInstances; i++) {
            w[clusterValues.get(i).intValue()][trueClassValues.get(i).intValue()]++;
        }

        double max = 0;
        for (int i = 0; i < d; i++) {
            for (int n = 0; n < d; n++) {
                if (w[i][n] > max)
                    max = w[i][n];
            }
        }

        double[][] nw = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int n = 0; n < d; n++) {
                nw[i][n] = max - w[i][n];
            }
        }

        int[] a = new HungarianAlgorithm(nw).execute();

        double sum = 0;
        for (int i = 0; i < d; i++) {
            sum += w[i][a[i]];
        }

        accuracy = sum / numInstances;
    }

    public long getBuildTime() {
        return buildTime;
    }

    public long getBuildTimeInNanos() {
        return timeUnit.toNanos(buildTime);
    }

    /**
     * @throws Exception if buildTime is less than 1
     */
    public void setBuildTime(long buildTime) {
        if (errorOnTimingOfZero && buildTime < 1)
            throw new RuntimeException("Build time passed has invalid value, " + buildTime + ". If greater resolution" +
                    " is needed, use nano seconds (e.g System.nanoTime()) and set the TimeUnit of the " +
                    "classifierResults object to nanoseconds.\n\nIf you are using nanoseconds but STILL getting this " +
                    "error, read the javadoc for and use turnOffZeroTimingsErrors() for this call");
        this.buildTime = buildTime;
    }

    public long getTestTime() { return testTime; }

    public long getTestTimeInNanos() { return timeUnit.toNanos(testTime); }

    public void setTestTime(long testTime) {
        this.testTime = testTime;
    }

    public long getMemory() {
        return memoryUsage;
    }

    public void setMemory(long memory) {
        this.memoryUsage = memory;
    }

    public int getNumClusters() {
        if (numClusters <= 0)
            inferNumClusters();
        return numClusters;
    }

    public void setNumClusters(int numClusters) {
        this.numClusters = numClusters;
    }

    private void inferNumClusters() {
        this.numClusters = distributions.get(0).length;
    }

    public long getBenchmarkTime() {
        return benchmarkTime;
    }

    public void setBenchmarkTime(long benchmarkTime) {
        this.benchmarkTime = benchmarkTime;
    }


    /****************************
     *
     *    PREDICTION STORAGE
     *
     */

    /**
     * Will update the internal prediction info using the values passed. User must pass the predicted cluster
     * so that they may resolve ties how they want (e.g first, randomly, etc).
     * The standard, used in most places, would be utilities.GenericTools.indexOfMax(double[] dist)
     * <p>
     * The description argument may be null, however all other arguments are required in full
     * <p>
     * The true class is missing, however can be added in one go later with the
     * method finaliseResults(double[] trueClassVals)
     */
    public void addPrediction(double[] dist, double cluster, long predictionTime, String description) throws Exception {
        distributions.add(dist);
        clusterValues.add(cluster);
        predTimes.add(predictionTime);

        if (testTime == -1)
            testTime = predictionTime;
        else
            testTime += predictionTime;

        if (description == null)
            descriptions.add("");
        else
            descriptions.add(description);

        numInstances++;
    }

    /**
     * Will update the internal prediction info using the values passed. User must pass the predicted cluster
     * so that they may resolve ties how they want (e.g first, randomly, etc).
     * The standard, used in most places, would be utilities.GenericTools.indexOfMax(double[] dist)
     * <p>
     * The description argument may be null, however all other arguments are required in full
     */
    public void addPrediction(double trueClassVal, double[] dist, double cluster, long predictionTime,
                              String description) throws Exception {
        addPrediction(dist, cluster, predictionTime, description);
        trueClassValues.add(trueClassVal);
    }

    /**
     * Adds all the prediction info onto this ClustererResults object. Does NOT finalise the results,
     * such that (e.g) predictions from multiple dataset splits can be added to the same object if wanted
     * <p>
     * The description argument may be null, however all other arguments are required in full
     */
    public void addAllPredictions(double[] trueClassVals, double[] predictions, double[][] distributions,
                                  long[] predictionTimes, String[] descriptions) throws Exception {
        assert (trueClassVals.length == predictions.length);
        assert (trueClassVals.length == distributions.length);
        assert (trueClassVals.length == predictionTimes.length);

        if (descriptions != null)
            assert (trueClassVals.length == descriptions.length);

        for (int i = 0; i < trueClassVals.length; i++) {
            if (descriptions == null)
                addPrediction(trueClassVals[i], distributions[i], predictions[i], predictionTimes[i], null);
            else
                addPrediction(trueClassVals[i], distributions[i], predictions[i], predictionTimes[i], descriptions[i]);
        }
    }

    /**
     * Adds all the prediction info onto this ClustererResults object. Does NOT finalise the results,
     * such that (e.g) predictions from multiple dataset splits can be added to the same object if wanted
     * <p>
     * True class values can later be supplied (ALL IN ONE GO, if working to the above example usage..) using
     * finaliseResults(double[] testClassVals)
     * <p>
     * The description argument may be null, however all other arguments are required in full
     */
    public void addAllPredictions(double[] predictions, double[][] distributions, long[] predictionTimes,
                                  String[] descriptions) throws Exception {
        assert (predictions.length == distributions.length);
        assert (predictions.length == predictionTimes.length);

        if (descriptions != null)
            assert (predictions.length == descriptions.length);

        for (int i = 0; i < predictions.length; i++) {
            if (descriptions == null)
                addPrediction(distributions[i], predictions[i], predictionTimes[i], "");
            else
                addPrediction(distributions[i], predictions[i], predictionTimes[i], descriptions[i]);
        }
    }

    /**
     * Will perform some basic validation to make sure that everything is here
     * that is expected, and compute the accuracy etc ready for file writing.
     * <p>
     * Typical usage: results.finaliseResults(instances.attributeToDoubleArray(instances.classIndex()))
     */
    public void finaliseResults(double[] testClassVals) throws Exception {
        if (finalised) {
            System.out.println("finaliseResults(double[] testClassVals): Results already finalised, skipping " +
                    "re-finalisation");
            return;
        }

        if (testClassVals.length != clusterValues.size())
            throw new Exception("finaliseTestResults(double[] testClassVals): Number of predictions "
                    + "made and number of true class values passed do not match");

        trueClassValues = new ArrayList<>();
        for (double d : testClassVals)
            trueClassValues.add(d);

        finaliseResults();
    }


    /**
     * Will perform some basic validation to make sure that everything is here
     * that is expected, and compute the accuracy etc ready for file writing.
     * <p>
     * You can use this method, instead of the version that takes the double[] testClassVals
     * as an argument, if you've been storing predictions via the addPrediction overload
     * that takes the true class value of each prediction.
     */
    public void finaliseResults() throws Exception {
        if (finalised) {
            printlnDebug("finaliseResults(): Results already finalised, skipping re-finalisation");
            return;
        }

        if (numInstances <= 0)
            inferNumInstances();
        if (numClusters <= 0)
            inferNumClusters();

        if (distributions == null || clusterValues == null || distributions.isEmpty() || clusterValues.isEmpty())
            throw new Exception("finaliseTestResults(): no predictions stored for this module");

        assert trueClassValues.size() == clusterValues.size();

        calculateAcc();

        finalised = true;
    }


    /******************************
     *
     *          RAW DATA ACCESSORS
     *
     *     getAsList, getAsArray, and getSingleElement of the four lists describing predictions
     *
     */

    /**
     *
     */
    public ArrayList<Double> getTrueClassVals() {
        return trueClassValues;
    }

    public double[] getTrueClassValsAsArray() {
        double[] d = new double[trueClassValues.size()];
        int i = 0;
        for (double x : trueClassValues)
            d[i++] = x;
        return d;
    }

    public double getTrueClassValue(int index) {
        return trueClassValues.get(index);
    }


    public ArrayList<Double> getClusterValues() {
        return clusterValues;
    }

    public double[] getClusterValuesAsArray() {
        double[] d = new double[clusterValues.size()];
        int i = 0;
        for (double x : clusterValues)
            d[i++] = x;
        return d;
    }

    public double getClusterValue(int index) {
        return clusterValues.get(index);
    }

    public ArrayList<double[]> getProbabilityDistributions() {
        return distributions;
    }

    public double[][] getProbabilityDistributionsAsArray() {
        return distributions.toArray(new double[][]{});
    }

    public double[] getProbabilityDistribution(int i) {
        if (i < distributions.size())
            return distributions.get(i);
        return null;
    }

    public ArrayList<Long> getPredictionTimes() {
        return predTimes;
    }

    public long[] getPredictionTimesAsArray() {
        long[] l=new long[predTimes.size()];
        int i=0;
        for(long x:predTimes)
            l[i++]=x;
        return l;
    }

    public long getPredictionTime(int index) {
        return predTimes.get(index);
    }

    public long getPredictionTimeInNanos(int index) {
        return timeUnit.toNanos(getPredictionTime(index));
    }

    public ArrayList<String> getDescriptions() {
        return descriptions;
    }

    public String[] getPredDescriptionsAsArray() {
        String[] ds = new String[descriptions.size()];
        int i = 0;
        for (String d : descriptions)
            ds[i++] = d;
        return ds;
    }

    public String getPredDescription(int index) {
        return descriptions.get(index);
    }

    public void cleanPredictionInfo() {
        distributions = null;
        clusterValues = null;
        trueClassValues = null;
        predTimes = null;
        descriptions = null;
    }


    /********************************
     *
     *     FILE READ/WRITING
     *
     */

    public static boolean exists(File file) {
        return file.exists() && file.length() > 0;
    }

    public static boolean exists(String path) {
        return exists(new File(path));
    }

    /**
     * Reads and STORES the prediction in this ClustererResults object.
     * <p>
     * INCREMENTS NUMINSTANCES
     * <p>
     * If numClasses is still less than 0, WILL set numclasses if distribution info is present.
     * <p>
     * [true],[pred], ,[dist[0]],...,[dist[c]], ,[description until end of line, may have commas in it]
     */
    private void instancePredictionFromString(String predLine) throws Exception {
        String[] split = predLine.split(",");

        //collect actual class and cluster
        double trueClassVal = Double.parseDouble(split[0].trim());
        double clusterVal = Double.parseDouble(split[1].trim());

        final int distStartInd = 3; //actual, cluster, space, distStart
        double[] dist = null;
        if (numClusters < 2) {
            List<Double> distL = new ArrayList<>();
            for (int i = distStartInd; i < split.length; i++) {
                if (split[i].equals(""))
                    break; //we're at the empty-space-separator between probs and timing
                else
                    distL.add(Double.valueOf(split[i].trim()));
            }

            numClusters = distL.size();
            assert (numClusters >= 2);

            dist = new double[numClusters];
            for (int i = 0; i < numClusters; i++)
                dist[i] = distL.get(i);
        } else {
            //we know how many clusters there should be, use this as implicit
            //file verification
            dist = new double[numClusters];
            for (int i = 0; i < numClusters; i++) {
                //now need to offset by 3.
                dist[i] = Double.parseDouble(split[i + distStartInd].trim());
            }
        }

        //collect timings
        long predTime = -1;
        final int timingInd = distStartInd + numClusters + 1; //actual, predicted, space, dist, space, timing
        if (split.length > timingInd)
            predTime = Long.parseLong(split[timingInd].trim());

        //collect description
        String description = "";
        final int descriptionInd = timingInd + 2; //actual, predicted, space, dist, , space, timing, space, description
        if (split.length > descriptionInd) {
            description = split[descriptionInd];

            //no reason currently why the description passed cannot have commas in it,
            //might be a natural way to separate it in to different parts.
            //description really just fills up the remainder of the line.
            for (int i = descriptionInd + 1; i < split.length; i++)
                description += "," + split[i];
        }

        addPrediction(trueClassVal, dist, clusterVal, predTime, description);
    }

    private void instancePredictionsFromScanner(Scanner in) throws Exception {
        while (in.hasNext()) {
            String line = in.nextLine();
            //may be trailing empty lines at the end of the file
            if (line == null || line.equals(""))
                break;

            instancePredictionFromString(line);
        }

        calculateAcc();
    }

    /**
     * [true],[pred], ,[dist[0]],...,[dist[c]], ,[predTime], ,[description until end of line, may have commas in it]
     */
    private String instancePredictionToString(int i) {
        StringBuilder sb = new StringBuilder();

        sb.append(trueClassValues.get(i).intValue()).append(",");
        sb.append(clusterValues.get(i).intValue());

        //probs
        sb.append(","); //<empty space>
        double[] probs = distributions.get(i);
        for (double d : probs)
            sb.append(",").append(GenericTools.RESULTS_DECIMAL_FORMAT.format(d));

        //timing
        sb.append(",,").append(predTimes.get(i)); //<empty space>, timing

        //description
        sb.append(",,").append(descriptions.get(i)); //<empty space>, description

        return sb.toString();
    }

    public String instancePredictionsToString() throws Exception {
        if (trueClassValues == null || trueClassValues.size() == 0 || trueClassValues.get(0) == -1)
            throw new Exception("No true class value stored, call finaliseResults(double[] trueClassVal)");

        if (numInstances() > 0 && (distributions.size() == trueClassValues.size() && distributions.size() ==
                clusterValues.size())) {
            StringBuilder sb = new StringBuilder("");

            for (int i = 0; i < numInstances(); i++) {
                sb.append(instancePredictionToString(i));

                if (i < numInstances() - 1)
                    sb.append("\n");
            }

            return sb.toString();
        } else
            return "No Instance Prediction Information";
    }

    @Override
    public String toString() {
        return generateFirstLine();
    }

    public String statsToString() {
        String s = "";
        s += "Clustering Accuracy: " + accuracy;
        s += "\nRand Index: " + ri;
        s += "\nAdjusted Rand Index: " + ari;
        s += "\nMutual Information: " + mi;
        s += "\nNormalised Mutual Information: " + nmi;
        s += "\nAdjusted Mutual Information: " + ami;
        return s;
    }

    public String writeFullResultsToString() throws Exception {
        finaliseResults();

        StringBuilder st = new StringBuilder();
        st.append(generateFirstLine()).append("\n");
        st.append(generateSecondLine()).append("\n");
        st.append(generateThirdLine()).append("\n");

        st.append(instancePredictionsToString());
        return st.toString();
    }

    public void writeFullResultsToFile(String path) throws Exception {
        OutFile out = null;
        try {
            out = new OutFile(path);
            out.writeString(writeFullResultsToString());
        } catch (Exception e) {
            throw new Exception("Error writing results file.\n"
                    + "Outfile most likely didnt open successfully, probably directory doesnt exist yet.\n"
                    + "Path: " + path + "\nError: " + e);
        } finally {
            if (out != null)
                out.closeFile();
        }
    }

    private void parseFirstLine(String line) {
        String[] parts = line.split(",");
        if (parts.length == 0)
            return;

        datasetName = parts[0];
        clustererName = parts[1];
        split = parts[2];
        foldID = Integer.parseInt(parts[3]);
        setTimeUnitFromString(parts[4]);

        //nothing stopping the description from having its own commas in it, just read until end of line
        for (int i = 5; i < parts.length; i++)
            description += "," + parts[i];
    }

    private String generateFirstLine() {
        return datasetName + "," + clustererName + "," + split + "," + foldID + "," + getTimeUnitAsString() +
                "," + description;
    }

    private void parseSecondLine(String line) {
        paras = line;
    }

    private String generateSecondLine() {
        return paras;
    }

    /**
     * Returns the test acc reported on this line, for comparison with acc
     * computed later to assert they align. Accuracy has always been reported
     * on this line in this file format, so fair to assume if this fails
     * then the file is simply malformed
     */
    private double parseThirdLine(String line) {
        String[] parts = line.split(",");

        accuracy = Double.parseDouble(parts[0]);
        buildTime = Long.parseLong(parts[1]);
        testTime = Long.parseLong(parts[2]);
        benchmarkTime = Long.parseLong(parts[3]);
        memoryUsage = Long.parseLong(parts[4]);
        numClasses = Integer.parseInt(parts[5]);
        numClusters = Integer.parseInt(parts[6]);

        return accuracy;
    }

    private String generateThirdLine() {
        String res = accuracy
                + "," + buildTime
                + "," + testTime
                + "," + benchmarkTime
                + "," + memoryUsage
                + "," + getNumClasses()
                + "," + getNumClusters();

        return res;
    }

    private String getTimeUnitAsString() {
        return timeUnit.name();
    }

    private void setTimeUnitFromString(String str) {
        timeUnit = TimeUnit.valueOf(str);
    }

    public void loadResultsFromFile(String path) throws Exception {
        try {
            //init
            trueClassValues = new ArrayList<>();
            clusterValues = new ArrayList<>();
            distributions = new ArrayList<>();
            predTimes = new ArrayList<>();
            descriptions = new ArrayList<>();
            numInstances = 0;
            accuracy = -1;
            buildTime = -1;
            memoryUsage = -1;

            //check file exists
            File f = new File(path);
            if (!(f.exists() && f.length() > 0))
                throw new FileNotFoundException("File " + path + " NOT FOUND");

            Scanner inf = new Scanner(f);

            //parse meta infos
            parseFirstLine(inf.nextLine());
            parseSecondLine(inf.nextLine());
            double reportedTestAcc = parseThirdLine(inf.nextLine());

            //parse predictions
            instancePredictionsFromScanner(inf);

            //acts as a basic form of verification, does the acc reported on line 3 align with
            //the acc calculated while reading predictions
            double eps = 1.e-8;
            if (Math.abs(reportedTestAcc - accuracy) > eps) {
                throw new ArithmeticException("Calculated accuracy (" + accuracy + ") differs from written accuracy " +
                        "(" + reportedTestAcc + ") by more than eps (" + eps + "). File = " + path + ". numinstances = "
                        + numInstances + ". numClasses = " + numClasses);
            }

            finalised = true;
            inf.close();
        } catch (FileNotFoundException fnf) {
            if (printOnFailureToLoad)
                System.out.println("File " + path + " NOT FOUND");
            throw fnf;
        } catch (Exception ex) {
            if (printOnFailureToLoad)
                System.out.println("File " + path + " FAILED TO LOAD");
            throw ex;
        }
    }


    /******************************************
     *
     *   METRIC CALCULATIONS
     *
     */

    private double tp = 0, tn = 0, fn = 0, fp = 0;
    private boolean foundPairConfusionMatrix = false;

    private double[] classCounts, clusterCounts;
    private boolean foundCounts = false;

    private double[][] contingencyMatrix;
    private boolean foundContingencyMatrix = false;

    private double classEntropy, clusterEntropy;
    private boolean foundEntropy = false;

    /**
     * Will calculate all the metrics that can be found from the prediction information
     * stored in this object. Will NOT call finaliseResults(..), and finaliseResults(..)
     * not have been called elsewhere, however if it has not been called then true
     * class values must have been supplied while storing predictions.
     * <p>
     * This is to allow iterative calculation of the metrics (in e.g. batches
     * of added predictions)
     */
    public void findAllStats() {

        //meta info
        if (numInstances <= 0)
            inferNumInstances();
        if (numClusters <= 0)
            inferNumClusters();

        if (accuracy < 0)
            calculateAcc();

        findPairConfusionMatrix();
        findCounts();
        findContingencyMatrix();
        findEntropy();

        ri = findRI();
        ari = findARI();
        mi = findMI();
        nmi = findNMI();
        ami = findAMI();

        allStatsFound = true;
    }

    public double findRI() {
        if (!foundPairConfusionMatrix)
            findPairConfusionMatrix();
        return (tp + tn) / (tp + tn + fn + fp);
    }

    public double findARI() {
        if (!foundPairConfusionMatrix)
            findPairConfusionMatrix();
        return 2 * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn));
    }

    public double findMI() {
        if (!foundContingencyMatrix)
            findContingencyMatrix();

        if (!foundCounts)
            findCounts();

        double logNI = Math.log(numInstances);
        double sum = 0;
        for (int i = 0; i < numClusters; i++) {
            for (int n = 0; n < numClasses; n++) {
                if (contingencyMatrix[i][n] != 0) {
                    double a = contingencyMatrix[i][n] / numInstances;
                    sum += a * (Math.log(contingencyMatrix[i][n]) - logNI) +
                            a * (-Math.log(clusterCounts[i] * classCounts[n]) + logNI * 2);
                }
            }
        }

        if (sum < 0)
            return 0;
        return sum;
    }

    public double findNMI() {
        if (mi == -1)
            mi = findMI();

        if (!foundEntropy)
            findEntropy();

        double norm = (classEntropy + clusterEntropy) / 2;
        return mi / norm;
    }

    public double findAMI() {
        if (mi == -1)
            mi = findMI();

        if (!foundEntropy)
            findEntropy();

        //expected mutual information

        double max = 0;
        double[] logClassCounts = new double[classCounts.length];
        double[] glnClass = new double[classCounts.length];
        double[] glnNClass = new double[classCounts.length];
        for (int i = 0; i < classCounts.length; i++) {
            if (classCounts[i] > max)
                max = classCounts[i];
            logClassCounts[i] = Math.log(classCounts[i]);
            glnClass[i] = logGamma(classCounts[i] + 1);
            glnNClass[i] = logGamma(numInstances - classCounts[i] + 1);
        }

        double[] logClusterCounts = new double[clusterCounts.length];
        double[] glnCluster = new double[clusterCounts.length];
        double[] glnNCluster = new double[clusterCounts.length];
        for (int i = 0; i < clusterCounts.length; i++) {
            if (clusterCounts[i] > max)
                max = clusterCounts[i];
            logClusterCounts[i] = Math.log(clusterCounts[i]);
            glnCluster[i] = logGamma(clusterCounts[i] + 1);
            glnNCluster[i] = logGamma(numInstances - clusterCounts[i] + 1);
        }

        double logNI = Math.log(numInstances);
        double[] nijs = new double[(int) max + 1];
        double[] logNnij = new double[nijs.length];
        double[] term1 = new double[nijs.length];
        double[] glnNij = new double[nijs.length];
        for (int i = 0; i < nijs.length; i++) {
            nijs[i] = i;
            logNnij[i] = logNI + Math.log(i);
            term1[i] = (double) i / numInstances;
            glnNij[i] = logGamma(i + 1);
        }

        double glnN = logGamma(numInstances + 1);

        int[][] start = new int[numClusters][numClasses];
        int[][] end = new int[numClusters][numClasses];
        for (int i = 0; i < numClusters; i++) {
            for (int n = 0; n < numClasses; n++) {
                double v = classCounts[n] - numInstances + clusterCounts[i];
                start[i][n] = (int) Math.max(v, 1);
                end[i][n] = (int) Math.min(clusterCounts[i], classCounts[n]) + 1;
            }
        }

        double emi = 0;
        for (int i = 0; i < numClusters; i++) {
            for (int n = 0; n < numClasses; n++) {
                for (int j = start[i][n]; j < end[i][n]; j++) {
                    double term2 = logNnij[j] - logClassCounts[n] - logClusterCounts[i];
                    double a = logGamma(classCounts[n] - j + 1);
                    double b = logGamma(clusterCounts[i] - j + 1);
                    double c = logGamma(numInstances - classCounts[n] - clusterCounts[i] + j + 1);
                    double gln = glnClass[n] + glnCluster[i] + glnNClass[n] + glnNCluster[i] - glnN - glnNij[j]
                            - a - b - c;
                    double term3 = Math.exp(gln);

                    emi += term1[j] * term2 * term3;
                }
            }
        }

        double norm = (classEntropy + clusterEntropy) / 2 - emi;
        if (norm == 0)
            norm = Double.MIN_VALUE;

        return (mi - emi) / norm;
    }

    private void findPairConfusionMatrix() {
        for (int i = 0; i < numInstances; i++) {
            for (int n = 0; n < numInstances; n++) {
                if (i == n)
                    continue;

                if (clusterValues.get(i).equals(clusterValues.get(n)) &&
                        trueClassValues.get(i).equals(trueClassValues.get(n))) {
                    tp++;
                } else if (!clusterValues.get(i).equals(clusterValues.get(n)) &&
                        !trueClassValues.get(i).equals(trueClassValues.get(n))) {
                    tn++;
                } else if (clusterValues.get(i).equals(clusterValues.get(n)) &&
                        !trueClassValues.get(i).equals(trueClassValues.get(n))) {
                    fn++;
                } else {
                    fp++;
                }
            }
        }

        foundPairConfusionMatrix = true;
    }

    private void findCounts() {
        classCounts = new double[numClasses];
        clusterCounts = new double[numClusters];

        for (int i = 0; i < numInstances; i++) {
            classCounts[trueClassValues.get(i).intValue()]++;
            clusterCounts[clusterValues.get(i).intValue()]++;
        }

        foundCounts = true;
    }

    private void findContingencyMatrix() {
        contingencyMatrix = new double[numClusters][numClasses];
        for (int i = 0; i < numInstances; i++) {
            contingencyMatrix[clusterValues.get(i).intValue()][trueClassValues.get(i).intValue()]++;
        }

        foundContingencyMatrix = true;
    }

    private void findEntropy() {
        if (!foundCounts)
            findCounts();

        classEntropy = entropy(classCounts);
        clusterEntropy = entropy(clusterCounts);

        foundEntropy = true;
    }

    private double entropy(double[] arr) {
        double x = 0;
        double logNI = Math.log(numInstances);
        for (double p : arr) {
            x -= p > 0 ? (p / numInstances) * (Math.log(p) - logNI) : 0;
        }
        return x;
    }


    /**
     * Will calculate all the metrics that can be found from the prediction information
     * stored in this object, UNLESS this object has been finalised (finaliseResults(..)) AND
     * has already had it's stats found (findAllStats()), e.g. if it has already been called
     * by another process.
     * <p>
     * In this latter case, this method does nothing.
     */
    public void findAllStatsOnce() {
        if (finalised && allStatsFound) {
            printlnDebug("Stats already found, ignoring findAllStatsOnce()");
            return;
        } else {
            findAllStats();
        }
    }


    /**
     * Concatenates the predictions of clusterers made on different folds on the data
     * into one results object
     * <p>
     * If ClustererResults ever gets split into separate classes for prediction and meta info,
     * this obviously gets cleaned up a lot
     *
     * @param cresults ClustererResults[fold]
     * @return single ClustererResults object
     */
    public static ClustererResults concatenateClustererResults( /*fold*/ ClustererResults[] cresults) throws Exception {
        return concatenateClustererResults(new ClustererResults[][]{cresults})[0];
    }

    /**
     * Concatenates the predictions of clusterers made on different folds on the data
     * into one results object per clusterer.
     * <p>
     * If ClustererResults ever gets split into separate classes for prediction and meta info,
     * this obviously gets cleaned up a lot
     *
     * @param cresults ClustererResults[clusterer][fold]
     * @return ClustererResults[clusterer]
     */
    public static ClustererResults[] concatenateClustererResults( /*clusterer*/ /*fold*/ ClustererResults[][] cresults)
            throws Exception {
        ClustererResults[] concatenatedResults = new ClustererResults[cresults.length];
        for (int classifierid = 0; classifierid < cresults.length; classifierid++) {
            if (cresults[classifierid].length == 1) {
                concatenatedResults[classifierid] = cresults[classifierid][0];
            } else {
                ClustererResults newCres = new ClustererResults(cresults[classifierid][0].numClasses);
                for (int foldid = 0; foldid < cresults[classifierid].length; foldid++) {
                    ClustererResults foldCres = cresults[classifierid][foldid];
                    for (int predid = 0; predid < foldCres.numInstances(); predid++) {
                        newCres.addPrediction(foldCres.getTrueClassValue(predid),
                                foldCres.getProbabilityDistribution(predid), foldCres.getClusterValue(predid),
                                foldCres.getPredictionTime(predid), foldCres.getPredDescription(predid));
                    }
                }
                concatenatedResults[classifierid] = newCres;
            }
        }
        return concatenatedResults;
    }

    public static void main(String[] args) {
        ClustererResults cr = new ClustererResults(3);
        Collections.addAll(cr.trueClassValues, 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 2., 2., 2., 2., 2.);
        Collections.addAll(cr.clusterValues, 0., 1., 1., 0., 0., 1., 0., 3., 3., 3., 2., 2., 2., 2., 2.);
        cr.numInstances = 15;
        cr.numClusters = 4;
        cr.findAllStats();
        System.out.println(cr.statsToString());
    }
}
