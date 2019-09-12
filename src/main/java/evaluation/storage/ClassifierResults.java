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

import fileIO.OutFile;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import utilities.DebugPrinting;
import utilities.GenericTools;
import utilities.InstanceTools;

/**
 * This is a container class for the storage of predictions and meta-info of a
 * classifier on a single set of instances (for example, the test set of a particular
 * resample of a particular dataset).
 *
 * Predictions can be stored via addPrediction(...) or addAllPredictions(...)
 * Currently, the information stored about each prediction is:
 *    - The true class value                            (double   getTrueClassValue(index))
 *    - The predicted class value                       (double   getPredClassValue(index))
 *    - The probability distribution for this instance  (double[] getProbabilityDistribution(index))
 *    - The time taken to predict this instance id      (long     getPredictionTime(index))
 *    - An optional description of the prediction       (String   getPredDescription(index))
 *
 * The meta info stored is:
 *  [LINE 1 OF FILE]
 *    - get/setDatasetName(String)
 *    - get/setClassifierName(String)
 *    - get/setSplit(String)
 *    - get/setFoldId(String)
 *    - get/setTimeUnit(TimeUnit)
 *    - FileType, set implicitly via the write...() method used
 *    - get/setDescription(String)
 *  [LINE 2 OF FILE]
 *    - get/setParas(String)
 *  [LINE 3 OF FILE]
 *    - getAccuracy() (calculated from predictions, only settable with a suitably annoying message)
 *    - get/setBuildTime(long)
 *    - get/setTestTime(long)
 *    - get/setBenchmarkTime(long)
 *    - get/setMemory(long)
 *    - (set)numClasses(int) (either set by user or indirectly found through predicted probability distributions)
 *    - get/setErrorEstimateMethod(String) (loosely formed, e.g. cv_10)
 *    - get/setErrorEstimateTime(long) (time to form an estimate from scratch, e.g. time of cv_10)
 *    - get/setBuildAndEstimateTime(long) (time to train on full data, AND estimate error on it)
 *  [REMAINING LINES: PREDICTIONS]
 *    - trueClassVal, predClassVal,[empty], dist[0], dist[1] ... dist[c],[empty], predTime, [empty], predDescription
 *
 * Supports reading/writing of results from/to file, in the 'classifierResults file-format'
 *    - loadResultsFromFile(String path)
 *    - writeFullResultsToFile(String path)  (other writing formats also supported, write...ToFile(...)
 *
 * Supports recording of timings in different time units. Milliseconds is the default for
 * backwards compatability, however nano seconds is generally preferred.
 * Older files that are read in and do not have a time unit specified are assumed to be in milliseconds.
 *
 * WARNING: The timeunit does not enforce/convert any already-stored times to the new time unit from the old.
 * If e.g build time is set to 10 (intended to mean 10 milliseconds, as would be default), but then time
 * unit was changed to e.g seconds, the value stored as the build time is still 10. User must make sure
 * to either perform conversions themselves or be consistent in their timing units
 *      long buildTimeInSecs = //some process
 *      long buildTimeInResultsUnit = results.getTimeUnit().convert(builtTimeInSecs, TimeUnit.SECONDS);
 *      results.setBuildTime(buildTimeInResultsUnit)
 *
 * Also supports the calculation of various evaluative performance metrics  based on the predictions (accuracy,
 * auroc, nll etc.) which are used in the MultipleClassifierEvaluation pipeline. For now, call
 * findAllStats() to calculate the performance metrics based on the stored predictions, and access them
 * via directly via the public variables. In the future, these metrics will likely be separated out
 * into their own package
 *
 *
 * EXAMPLE USAGE:
 *          ClassifierResults res = new ClassifierResults();
 *          //set a particular timeunit, if using something other than millis. Nanos recommended
 *          //set any meta info you want to keep, e.g classifiername, datasetname...
 *
 *          for (Instance inst : test) {
 *              long startTime = //time
 *              double[] dist = classifier.distributionForInstance(inst);
 *              long predTime = //time - startTime
 *
 *              double pred = max(dist); //with some particular tie breaking scheme built in.
 *                              //easiest is utilities.GenericTools.indexOfMax(double[])
 *
 *              res.addPrediction(inst.classValue(), dist, pred, predTime, ""); //desription is optional
 *          }
 *
 *          res.finaliseResults(); //performs some basic validation, and calcs some relevant internal info
 *
 *          //can now find summary scores for these predictions
 *          //stats stored in simple public members for now
 *          res.findAllStats();
 *
 *          //and/or save to file
 *          res.writeFullResultsToFile(path);
 *
 *          //and could then load them back in
 *          ClassifierResults res2 = new ClassifierResults(path);
 *
 *          //the are automatically finalised, however the stats are not automatically found
 *          res2.findAllStats();
 *
 * TODOS:
 *      - Move metric/scores/stats into their own packge, and rename consistently to scores OR metrics.
 *      - Rename finaliseResults to finalisePredictions, and add in the extra validation
 *      - Consult with group and implement writeCompactResultsTo...(...) as wanted
 *      - Maybe break down the object into different parts to reduce the get/set bloat. This
 *           is a very large and needlessly complex object. Predictions object, ExpInfo (line1) object, etcetc
 *
 * @author James Large (james.large@uea.ac.uk) + edits from just about everybody
 * @date 19/02/19
 */
public class ClassifierResults implements DebugPrinting, Serializable{

    /**
     * Print a message with the filename to stdout when a file cannot be loaded.
     * Can get very tiresome if loading thousands of files with some expected failures,
     * and a higher level process already summarises them, thus this option to
     * turn off the messages
     */
    public static boolean printOnFailureToLoad = true;


//LINE 1: meta info, set by user
    private String classifierName = "";
    private String datasetName = "";
    private int foldID = -1;
    private String split = ""; //e.g train or test


    private enum FileType {
        /**
         * Writes/loads the first 3 lines, and all prediction info on the remaining numInstances lines
         *
         * Usable in all evaluations and post-processed ensembles etc.
         */
        PREDICTIONS,

        /**
         * Writes/can only be guaranteed to contain the first 3 lines, and the summative metrics, NOT
         * full prediction info
         *
         * Usable in evaluations that are restricted to the metrics described in this file,
         * but not post-processed ensembles
         */
        METRICS,

        /**
         * To be defined more precisely at later date. Intended use would be a classifiers' internal
         * storage, perhaps for checkpointing etc if full writing/reading would simply take up too much space
         * and IO compute overhead. Goastler to define
         */
        COMPACT
    };
    private FileType fileType = FileType.PREDICTIONS;

    private String description= ""; //human-friendly optional extra info if wanted.

//LINE 2: classifier setup/info, parameters. precise format is up to user.
    //e.g maybe this line includes the accuracy of each parameter set searched for in a tuning process, etc
    //old versions of file format also include build time.
    private String paras = "No parameter info";

//LINE 3: acc, buildTime, testTime, memoryUsage
    //simple summarative performance stats.

    /**
     * Calculated from the stored predictions, cannot be explicitly set by user
     */
    private double acc = -1;

    /**
     * The time taken to complete buildClassifier(Instances), aka training. May be cumulative time over many parameter set builds, etc
     *
     * It is assumed that the time given will be in the unit of measurement set by this object TimeUnit, default milliseconds, nanoseconds recommended.
     * If no benchmark time is supplied, the default value is -1
     */
    private long buildTime = -1;

    /**
     * The cumulative prediction time, equal to the sum of the individual prediction times stored. Intended as a quick helper/summary
     * in case complete prediction information is not stored, and/or for a human reader to quickly compare times.
     *
     * It is assumed that the time given will be in the unit of measurement set by this object TimeUnit, default milliseconds, nanoseconds recommended.
     * If no benchmark time is supplied, the default value is -1
     */
    private long testTime = -1; //total testtime for all predictions

    /**
     * The time taken to perform some standard benchmarking operation, to allow for a (not necessarily precise)
     * way to measure the general speed of the hardware that these results were made on, such that users
     * analysing the results may scale the timings in this file proportional to the benchmarks to get a consistent relative scale
     * across different results sets. It is up to the user what this benchmark operation is, and how long it is (roughly) expected to take.
     *
     * It is assumed that the time given will be in the unit of measurement set by this object TimeUnit, default milliseconds, nanoseconds recommended.
     * If no benchmark time is supplied, the default value is -1
     */
    private long benchmarkTime = -1;

    /**
     * It is user dependent on exactly what this field means and how accurate it may be (because of Java's lazy gc).
     * Intended purpose would be the size of the model at the end of/after buildClassifier, aka the classifier
     * has been trained.
     *
     * The assumption, for now, is that this is measured in BYTES, but this is not enforced/ensured
     * If no memoryUsage value is supplied, the default value is -1
     */
    private long memoryUsage = -1;


    /**
     * todo initially intended as a temporary measure, but might stay here until a switch
     * over to json etc is made
     *
     * See the experiments parameter trainEstimateMethod
     *
     * This defines the method and parameter of train estimate used, if one was done
     */
    private String errorEstimateMethod = "";

    /**
     * todo initially intended as a temporary measure, but might stay here until a switch
     * over to json etc is made
     *
     * This defines the total time taken to estimate the classifier's error. This currently
     * does not mean anything for classifiers implementing the TrainAccuracyEstimate interface,
     * and as such would need to set this themselves (but likely do not)
     *
     * For those classifiers that do not implement that, Experiments.findOrSetupTrainEstimate(...) will set this value
     * as a wrapper around the entire evaluate call for whichever errorEstimateMethod is being used
     */
    private long errorEstimateTime = -1;

    /**
     * This measures the total time to build the classifier on the train data
     * AND to estimate the classifier's error on the same train data. For classifiers
     * that do not implement TrainAccuracyEstimator, i.e. that do not estimate their
     * own error in some way during the build process, this will simply be the
     * buildTime and the errorEstimateTime added together.
     *
     * For classifiers that DO implement TrainAccuracyEstimator, buildPlusEstimateTime may
     * be anywhere between buildTime and buildTime+errorEstimateTime. Some or all of
     * the work needed to form an estimate (which the field errorEstimateTime measures from scratch)
     * may have already been accounted for by the buildTime
     */
    private long buildPlusEstimateTime = -1;

//REMAINDER OF THE FILE - 1 prediction per line
    //raw performance data. currently just four parallel arrays
    private ArrayList<Double> trueClassValues;
    private ArrayList<Double> predClassValues;
    private ArrayList<double[]> predDistributions;
    private ArrayList<Long> predTimes;
    private ArrayList<String> predDescriptions;

    //inferred/supplied dataset meta info
    private int numClasses;
    private int numInstances;

    //calculated performance metrics
        //accuracy can be re-calced, as well as stored on line three in files
    public double balancedAcc;
    public double sensitivity;
    public double specificity;
    public double precision;
    public double recall;
    public double f1;
    public double mcc; //mathews correlation coefficient
    public double nll;
    public double meanAUROC;
    public double stddev; //across cv folds, where applicable
    public long medianPredTime;
    public double[][] confusionMatrix; //[actual class][predicted class]
    public double[] countPerClass;


    /**
     * Used to avoid infinite NLL scores when prob of true class =0 or
     */
    private static double NLL_PENALTY=-6.64; //Log_2(0.01)

    /**
     * Consistent time unit ASSUMED across build times, test times, individual prediction times.
     * Before considering different timeunits, all timing were in milliseconds, via
     * System.currentTimeMillis(). Some classifiers on some datasets may train/predict in less than 1 millisecond
     * however. The default timeunit will still be milliseconds, however if any time passed has a value of 0,
     * an exception will be thrown. This is in part to convince people using/owning older code and writing
     * new code to switch to nanoseconds where it may be clearly needed.
     *
     * A long can contain 292 years worth of nanoseconds, which I assume to be enough for now.
     * Could be conceivable that the cumulative time of a large meta ensemble that is run
     * multi-threaded on a large dataset might exceed this.
     *
     * In results files made before 19/2/2019, which only stored build times and
     * milliseconds was assumed, there will be no unit of measurement for the time.
     */
    private TimeUnit timeUnit = TimeUnit.MILLISECONDS;


    //self-management flags
    /**
     * essentially controls whether a classifierresults object can have finaliseResults(trueClassVals)
     * called upon it. In theory, every class using the classifierresults object should make new
     * instantiations of it each time a set of results is being computed, and so this is not needed
     *
     * this was relevant especially prior to on-line prediction storage being supported, and effectively
     * the intention was to turn the results into a const object after all the results were stored
     *
     * todo: verify that this can be removed, or update to be more relevant.
     */
    private boolean finalised = false;
    private boolean allStatsFound = false;
    private boolean buildTimeDuplicateWarningPrinted = false; //flag such that a warning about build times in parseThirdLine(String) is only printed once, not spammed


    /**
     * System.nanoTime() can STILL return zero on some tiny datasets with simple classifiers,
     * because it does not have enough precision. This flag, if true, will allow timings
     * of zero, under the partial assumption/understanding from the user that times under
     * ~200 nanoseconds can be equated to 0.
     *
     * The flag defaults to false, however. Correct usage of this flag would be
     * to set it to true in circumstances where you, the coder supplying some kind of
     * timing, KNOW that you are measuring in millis, AND the classifierResults object's
     * timeunit is in millis, AND you reset the flag to false again immediately after
     * adding the potentially offending time, such that the flag is not mistakenly left
     * on for genuinely erroneous timing additions later on.
     *
     * This is in effect a double check that you the user know what you are doing, and old
     * code that sets (buildtimes in millis, mostly) times can be caught and updated if they cause
     * problems
     *
     * E.g
     * results.turnOffZeroTimingsErrorSuppression();
     * results.setBuildTime(time);        // or e.g results.addPrediction(...., time, ...)
     * results.turnOnZeroTimingsErrorSuppression();
     */
    private boolean errorOnTimingOfZero = false;

    //functional getters to retrieve info from a classifierresults object, initialised/stored here for conveniance
    public static final Function<ClassifierResults, Double> GETTER_Accuracy = (ClassifierResults cr) -> {return cr.acc;};
    public static final Function<ClassifierResults, Double> GETTER_BalancedAccuracy = (ClassifierResults cr) -> {return cr.balancedAcc;};
    public static final Function<ClassifierResults, Double> GETTER_AUROC = (ClassifierResults cr) -> {return cr.meanAUROC;};
    public static final Function<ClassifierResults, Double> GETTER_NLL = (ClassifierResults cr) -> {return cr.nll;};
    public static final Function<ClassifierResults, Double> GETTER_F1 = (ClassifierResults cr) -> {return cr.f1;};
    public static final Function<ClassifierResults, Double> GETTER_MCC = (ClassifierResults cr) -> {return cr.mcc;};
    public static final Function<ClassifierResults, Double> GETTER_Precision = (ClassifierResults cr) -> {return cr.precision;};
    public static final Function<ClassifierResults, Double> GETTER_Recall = (ClassifierResults cr) -> {return cr.recall;};
    public static final Function<ClassifierResults, Double> GETTER_Sensitivity = (ClassifierResults cr) -> {return cr.sensitivity;};
    public static final Function<ClassifierResults, Double> GETTER_Specificity = (ClassifierResults cr) -> {return cr.specificity;};

    public static final Function<ClassifierResults, Double> NegMAA = (ClassifierResults cr) -> {
        double MAA = 0;
        for (int i = 0; i < cr.numInstances; i++){
            MAA += Math.abs(cr.trueClassValues.get(i) - cr.predClassValues.get(i));
        }
        return -(MAA/cr.numInstances);
    };

    //todo revisit these when more willing to refactor stats pipeline to avoid assumption of doubles.
    //a double can accurately (except for the standard double precision problems) hold at most ~7 weeks worth of nano seconds
    //      a double's mantissa = 52bits, 2^52 / 1000000000 / 60 / 60 / 24 / 7 = 7.something weeks
    //so, will assume the usage/requirement for milliseconds in the stats pipeline, to avoid the potential future problem
    //of meta-ensembles taking more than a week, etc. (or even just summing e.g 30 large times to be averaged)
    //it is still preferable of course to store any timings in nano's in the classifierresults object since they'll
    //store them as longs.
    public static final Function<ClassifierResults, Double> GETTER_buildTimeDoubleMillis = (ClassifierResults cr) -> {return toDoubleMillis(cr.buildTime, cr.timeUnit);};
    public static final Function<ClassifierResults, Double> GETTER_totalTestTimeDoubleMillis = (ClassifierResults cr) -> {return toDoubleMillis(cr.testTime, cr.timeUnit);};
    public static final Function<ClassifierResults, Double> GETTER_avgTestPredTimeDoubleMillis = (ClassifierResults cr) -> {return toDoubleMillis(cr.medianPredTime, cr.timeUnit);};
    public static final Function<ClassifierResults, Double> GETTER_fromScratchEstimateTimeDoubleMillis = (ClassifierResults cr) -> {return toDoubleMillis(cr.errorEstimateTime, cr.timeUnit);};
    public static final Function<ClassifierResults, Double> GETTER_totalBuildPlusEstimateTimeDoubleMillis = (ClassifierResults cr) -> {return toDoubleMillis(cr.buildPlusEstimateTime, cr.timeUnit);};
    public static final Function<ClassifierResults, Double> GETTER_additionalTimeForEstimateDoubleMillis = (ClassifierResults cr) -> {return toDoubleMillis(cr.buildPlusEstimateTime - cr.buildTime, cr.timeUnit);};

    private static double toDoubleMillis(long time, TimeUnit unit) {
        if (time < 0)
            return -1;
        if (time == 0)
            return 0;

        if (unit.equals(TimeUnit.MICROSECONDS)) {
            long pre = time / 1000;  //integer division for pre - decimal point
            long post = time % 1000;  //the remainder that needs to be converted to post decimal point, some value < 1000
            double convertedPost = (double)post / 1000; // now some fraction < 1

            return pre + convertedPost;
        }
        else if (unit.equals(TimeUnit.NANOSECONDS)) {
            long pre = time / 1000000;  //integer division for pre - decimal point
            long post = time % 1000000;  //the remainder that needs to be converted to post decimal point, some value < 1000
            double convertedPost = (double)post / 1000000; // now some fraction < 1

            return pre + convertedPost;
        }
        else {
            //not higher resolution than millis, no special conversion needed just cast to double
            return (double)unit.toMillis(time);
        }
    }



    /*********************************
     *
     *       CONSTRUCTORS
     *
     */

    /**
     * Create an empty classifierResults object.
     *
     * If number of classes is known when making the object, it is safer to use the constructor
     * the takes an int representing numClasses and supply the number of classes directly.
     *
     * In some extreme use cases, predictions on dataset splits that a particular classifier results represents
     * may not have examples of each class that actually exists in the full dataset. If it is left
     * to infer the number of classes, some may be missing.
     */
    public ClassifierResults() {
        trueClassValues= new ArrayList<>();
        predClassValues = new ArrayList<>();
        predDistributions = new ArrayList<>();
        predTimes = new ArrayList<>();
        predDescriptions = new ArrayList<>();

        finalised = false;
    }

    /**
     * Create an empty classifierResults object.
     *
     * If number of classes is known when making the object, it is safer to use this constructor
     * and supply the number of classes directly.
     *
     * In some extreme use cases, predictions on dataset splits that a particular classifier results represents
     * may not have examples of each class that actually exists in the full dataset. If it is left
     * to infer the number of classes, some may be missing.
     */
    public ClassifierResults(int numClasses) {
        trueClassValues= new ArrayList<>();
        predClassValues = new ArrayList<>();
        predDistributions = new ArrayList<>();
        predTimes = new ArrayList<>();
        predDescriptions = new ArrayList<>();

        this.numClasses = numClasses;
        finalised = false;
    }

    /**
     * Load a classifierresults object from the file at the specified path
     */
    public ClassifierResults(String filePathAndName) throws FileNotFoundException, Exception {
        loadResultsFromFile(filePathAndName);
    }

    /**
     * Create a classifier results object with complete predictions (equivalent to addAllPredictions()). The results are
     * FINALISED after initialisation. Meta info such as classifier name, datasetname... can still be set after construction.
     *
     * The descriptions array argument may be null, in which case the descriptions are stored as empty strings.
     *
     * All other arguments are required in full, however
     */
    public ClassifierResults(double[] trueClassVals, double[] predictions, double[][] distributions, long[] predTimes, String[] descriptions) throws Exception {
        trueClassValues= new ArrayList<>();
        predClassValues = new ArrayList<>();
        predDistributions = new ArrayList<>();
        this.predTimes = new ArrayList<>();
        predDescriptions = new ArrayList<>();

        addAllPredictions(trueClassVals, predictions, distributions, predTimes, descriptions);
        finaliseResults();
    }

    /**
     * System.nanoTime() can STILL return zero on some tiny datasets with simple classifiers,
     * because it does not have enough precision. This flag, if true, will allow timings
     * of zero, under the partial assumption/understanding from the user that times under
     * ~200 nanoseconds can be equated to 0.
     *
     * The flag defaults to false, however. Correct usage of this flag would be
     * to set it to true in circumstances where you, the coder supplying some kind of
     * timing, KNOW that you are measuring in nanos, AND the classifierResults object's
     * timeunit is in nanos, AND you reset the flag to false again immediately after
     * adding the potentially offending time, such that the flag is not mistakenly left
     * on for genuinely erroneous timing additions later on.
     *
     * This is in effect a double check that you the user know what you are doing, and old
     * code that sets (buildtimes in millis, mostly) times can be caught and updated if they cause
     * problems
     *
     * E.g
     * results.turnOffZeroTimingsErrorSuppression();
     * results.setBuildTime(time);        // or e.g results.addPrediction(...., time, ...)
     * results.turnOnZeroTimingsErrorSuppression();
     */
    public void turnOffZeroTimingsErrors() {
        errorOnTimingOfZero = false;
    }
    /**
     * System.nanoTime() can STILL return zero on some tiny datasets with simple classifiers,
     * because it does not have enough precision. This flag, if true, will allow timings
     * of zero, under the partial assumption/understanding from the user that times under
     * ~200 nanoseconds can be equated to 0.
     *
     * The flag defaults to false, however. Correct usage of this flag would be
     * to set it to true in circumstances where you, the coder supplying some kind of
     * timing, KNOW that you are measuring in millis, AND the classifierResults object's
     * timeunit is in millis, AND you reset the flag to false again immediately after
     * adding the potentially offending time, such that the flag is not mistakenly left
     * on for genuinely erroneous timing additions later on.
     *
     * This is in effect a double check that you the user know what you are doing, and old
     * code that sets (buildtimes in millis, mostly) times can be caught and updated if they cause
     * problems
     *
     * E.g
     * results.turnOffZeroTimingsErrorSuppression();
     * results.setBuildTime(time);        // or e.g results.addPrediction(...., time, ...)
     * results.turnOnZeroTimingsErrorSuppression();
     */
    public void turnOnZeroTimingsErrors() {
        errorOnTimingOfZero = true;
    }


    /***********************
     *
     *      DATASET META INFO
     *
     *
     */

    /**
     * Will return the number of classes if it has been a) explicitly set or b) found via
     * the size of the probability distributions attached to predictions that have been
     * stored/loaded, otherwise this will return 0.
     */
    public int numClasses() {
        if (numClasses <= 0)
            inferNumClasses();
        return numClasses;
    }
    public void setNumClasses(int numClasses) {
        this.numClasses = numClasses;
    }
    private void inferNumClasses() {
        if (predDistributions.isEmpty())
            this.numClasses = 0;
        else
            this.numClasses = predDistributions.get(0).length;
    }

    public int numInstances() {
        if (numInstances <= 0)
            inferNumInstances();
        return numInstances;
    }

    private void inferNumInstances() {
        this.numInstances = predClassValues.size();
    }




    /***************************
     *
     *   LINE 1 GETS/SETS
     *
     *  Just basic descriptive stuff, nothing fancy goign on here
     *
     */

    public String getClassifierName() { return classifierName; }
    public void setClassifierName(String classifierName) { this.classifierName = classifierName; }

    public String getDatasetName() { return datasetName; }
    public void setDatasetName(String datasetName) { this.datasetName = datasetName; }

    public int getFoldID() { return foldID; }
    public void setFoldID(int foldID) { this.foldID = foldID; }

    /**
     * e.g "train", "test", "validation"
     */
    public String getSplit() { return split; }

    /**
     * e.g "train", "test", "validation"
     */
    public void setSplit(String split) { this.split = split; }


    /**
     * Consistent time unit ASSUMED across build times, test times, individual prediction times.
     * Before considering different timeunits, all timing were in milliseconds, via
     * System.currentTimeMillis(). Some classifiers on some datasets may run in less than 1 millisecond
     * however, so as of 19/2/2019, classifierResults now defaults to working in nanoseconds.
     *
     * A long can contain 292 years worth of nanoseconds, which I assume to be enough for now.
     * Could be conceivable that the cumulative time of a large meta ensemble that is run
     * multi-threaded on a large dataset might exceed this.
     *
     * In results files made before 19/2/2019, which only stored build times and
     * milliseconds was assumed, there will be no unit of measurement for the time.
     */
    public TimeUnit getTimeUnit() {
        return timeUnit;
    }

    /**
     * This will NOT convert any timings already stored in this classifier results object
     * to the new time unit. e.g if build time was had already been stored in seconds as 10, THEN
     * setTimeUnit(TimeUnit.MILLISECONDS) was called, the actual value of build time would still be 10,
     * but now assumed to mean 10 milliseconds.
     *
     * Consistent time unit ASSUMED across build times, test times, individual prediction times.
     * Before considering different timeunits, all timing were in milliseconds, via
     * System.currentTimeMillis(). Some classifiers on some datasets may run in less than 1 millisecond
     * however, so as of 19/2/2019, classifierResults now defaults to working in nanoseconds.
     *
     * A long can contain 292 years worth of nanoseconds, which I assume to be enough for now.
     * Could be conceivable that the cumulative time of a large meta ensemble that is run
     * multi-threaded on a large dataset might exceed this.
     *
     * In results files made before 19/2/2019, which only stored build times and
     * milliseconds was assumed, there will be no unit of measurement for the time.
     */
    public void setTimeUnit(TimeUnit timeUnit) {
        this.timeUnit = timeUnit;
    }



    /**
     * This is a free-form description that can hold any info you want, with the only caveat
     * being that it cannot contain newline characters. Description could be the experiment
     * that these results were made for, e.g "Initial Univariate Benchmarks". Entirely
     * up to the user to process if they want to.
     *
     * By default, it is an empty string.
     */
    public String getDescription() {
        return description;
    }
    /**
     * This is a free-form description that can hold any info you want, with the only caveat
     * being that it cannot contain newline characters. Description could be the experiment
     * that these results were made for, e.g "Initial Univariate Benchmarks". Entirely
     * up to the user to process if they want to.
     *
     * By default, it is an empty string.
     */
    public void setDescription(String description) {
        this.description = description;
    }




    /*****************************
     *
     *     LINE 2 GETS/SETS
     *
     */

    /**
     * For now, user dependent on the formatting of this string, and really, the contents of it.
     * It is notionally intended to contain the parameters of the classifier used to produce the
     * attached predictions, but could also store other things as well.
     */
    public String getParas() { return paras; }
    /**
     * For now, user dependent on the formatting of this string, and really, the contents of it.
     * It is notionally intended to contain the parameters of the classifier used to produce the
     * attached predictions, but could also store other things as well.
     */
    public void setParas(String paras) { this.paras = paras; }



    /*****************************
     *
     *     LINE 3 GETS/SETS
     *
     */

    /**
     * This setter exists purely for backwards compatibility, for classifiers that
     * for whatever reason do not have per-instance prediction info.
     *
     * This might be because
     *     a) The accuracy is gathered from some internal/weka eval process that we dont
     *          want to edit, e.g out of bag error in some forests.
     *     b) The classifier (typically implementing TrainAccuracyEstimate) does not yet
     *          save prediction info, simply because it was written before we did that and
     *          hasnt been updated. These SHOULD be refactored over time.
     *
     * This method will print a suitably annoying message when first called, as a reminder
     * until the accuracy is no longer directly set
     *
     * If you REALLY dont want this message being printed, since e.g. it's messing up your own print formatting,
     * set ClassifierResults.printSetAccWarning to false. This also acts a way of ensuring that you've read this
     * message...
     *
     * Todo: remove this method, i.e. the possibility to directly set the accuracy instead of
     * have it calculated implicitly, when possible.
     */
    public void setAcc(double acc) {
        if (printSetAccWarning && firstTimeInSetAcc) {
            System.out.println("*********");
            System.out.println("");
            System.out.println("ClassifierResults.setAcc(double acc) called, friendly reminder to refactor the code that "
                    + "made this call. If you REALLY dont want this message being printed right now, since e.g. it's messing up your "
                    + "own print formatting, set ClassifierResults.printSetAccWarning to false.");
            System.out.println("");
            System.out.println("*********");

            firstTimeInSetAcc = false;
        }

        this.acc = acc;
    }
    public static boolean printSetAccWarning = true;
    private boolean firstTimeInSetAcc = true;

    public double getAcc() {
        if (acc < 0)
            calculateAcc();
        return acc;
    }
    public boolean isAccSet(){
        return acc<0 ? false: true;
    }
    private void calculateAcc() {
        if (trueClassValues == null || trueClassValues.isEmpty() || trueClassValues.get(0) == -1) {
            System.out.println("**getAcc():calculateAcc() no true class values supplied yet, cannot calculate accuracy");
            return;
        }

        int size = predClassValues.size();
        double correct = .0;
        for (int i = 0; i < size; i++) {
            if (predClassValues.get(i).equals(trueClassValues.get(i)))
                correct++;
        }

        acc = correct / size;
    }

    public long getBuildTime() { return buildTime; }
    public long getBuildTimeInNanos() { return timeUnit.toNanos(buildTime); }
    /**
     * @throws Exception if buildTime is less than 1
     */
    public void setBuildTime(long buildTime) throws Exception {
        if (errorOnTimingOfZero && buildTime < 1)
            throw new Exception("Build time passed has invalid value, " + buildTime + ". If greater resolution is needed, "
                        + "use nano seconds (e.g System.nanoTime()) and set the TimeUnit of the classifierResults object to nanoseconds.\n\n"
                    + "If you are using nanoseconds but STILL getting this error, read the javadoc for and use turnOffZeroTimingsErrors() "
                    + "for this call");
        this.buildTime = buildTime;
    }

    public long getTestTime() { return testTime; }
    public long getTestTimeInNanos() { return timeUnit.toNanos(testTime); }
    /**
     * @throws Exception if testTime is less than 1
     */
    public void setTestTime(long testTime) throws Exception {
        if (errorOnTimingOfZero && testTime < 1)
            throw new Exception("Test time passed has invalid value, " + testTime + ". If greater resolution is needed, "
                    + "use nano seconds (e.g System.nanoTime()) and set the TimeUnit of the classifierResults object to nanoseconds.\n\n"
                    + "If you are using nanoseconds but STILL getting this error, read the javadoc for and use turnOffZeroTimingsErrors() "
                    + "for this call");
        this.testTime = testTime;
    }

    public long getMemory() { return memoryUsage; }
    public void setMemory(long memory) {
        this.memoryUsage = memory;
    }


    /**
     * The time taken to perform some standard benchmarking operation, to allow for a (not necessarily precise)
     * way to measure the general speed of the hardware that these results were made on, such that users
     * analysing the results may scale the timings in this file proportional to the benchmarks to get a consistent relative scale
     * across different results sets.
     *
     * It is up to the user what this benchmark operation is, and how long it is (roughly) expected to take. If no benchmark
     * time is supplied, the default value is -1
     */
    public long getBenchmarkTime() {
        return benchmarkTime;
    }

    /**
     * The time taken to perform some standard benchmarking operation, to allow for a (not necessarily precise)
     * way to measure the general speed of the hardware that these results were made on, such that users
     * analysing the results may scale the timings in this file proportional to the benchmarks to get a consistent relative scale
     * across different results sets.
     *
     * It is up to the user what this benchmark operation is, and how long it is (roughly) expected to take. If no benchmark
     * time is supplied, the default value is -1
     */
    public void setBenchmarkTime(long benchmarkTime) {
        this.benchmarkTime = benchmarkTime;
    }

    /**
     * todo initially intended as a temporary measure, but might stay here until a switch
     * over to json etc is made
     *
     * See the experiments parameter trainEstimateMethod
     *
     * This defines the method and parameter of train estimate used, if one was done
     */
    public String getErrorEstimateMethod() {
        return errorEstimateMethod;
    }

    /**
     * todo initially intended as a temporary measure, but might stay here until a switch
     * over to json etc is made
     *
     * See the experiments parameter trainEstimateMethod
     *
     * This defines the method and parameter of train estimate used, if one was done
     */
    public void setErrorEstimateMethod(String errorEstimateMethod) {
        this.errorEstimateMethod = errorEstimateMethod;
    }

    /**
     * todo initially intended as a temporary measure, but might stay here until a switch
     * over to json etc is made
     *
     * This defines the total time taken to estimate the classifier's error. This currently
     * does not mean anything for classifiers implementing the TrainAccuracyEstimate interface,
     * and as such would need to set this themselves (but likely do not)
     *
     * For those classifiers that do not implement that, Experiments.findOrSetupTrainEstimate(...) will set this value
     * as a wrapper around the entire evaluate call for whichever errorEstimateMethod is being used
     */
    public long getErrorEstimateTime() {
        return errorEstimateTime;
    }

    /**
     * todo initially intended as a temporary measure, but might stay here until a switch
     * over to json etc is made
     *
     * This defines the total time taken to estimate the classifier's error. This currently
     * does not mean anything for classifiers implementing the TrainAccuracyEstimate interface,
     * and as such would need to set this themselves (but likely do not)
     *
     * For those classifiers that do not implement that, Experiments.findOrSetupTrainEstimate(...) will set this value
     * as a wrapper around the entire evaluate call for whichever errorEstimateMethod is being used
     */
    public void setErrorEstimateTime(long errorEstimateTime) {
        this.errorEstimateTime = errorEstimateTime;
    }


    /**
     * This measures the total time to build the classifier on the train data
     * AND to estimate the classifier's error on the same train data. For classifiers
     * that do not implement TrainAccuracyEstimator, i.e. that do not estimate their
     * own error in some way during the build process, this will simply be the
     * buildTime and the errorEstimateTime added together.
     *
     * For classifiers that DO implement TrainAccuracyEstimator, buildPlusEstimateTime may
     * be anywhere between buildTime and buildTime+errorEstimateTime. Some or all of
     * the work needed to form an estimate (which the field errorEstimateTime measures from scratch)
     * may have already been accounted for by the buildTime
     */
    public long getBuildPlusEstimateTime() {
        return buildPlusEstimateTime;
    }

    /**
     * This measures the total time to build the classifier on the train data
     * AND to estimate the classifier's error on the same train data. For classifiers
     * that do not implement TrainAccuracyEstimator, i.e. that do not estimate their
     * own error in some way during the build process, this will simply be the
     * buildTime and the errorEstimateTime added together.
     *
     * For classifiers that DO implement TrainAccuracyEstimator, buildPlusEstimateTime may
     * be anywhere between buildTime and buildTime+errorEstimateTime. Some or all of
     * the work needed to form an estimate (which the field errorEstimateTime measures from scratch)
     * may have already been accounted for by the buildTime
     */
    public void setBuildPlusEstimateTime(long buildPlusEstimateTime) {
        this.buildPlusEstimateTime = buildPlusEstimateTime;
    }




    /****************************
     *
     *    PREDICTION STORAGE
     *
     */
    /**
     * Will update the internal prediction info using the values passed. User must pass the predicted class
     * so that they may resolve ties how they want (e.g first, randomly, take modal class, etc).
     * The standard, used in most places, would be utilities.GenericTools.indexOfMax(double[] dist)
     *
     * The description argument may be null, however all other arguments are required in full
     *
     * Todo future, maaaybe add enum/functor arg for tie resolution to handle it here.
     *
     * The true class is missing, however can be added in one go later with the
     * method finaliseResults(double[] trueClassVals)
     */
    public void addPrediction(double[] dist, double predictedClass, long predictionTime, String description) throws RuntimeException {
        predDistributions.add(dist);
        predClassValues.add(predictedClass);

        if (description == null)
            predDescriptions.add("");
        else
            predDescriptions.add(description);


        if (errorOnTimingOfZero && predictionTime < 1)
            throw new RuntimeException("Prediction time passed has invalid value, " + predictionTime + ". If greater resolution is needed, "
                    + "use nano seconds (e.g System.nanoTime()) and set the TimeUnit of the classifierResults object to nanoseconds.\n\n"
                    + "If you are using nanoseconds but STILL getting this error, read the javadoc for and use turnOffZeroTimingsErrors() "
                    + "for this call");
        else {
            predTimes.add(predictionTime);

            if (testTime == -1)
                testTime = predictionTime;
            else
                testTime += predictionTime;
        }

        numInstances++;
    }

    /**
     * Will update the internal prediction info using the values passed. User must pass the predicted class
     * so that they may resolve ties how they want (e.g first, randomly, take modal class, etc).
     * The standard, used in most places, would be utilities.GenericTools.indexOfMax(double[] dist)
     *
     * The description argument may be null, however all other arguments are required in full
     *
     * Todo future, maaaybe add enum for tie resolution to handle it here.
     */
    public void addPrediction(double trueClassVal, double[] dist, double predictedClass, long predictionTime, String description) throws RuntimeException {
        addPrediction(dist,predictedClass,predictionTime,description);
        trueClassValues.add(trueClassVal);
    }


    /**
     * Adds all the prediction info onto this classifierResults object. Does NOT finalise the results,
     * such that (e.g) predictions from multiple dataset splits can be added to the same object if wanted
     *
     * The description argument may be null, however all other arguments are required in full
     */
    public void addAllPredictions(double[] trueClassVals, double[] predictions, double[][] distributions, long[] predTimes, String[] descriptions) throws RuntimeException {
        assert(trueClassVals.length == predictions.length);
        assert(trueClassVals.length == distributions.length);
        assert(trueClassVals.length == predTimes.length);

        if (descriptions != null)
            assert(trueClassVals.length == descriptions.length);

        for (int i = 0; i < trueClassVals.length; i++) {
            if (descriptions == null)
                addPrediction(trueClassVals[i], distributions[i], predictions[i], predTimes[i], null);
            else
                addPrediction(trueClassVals[i], distributions[i], predictions[i], predTimes[i], descriptions[i]);
        }
    }

    /**
     * Adds all the prediction info onto this classifierResults object. Does NOT finalise the results,
     * such that (e.g) predictions from multiple dataset splits can be added to the same object if wanted
     *
     * True class values can later be supplied (ALL IN ONE GO, if working to the above example usage..) using
     * finaliseResults(double[] testClassVals)
     *
     * The description argument may be null, however all other arguments are required in full
     */
    public void addAllPredictions(double[] predictions, double[][] distributions, long[] predTimes, String[] descriptions ) throws RuntimeException {

        //todo replace asserts with actual exceptions
        assert(predictions.length == distributions.length);
        assert(predictions.length == predTimes.length);

        if (descriptions != null)
            assert(predictions.length == descriptions.length);

        for (int i = 0; i < predictions.length; i++) {
            if (descriptions == null)
                addPrediction(distributions[i], predictions[i], predTimes[i], "");
            else
                addPrediction(distributions[i], predictions[i], predTimes[i], descriptions[i]);
        }
    }

    /**
     * Will perform some basic validation to make sure that everything is here
     * that is expected, and compute the accuracy etc ready for file writing.
     *
     * Typical usage: results.finaliseResults(instances.attributeToDoubleArray(instances.classIndex()))
     */
    public void finaliseResults(double[] testClassVals) throws Exception {

        //todo extra verification

        if (finalised) {
            System.out.println("finaliseResults(double[] testClassVals): Results already finalised, skipping re-finalisation");
            return;
        }

        if (testClassVals.length != predClassValues.size())
            throw new Exception("finaliseTestResults(double[] testClassVals): Number of predictions "
                    + "made and number of true class values passed do not match");

        trueClassValues = new ArrayList<>();
        for(double d:testClassVals)
            trueClassValues.add(d);

        finaliseResults();
    }


    /**
     * Will perform some basic validation to make sure that everything is here
     * that is expected, and compute the accuracy etc ready for file writing.
     *
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
       if (numClasses <= 0)
           inferNumClasses();

        //todo extra verification

        if (predDistributions == null || predClassValues == null ||
                predDistributions.isEmpty() || predClassValues.isEmpty())
            throw new Exception("finaliseTestResults(): no test predictions stored for this module");

        double correct = .0;
        for (int inst = 0; inst < predClassValues.size(); inst++)
            if (trueClassValues.get(inst).equals(predClassValues.get(inst)))
                ++correct;

        acc = correct/trueClassValues.size();

        finalised = true;
    }

    public boolean hasProbabilityDistributionInformation() {
        return predDistributions != null &&
                !predDistributions.isEmpty() &&
                predDistributions.size() == predClassValues.size() &&
                predDistributions.get(0) != null;
    }

    /**
     * If this results object does not contain probability distributions but does
     * contain predicted classes, this will infer distributions as one-hot vectors
     * from the predicted class values, i.e if class 0 is predicted in a three class
     * problem, dist would be [ 1.0, 0.0, 0.0 ]
     *
     * If this object already contains distributions, this method will do nothing
     *
     * Returns whether or not values were missing but have been populated
     *
     * The number of classes is inferred from via length(unique(trueclassvalues)). As a
     * reminder of why this method should not generally be used unless you have a specific
     * reason, this may not be entirely correct, if e.g a particular cv fold of a particular
     * subsample does not contain instances of every class. And also in general it assumes
     * that the true class values supplied (as they would be if read from file) Consider yourself warned
     *
     * Intended to help with old results files that may not have distributions stored.
     * Should not be used by default anywhere and everywhere to overcome laziness in
     * newly generated results, thus in part it's implementation as a single method applied
     * to an already populated set of results.
     *
     * Intended usage:
     * res.loadFromFile(someOldFilePotentiallyMissingDists);
     * if (ignoreMissingDists) {
     *   res.populateMissingDists();
     * }
     * // res.findAllStats() etcetcetc
     */
    public boolean populateMissingDists() {
        if (this.hasProbabilityDistributionInformation())
            return false;

        if (this.numClasses <= 0)
            //ayyyy java8 being used for something
            numClasses = (int) trueClassValues.stream().distinct().count();

        predDistributions = new ArrayList<>(predClassValues.size());
        for (double d : predClassValues) {
            double[] dist = new double[numClasses];
            dist[(int)d] = 1;
            predDistributions.add(dist);
        }

        return true;
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

    public double[] getTrueClassValsAsArray(){
        double[] d=new double[trueClassValues.size()];
        int i=0;
        for(double x:trueClassValues)
            d[i++]=x;
        return d;
    }

    public double getTrueClassValue(int index){
        return trueClassValues.get(index);
    }


    public ArrayList<Double> getPredClassVals(){
        return predClassValues;
    }

    public double[] getPredClassValsAsArray(){
        double[] d=new double[predClassValues.size()];
        int i=0;
        for(double x:predClassValues)
            d[i++]=x;
        return d;
    }

    public double getPredClassValue(int index){
        return predClassValues.get(index);
    }


    public ArrayList<double[]> getProbabilityDistributions() {
        return predDistributions;
    }

    public double[][] getProbabilityDistributionsAsArray() {
        return predDistributions.toArray(new double[][] {});
    }

    public double[] getProbabilityDistribution(int i){
       if(i<predDistributions.size())
            return predDistributions.get(i);
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

    public ArrayList<String> getPredDescriptions() {
        return predDescriptions;
    }

    public String[] getPredDescriptionsAsArray() {
        String[] ds=new String[predDescriptions.size()];
        int i=0;
        for(String d:predDescriptions)
            ds[i++]=d;
        return ds;
    }

    public String getPredDescription(int index) {
        return predDescriptions.get(index);
    }

    public void cleanPredictionInfo() {
        predDistributions = null;
        predClassValues = null;
        trueClassValues = null;
        predTimes = null;
        predDescriptions = null;
    }




    /********************************
    *
    *     FILE READ/WRITING
    *
    */

    public static boolean exists(File file) {
       return file.exists() && file.length()>0;
       //todo and is valid, maybe
    }
    public static boolean exists(String path) {
        return exists(new File(path));
    }

    private boolean firstTimeDistMissing = true;
    public static boolean printDistMissingWarning = true;
    /**
     * Reads and STORES the prediction in this classifierresults object
     * returns true if the prediction described by this string was correct (i.e. truclass==predclass)
     *
     * INCREMENTS NUMINSTANCES
     *
     * If numClasses is still less than 0, WILL set numclasses if distribution info is present.
     *
     * [true],[pred], ,[dist[0]],...,[dist[c]], ,[predTime], ,[description until end of line, may have commas in it]
     */
    private boolean instancePredictionFromString(String predLine) throws Exception {
        String[] split=predLine.split(",");

        //collect actual/predicted class
        double trueClassVal=Double.valueOf(split[0].trim());
        double predClassVal=Double.valueOf(split[1].trim());

        if(split.length<3) { //no probabilities, no timing. VERY old files will not have them
            if (printDistMissingWarning && firstTimeDistMissing) {
                System.out.println("*********");
                System.out.println("");
                System.out.println("Probability distribution information missing in file. Be aware that certain stats cannot be computed, usability will be diminished. "
                        + "If you know this and dont want this message being printed right now, since e.g. it's messing up your "
                        + "own print formatting, set ClassifierResults.printDistMissingWarning to false.");
                System.out.println("");
                System.out.println("*********");

                firstTimeDistMissing = false;
            }

            addPrediction(trueClassVal, null, predClassVal, -1, "");
            return trueClassVal==predClassVal;
        }
        //else
        //collect probabilities
        final int distStartInd = 3; //actual, predicted, space, distStart
        double[] dist = null;
        if (numClasses < 2) {
            List<Double> distL = new ArrayList<>();
            for(int i = distStartInd; i < split.length; i++) {
                if (split[i].equals(""))
                    break; //we're at the empty-space-separator between probs and timing
                else
                    distL.add(Double.valueOf(split[i].trim()));
            }

            numClasses = distL.size();
            assert(numClasses >= 2);

            dist = new double[numClasses];
            for (int i = 0; i < numClasses; i++)
                dist[i] = distL.get(i);
        }
        else {
            //we know how many classes there should be, use this as implicit
            //file verification
            dist = new double[numClasses];
            for (int i = 0; i < numClasses; i++) {
                //now need to offset by 3.
                dist[i] = Double.valueOf(split[i+distStartInd].trim());
            }
        }

        //collect timings
        long predTime = -1;
        final int timingInd = distStartInd + (numClasses-1) + 1 + 1; //actual, predicted, space, dist, space, timing
        if (split.length > timingInd)
            predTime = Long.parseLong(split[timingInd].trim());

        //collect description
        String description = "";
        final int descriptionInd = timingInd + 1 + 1; //actual, predicted, space, dist, space, timing, space, description
        if (split.length > descriptionInd) {
            description = split[descriptionInd];

            //no reason currently why the description passed cannot have commas in it,
            //might be a natural way to separate it in to different parts.
            //description reall just fills up the remainder of the line.
            for (int i = descriptionInd+1; i < split.length; i++)
                description += "," + split[i];
        }


        addPrediction(trueClassVal, dist, predClassVal, predTime, description);
        return trueClassVal==predClassVal;
    }

    private void instancePredictionsFromScanner(Scanner in) throws Exception {
        double correct = 0;
        while (in.hasNext()) {
            String line = in.nextLine();
            //may be trailing empty lines at the end of the file
            if (line == null || line.equals(""))
                break;

            if (instancePredictionFromString(line))
                correct++;
        }

        acc = correct / numInstances;
    }

    /**
     * [true],[pred], ,[dist[0]],...,[dist[c]], ,[predTime], ,[description until end of line, may have commas in it]
     */
    private String instancePredictionToString(int i) {
        StringBuilder sb = new StringBuilder();

        sb.append(trueClassValues.get(i).intValue()).append(",");
        sb.append(predClassValues.get(i).intValue());

        //probs
        sb.append(","); //<empty space>
        double[] probs=predDistributions.get(i);
        for(double d:probs)
            sb.append(",").append(GenericTools.RESULTS_DECIMAL_FORMAT.format(d));

        //timing
        sb.append(",,").append(predTimes.get(i)); //<empty space>, timing

        //description
        sb.append(",,").append(predDescriptions.get(i)); //<empty space>, description

        return sb.toString();
    }

    public String instancePredictionsToString() throws Exception{

        //todo extra verification

        if (trueClassValues == null || trueClassValues.size() == 0 || trueClassValues.get(0) == -1)
            throw new Exception("No true class value stored, call finaliseResults(double[] trueClassVal)");

        if(numInstances()>0 &&(predDistributions.size()==trueClassValues.size()&& predDistributions.size()==predClassValues.size())){
            StringBuilder sb=new StringBuilder("");

            for(int i=0;i<numInstances();i++){
                sb.append(instancePredictionToString(i));

                if(i<numInstances()-1)
                    sb.append("\n");
            }

            return sb.toString();
        }
        else
           return "No Instance Prediction Information";
    }

    @Override
    public String toString() {
        return generateFirstLine();
    }

    public String writeFullResultsToString() throws Exception {
        finaliseResults();
        fileType = FileType.PREDICTIONS;

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
                     + "Path: " + path +"\nError: "+ e);
        } finally {
            if (out != null)
                out.closeFile();
        }
    }

    public String writeCompactResultsToString() throws Exception {
        finaliseResults();
        fileType = FileType.COMPACT;

        StringBuilder st = new StringBuilder();

        throw new UnsupportedOperationException("COMPACT file writing not yet supported ");

//        return st.toString();
    }

    public void writeCompactResultsToFile(String path) throws Exception {
        OutFile out = null;
        try {
            out = new OutFile(path);
            out.writeString(writeFullResultsToString());
        } catch (Exception e) {
             throw new Exception("Error writing results file.\n"
                     + "Outfile most likely didnt open successfully, probably directory doesnt exist yet.\n"
                     + "Path: " + path +"\nError: "+ e);
        } finally {
            if (out != null)
                out.closeFile();
        }
    }

    /**
     * Writes the first three meta-data lines of the file as normal, but INSTEAD OF
     * writing predictions, writes the evaluative metrics produced by allPerformanceMetricsToString()
     * to fill the rest of the file. This is intended to save disk space and/or memory where
     * full prediction info is not needed, only the summative information. Results files
     * written using this method would not be used to train a post-processed ensemble at a
     * later date, forexample, but could still be used as part of a comparative evaluation
     */
    public String writeSummaryResultsToString() throws Exception {
        finaliseResults();
        findAllStatsOnce();
        fileType = FileType.METRICS;

        StringBuilder st = new StringBuilder();
        st.append(generateFirstLine()).append("\n");
        st.append(generateSecondLine()).append("\n");
        st.append(generateThirdLine()).append("\n");

        st.append(allPerformanceMetricsToString());
        return st.toString();
    }

    /**
     * Writes the first three meta-data lines of the file as normal, but INSTEAD OF
     * writing predictions, writes the evaluative metrics produced by allPerformanceMetricsToString()
     * to fill the rest of the file. This is intended to save disk space and/or memory where
     * full prediction info is not needed, only the summative information. Results files
     * written using this method would not be used to train a post-processed ensemble at a
     * later date, forexample, but could still be used as part of a comparative evaluation
     */
    public void writeSummaryResultsToFile(String path) throws Exception {
        OutFile out = null;
        try {
            out = new OutFile(path);
            out.writeString(writeSummaryResultsToString());
        } catch (Exception e) {
             throw new Exception("Error writing results file.\n"
                     + "Outfile most likely didnt open successfully, probably directory doesnt exist yet.\n"
                     + "Path: " + path +"\nError: "+ e);
        } finally {
            if (out != null)
                out.closeFile();
        }
    }

    private void parseFirstLine(String line) {
        String[] parts = line.split(",");
        if (parts.length == 0)
            return;

        //old tuned classifiers (and maybe others) just wrote a classifier name identifier
        //covering for backward compatability, otherwise datasetname is first
        if (parts.length == 1)
            classifierName = parts[0];
        else {
            datasetName = parts[0];
            classifierName = parts[1];
        }

        if (parts.length > 2)
            split = parts[2];

        if (parts.length > 3)
            foldID = Integer.parseInt(parts[3]);

        if (parts.length > 4)
            setTimeUnitFromString(parts[4]);
        else //time unit is missing, assumed to be older file, which recorded build times in milliseconds by default
            timeUnit = TimeUnit.MILLISECONDS;

        if (parts.length > 5)
            fileType = FileType.valueOf(parts[5]);

        if (parts.length > 6)
            description = parts[6];

        //nothing stopping the description from having its own commas in it, jsut read until end of line
        for (int i = 6; i < parts.length; i++)
            description += "," + parts[i];
    }
    private String generateFirstLine() {
        return datasetName + "," + classifierName + "," + split + "," + foldID + "," + getTimeUnitAsString() + "," + fileType.name() + ", "+ description;
    }

    private void parseSecondLine(String line) {
        paras = line;

        //handle buildtime if it's on this line like older files may have,
        //taking it out of the generic paras string and putting the value into the actual field
        String[] parts = paras.split(",");
        if (parts.length > 0 && parts[0].contains("BuildTime")) {
            buildTime = (long)Double.parseDouble(parts[1].trim());

            if (parts.length > 2) { //this has actual paras too, rebuild this string without buildtime
                paras = parts[2];
                for (int i = 3; i < parts.length; i++) {
                    paras += "," + parts[i];
                }
            }
        }
    }
    private String generateSecondLine() {
        //todo decide what to do with this
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

        acc = Double.parseDouble(parts[0]);

        //if buildtime is here, it shouldn't be on the paras line too.
        //if it is, likely an old SaveParameterInfo implementation put it there
        //for now, overwriting that buildtime with this one, but printing warning
        if (parts.length > 1)  {
            if (buildTime != -1 && !buildTimeDuplicateWarningPrinted)  {
                System.out.println("CLASSIFIERRESULTS READ WARNING: build time reported on both "
                        + "second and third line. Using the value reported on the third line");

                buildTimeDuplicateWarningPrinted = true;
            }

            buildTime = Long.parseLong(parts[1]);
        }
        if (parts.length > 2)
            testTime = Long.parseLong(parts[2]);
        if (parts.length > 3)
            benchmarkTime = Long.parseLong(parts[3]);
        if (parts.length > 4)
            memoryUsage = Long.parseLong(parts[4]);
        if (parts.length > 5)
            numClasses = Integer.parseInt(parts[5]);
        if (parts.length > 6)
            errorEstimateMethod = parts[6];
        if (parts.length > 7)
            errorEstimateTime = Long.parseLong(parts[7]);
            errorEstimateMethod = parts[6];
        if (parts.length > 8)
            buildPlusEstimateTime = Long.parseLong(parts[8]);

        return acc;
    }
    private String generateThirdLine() {
        String res = acc
            + "," + buildTime
            + "," + testTime
            + "," + benchmarkTime
            + "," + memoryUsage
            + "," + numClasses()
            + "," + errorEstimateMethod
            + "," + errorEstimateTime
            + "," + buildPlusEstimateTime;

        return res;
    }

    private String getTimeUnitAsString() {
        return timeUnit.name();
    }

    private void setTimeUnitFromString(String str) {
        timeUnit = TimeUnit.valueOf(str);
    }

    public void loadResultsFromFile(String path) throws FileNotFoundException, Exception {

        try {
            //init
            trueClassValues = new ArrayList<>();
            predClassValues = new ArrayList<>();
            predDistributions = new ArrayList<>();
            predTimes = new ArrayList<>();
            predDescriptions = new ArrayList<>();
            numInstances = 0;
            acc = -1;
            buildTime = -1;
            testTime = -1;
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

            //fileType was read in from first line.
            switch (fileType) {
                case PREDICTIONS: {
                    //have all meta info, start reading predictions or metrics
                    instancePredictionsFromScanner(inf);

                    //acts as a basic form of verification, does the acc reported on line 3 align with
                    //the acc calculated while reading predictions
                    double eps = 1.e-8;
                    if (Math.abs(reportedTestAcc - acc) > eps) {
                        throw new ArithmeticException("Calculated accuracy (" + acc + ") differs from written accuracy (" + reportedTestAcc + ") "
                                + "by more than eps (" + eps + "). File = " + path + ". numinstances = " + numInstances + ". numClasses = " + numClasses);
                    }

                    if (predDistributions == null || predDistributions.isEmpty() || predDistributions.get(0) == null) {
                        if (printDistMissingWarning)
                            System.out.println("Probabiltiy distributions missing from file: " + path);
                    }

                    break;
                }
                case METRICS:
                    allPerformanceMetricsFromScanner(inf);
                    break;
                case COMPACT:
                    throw new UnsupportedOperationException("COMPACT file reading not yet supported");
            }

            finalised = true;
            inf.close();
        }
        catch (FileNotFoundException fnf) {
            if (printOnFailureToLoad)
                System.out.println("File " + path + " NOT FOUND");
            throw fnf;
        }
        catch (Exception ex) {
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



    /**
     * Will calculate all the metrics that can be found from the prediction information
     * stored in this object. Will NOT call finaliseResults(..), and finaliseResults(..)
     * not have been called elsewhere, however if it has not been called then true
     * class values must have been supplied while storing predictions.
     *
     * This is to allow iterative calculation of the metrics (in e.g. batches
     * of added predictions)
     */
    public void findAllStats(){

        //meta info
        if (numInstances <= 0)
            inferNumInstances();
        if (numClasses <= 0)
            inferNumClasses();

        //predictions-only
        confusionMatrix=buildConfusionMatrix();

        countPerClass=new double[confusionMatrix.length];
        for(int i=0;i<trueClassValues.size();i++)
            countPerClass[trueClassValues.get(i).intValue()]++;

        if (acc < 0)
            calculateAcc();
        balancedAcc=findBalancedAcc(confusionMatrix);

        mcc = computeMCC(confusionMatrix);
        f1=findF1(confusionMatrix); //also handles spec/sens/prec/recall in the process of finding f1

        //need probabilities. very old files that have been read in may not have them.
        if (predDistributions != null && !predDistributions.isEmpty() && predDistributions.get(0) != null ) {
            nll=findNLL();
            meanAUROC=findMeanAUROC();
        }

        //timing
        medianPredTime=findMedianPredTime();

        allStatsFound = true;
    }


    /**
     * Will calculate all the metrics that can be found from the prediction information
     * stored in this object, UNLESS this object has been finalised (finaliseResults(..)) AND
     * has already had it's stats found (findAllStats()), e.g. if it has already been called
     * by another process.
     *
     * In this latter case, this method does nothing.
     */
    public void findAllStatsOnce(){
        if (finalised && allStatsFound) {
            printlnDebug("Stats already found, ignoring findAllStatsOnce()");
            return;
        }
        else {
            findAllStats();
        }
    }


    /**
    * @return [actual class][predicted class]
    */
    private double[][] buildConfusionMatrix() {
        double[][] matrix = new double[numClasses][numClasses];
        for (int i = 0; i < predClassValues.size(); ++i){
            double actual=trueClassValues.get(i);
            double predicted=predClassValues.get(i);
            ++matrix[(int)actual][(int)predicted];
        }
        return matrix;
    }


    /**
     * uses only the probability of the true class
     */
    public double findNLL(){
        double nll=0;
        for(int i=0;i<trueClassValues.size();i++){
            double[] dist=getProbabilityDistribution(i);
            int trueClass = trueClassValues.get(i).intValue();

            if(dist[trueClass]==0)
                nll+=NLL_PENALTY;
            else
                nll+=Math.log(dist[trueClass])/Math.log(2);//Log 2
        }
        return -nll/trueClassValues.size();
    }

    public double findMeanAUROC(){
        double a=0;
        if(numClasses==2){
            a=findAUROC(1);
/*            if(countPerClass[0]<countPerClass[1])
            else
                a=findAUROC(1);
 */       }
        else{
            double[] classDist = InstanceTools.findClassDistributions(trueClassValues, numClasses);
            for(int i=0;i<numClasses;i++){
                a+=findAUROC(i) * classDist[i];
            }

            //original, unweighted
//            for(int i=0;i<numClasses;i++){
//                a+=findAUROC(i);
//            }
//            a/=numClasses;
        }
        return a;
    }

    /**
     * todo could easily be optimised further if really wanted
     */
    public double computeMCC(double[][] confusionMatrix) {

        double num=0.0;
        for (int k = 0; k < confusionMatrix.length; ++k)
            for (int l = 0; l < confusionMatrix.length; ++l)
                for (int m = 0; m < confusionMatrix.length; ++m)
                    num += (confusionMatrix[k][k]*confusionMatrix[m][l])-
                            (confusionMatrix[l][k]*confusionMatrix[k][m]);

        if (num == 0.0)
            return 0;

        double den1 = 0.0;
        double den2 = 0.0;
        for (int k = 0; k < confusionMatrix.length; ++k) {

            double den1Part1=0.0;
            double den2Part1=0.0;
            for (int l = 0; l < confusionMatrix.length; ++l) {
                den1Part1 += confusionMatrix[l][k];
                den2Part1 += confusionMatrix[k][l];
            }

            double den1Part2=0.0;
            double den2Part2=0.0;
            for (int kp = 0; kp < confusionMatrix.length; ++kp)
                if (kp!=k) {
                    for (int lp = 0; lp < confusionMatrix.length; ++lp) {
                        den1Part2 += confusionMatrix[lp][kp];
                        den2Part2 += confusionMatrix[kp][lp];
                    }
                }

            den1 += den1Part1 * den1Part2;
            den2 += den2Part1 * den2Part2;
        }

        return num / (Math.sqrt(den1)*Math.sqrt(den2));
    }

    /**
     * Balanced accuracy: average of the accuracy for each class
     * @param cm
     * @return
     */
    public double findBalancedAcc(double[][] cm){
        double[] accPerClass=new double[cm.length];
        for(int i=0;i<cm.length;i++)
            accPerClass[i]=cm[i][i]/countPerClass[i];
        double b=accPerClass[0];
        for(int i=1;i<cm.length;i++)
            b+=accPerClass[i];
        b/=cm.length;
        return b;
    }

    /**
     * F1: If it is a two class problem we use the minority class
     * if it is multiclass we average over all classes.
     * @param cm
     * @return
     */
    public double findF1(double[][] cm){
        double f=0;
        if(numClasses==2){
            if(countPerClass[0]<countPerClass[1])
                f=findConfusionMatrixMetrics(cm,0,1);
            else
                f=findConfusionMatrixMetrics(cm,1,1);
        }
        else{//Average over all of them
            for(int i=0;i<numClasses;i++)
                f+=findConfusionMatrixMetrics(cm,i,1);
            f/=numClasses;
        }
        return f;
    }

    protected double findConfusionMatrixMetrics(double[][] confMat, int c,double beta) {
        double tp = confMat[c][c]; //[actual class][predicted class]
        //some very small non-zero value, in the extreme case that no cases of
        //this class were correctly classified
        if (tp == .0)
            return .0000001;

        double fp = 0.0, fn = 0.0,tn=0.0;

        for (int i = 0; i < confMat.length; i++) {
            if (i!=c) {
                fp += confMat[i][c];
                fn += confMat[c][i];
                tn+=confMat[i][i];
            }
        }

        precision = tp / (tp+fp);
        recall = tp / (tp+fn);
        sensitivity=recall;
        specificity=tn/(fp+tn);

        //jamesl
        //one in a million case on very small AND unbalanced datasets (lenses...) that particular train/test splits and their cv splits
        //lead to a divide by zero on one of these stats (C4.5, lenses, trainFold7 (and a couple others), specificity in the case i ran into)
        //as a little work around, if this case pops up, will simply set the stat to 0
        if (Double.compare(precision, Double.NaN) == 0)
            precision = 0;
        if (Double.compare(recall, Double.NaN) == 0)
            recall = 0;
        if (Double.compare(sensitivity, Double.NaN) == 0)
            sensitivity = 0;
        if (Double.compare(specificity, Double.NaN) == 0)
            specificity = 0;

        return (1+beta*beta) * (precision*recall) / ((beta*beta)*precision + recall);
    }

    /**
     * Makes copy of pred times to easily maintain original ordering
     */
    protected long findMedianPredTime() {
        List<Long> copy = new ArrayList<>(predTimes);
        Collections.sort(copy);

        int mid = copy.size()/2;
        if (copy.size() % 2 == 0)
            return (copy.get(mid) + copy.get(mid-1)) / 2;
        else
            return copy.get(mid);
    }

    protected double findAUROC(int c){
        class Pair implements Comparable<Pair>{
            Double x;
            Double y;
            public Pair(Double a, Double b){
                x=a;
                y=b;
            }
            @Override
            public int compareTo(Pair p) {
                return p.x.compareTo(x);
            }
            public String toString(){ return "("+x+","+y+")";}
        }

        ArrayList<Pair> p=new ArrayList<>();
        double nosPositive=0,nosNegative;
        for(int i=0;i<numInstances;i++){
            Pair temp=new Pair(predDistributions.get(i)[c],trueClassValues.get(i));
            if(c==trueClassValues.get(i))
                nosPositive++;
            p.add(temp);
        }
        nosNegative=trueClassValues.size()-nosPositive;
        Collections.sort(p);

        /* http://www.cs.waikato.ac.nz/~remco/roc.pdf
                Determine points on ROC curve as follows;
                starts in the origin and goes one unit up, for every
        negative outcome the curve goes one unit to the right. Units on the x-axis
        are 1
        #TN and on the y-axis 1
        #TP where #TP (#TN) is the total number
        of true positives (true negatives). This gives the points on the ROC curve
        (0; 0); (x1; y1); : : : ; (xn; yn); (1; 1).
        */
        ArrayList<Pair> roc=new ArrayList<>();
        double x=0;
        double oldX=0;
        double y=0;
        int xAdd=0, yAdd=0;
        boolean xLast=false,yLast=false;
        roc.add(new Pair(x,y));
        for(int i=0;i<numInstances;i++){
            if(p.get(i).y==c){
                if(yLast)
                    roc.add(new Pair(x,y));
                xLast=true;
                yLast=false;
                x+=1/nosPositive;
                xAdd++;
                if(xAdd==nosPositive)
                    x=1.0;

            }
            else{
                if(xLast)
                    roc.add(new Pair(x,y));
                yLast=true;
                xLast=false;
                y+=1/nosNegative;
                yAdd++;
                if(yAdd==nosNegative)
                    y=1.0;
            }
        }
        roc.add(new Pair(1.0,1.0));

        //Calculate the area under the ROC curve, as the sum over all trapezoids with
        //base xi+1 to xi , that is, A

        double auroc=0;
        for(int i=0;i<roc.size()-1;i++){
            auroc+=(roc.get(i+1).y-roc.get(i).y)*(roc.get(i+1).x);
        }
        return auroc;
    }

    public String allPerformanceMetricsToString() {

        String str="numClasses,"+numClasses+"\n";
        str+="numInstances,"+numInstances+"\n";
        str+="acc,"+acc+"\n";
        str+="balancedAcc,"+balancedAcc+"\n";
        str+="sensitivity,"+sensitivity+"\n";
        str+="precision,"+precision+"\n";
        str+="recall,"+recall+"\n";
        str+="specificity,"+specificity+"\n";
        str+="f1,"+f1+"\n";
        str+="mcc,"+mcc+"\n";
        str+="nll,"+nll+"\n";
        str+="meanAUROC,"+meanAUROC+"\n";
        str+="stddev,"+stddev+"\n";
        str+="medianPredTime,"+medianPredTime+"\n";
        str+="countPerClass:\n";
        for(int i=0;i<countPerClass.length;i++)
            str+="Class "+i+","+countPerClass[i]+"\n";
        str+="confusionMatrix:\n";
        for(int i=0;i<confusionMatrix.length;i++){
            for(int j=0;j<confusionMatrix[i].length;j++)
                str+=confusionMatrix[i][j]+",";
            str+="\n";
        }
        return str;
    }
    public void allPerformanceMetricsFromScanner(Scanner scan) throws NoSuchElementException, NumberFormatException {

        try {
            numClasses =    Integer.parseInt(scan.nextLine().split(",")[1]);
            numInstances =  Integer.parseInt(scan.nextLine().split(",")[1]);
            acc =           Double.parseDouble(scan.nextLine().split(",")[1]);
            balancedAcc =   Double.parseDouble(scan.nextLine().split(",")[1]);
            sensitivity =   Double.parseDouble(scan.nextLine().split(",")[1]);
            precision =     Double.parseDouble(scan.nextLine().split(",")[1]);
            recall =        Double.parseDouble(scan.nextLine().split(",")[1]);
            specificity =   Double.parseDouble(scan.nextLine().split(",")[1]);
            f1 =            Double.parseDouble(scan.nextLine().split(",")[1]);
            mcc =           Double.parseDouble(scan.nextLine().split(",")[1]);
            nll =           Double.parseDouble(scan.nextLine().split(",")[1]);
            meanAUROC =     Double.parseDouble(scan.nextLine().split(",")[1]);
            stddev =        Double.parseDouble(scan.nextLine().split(",")[1]);
            medianPredTime= Long.parseLong(scan.nextLine().split(",")[1]);

            assert(scan.nextLine() == "countPerClass");//todo change to if not throws
            countPerClass = new double[numClasses];
            for (int i = 0; i < numClasses; i++)
                countPerClass[i] = Double.parseDouble(scan.nextLine().split(",")[1]);

            assert(scan.nextLine() == "confusionMatrix"); //todo change to if not throws
            confusionMatrix = new double[numClasses][numClasses];
            for (int i = 0; i < numClasses; i++) {
                String[] vals = scan.nextLine().split(",");
                for (int j = 0; j < numClasses; j++)
                    confusionMatrix[i][j] = Double.parseDouble(vals[j]);
            }
        } catch (NoSuchElementException e) {
            System.err.println("Error reading metrics in allPerformanceMetricsFromString(str), scanner reached end prematurely");
            throw e;
        } catch (NumberFormatException e) {
            System.err.println("Error reading metrics in allPerformanceMetricsFromString(str), parsing metric value failed");
            throw e;
        }
    }



    /**
     * Concatenates the predictions of classifiers made on different folds on the data
     * into one results object
     *
     * If ClassifierResults ever gets split into separate classes for prediction and meta info,
     * this obviously gets cleaned up a lot
     *
     * @param cresults ClassifierResults[fold]
     * @return         single ClassifierResults object
     */
    public static ClassifierResults concatenateClassifierResults( /*fold*/ ClassifierResults[] cresults) throws Exception {
        return concatenateClassifierResults(new ClassifierResults[][]{cresults})[0];
    }

    /**
     * Concatenates the predictions of classifiers made on different folds on the data
     * into one results object per classifier.
     *
     * If ClassifierResults ever gets split into separate classes for prediction and meta info,
     * this obviously gets cleaned up a lot
     *
     * @param cresults ClassifierResults[classifier][fold]
     * @return         ClassifierResults[classifier]
     */
    public static ClassifierResults[] concatenateClassifierResults( /*classiifer*/ /*fold*/ ClassifierResults[][] cresults) throws Exception {
        ClassifierResults[] concatenatedResults = new ClassifierResults[cresults.length];
        for (int classifierid = 0; classifierid < cresults.length; classifierid++) {
            if (cresults[classifierid].length == 1) {
                concatenatedResults[classifierid] = cresults[classifierid][0];
            } else {
                ClassifierResults newCres = new ClassifierResults();
                for (int foldid = 0; foldid < cresults[classifierid].length; foldid++) {
                    ClassifierResults foldCres = cresults[classifierid][foldid];
                    for (int predid = 0; predid < foldCres.numInstances(); predid++) {
                        newCres.addPrediction(foldCres.getTrueClassValue(predid), foldCres.getProbabilityDistribution(predid), foldCres.getPredClassValue(predid), foldCres.getPredictionTime(predid), foldCres.getPredDescription(predid));
                        // TODO previously didnt copy of pred times and predictions
                        // not sure if there was any particular reason why i didnt,
                        // aside from saving space?
                    }
                }
                concatenatedResults[classifierid] = newCres;
            }
        }
        return concatenatedResults;
    }










    public static void main(String[] args) throws Exception {
        readWriteTest();
    }

    private static void readWriteTest() throws Exception {
        ClassifierResults res = new ClassifierResults();

        res.setClassifierName("testClassifier");
        res.setDatasetName("testDataset");
        //empty split
        //empty foldid
        res.setDescription("boop, guest");

        res.setParas("test,west,best");

        //acc handled internally
        res.setBuildTime(2);
        res.setTestTime(1);
        //empty benchmark
        //empty memory

        Random rng = new Random(0);
        for (int i = 0; i < 10; i++) { //obvs dists dont make much sense, not important here
            res.addPrediction(rng.nextInt(2), new double[] { rng.nextDouble(), rng.nextDouble()}, rng.nextInt(2), rng.nextInt(5)+1, "test,again");
        }

        res.finaliseResults();

        System.out.println(res.writeFullResultsToString());
        System.out.println("\n\n");

        res.writeFullResultsToFile("test.csv");

        ClassifierResults res2 = new ClassifierResults("test.csv");
        System.out.println(res2.writeFullResultsToString());
    }
}
