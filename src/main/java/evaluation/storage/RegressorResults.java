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

import fileIO.OutFile;
import utilities.DebugPrinting;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

/**
 * This is a container class for the storage of predictions and meta-info of a
 * regressor on a single set of instances (for example, the test set of a particular
 * resample of a particular dataset).
 *
 * Predictions can be stored via addPrediction(...) or addAllPredictions(...)
 * Currently, the information stored about each prediction is:
 *    - The true label value                            (double   getTrueClassValue(index))
 *    - The predicted label value                       (double   getPredClassValue(index))
 *    - The time taken to predict this instance id      (long     getPredictionTime(index))
 *    - An optional description of the prediction       (String   getPredDescription(index))
 *
 * The meta info stored is:
 *  [LINE 1 OF FILE]
 *    - get/setDatasetName(String)
 *    - get/setRegressorName(String)
 *    - get/setSplit(String)
 *    - get/setFoldId(String)
 *    - get/setTimeUnit(TimeUnit)
 *    - get/setDescription(String)
 *  [LINE 2 OF FILE]
 *    - get/setParas(String)
 *  [LINE 3 OF FILE]
 *   1 - getAccuracy() (calculated from predictions, only settable with a suitably annoying message)
 *   2 - get/setBuildTime(long)
 *   3 - get/setTestTime(long)
 *   4 - get/setBenchmarkTime(long)
 *   5 - get/setMemory(long)
 *   6 - get/setErrorEstimateMethod(String) (loosely formed, e.g. cv_10)
 *   7 - get/setErrorEstimateTime(long) (time to form an estimate from scratch, e.g. time of cv_10)
 *   8 - get/setBuildPlusEstimateTime(long) (time to train on full data, AND estimate error on it)
 *
 *  [REMAINING LINES: PREDICTIONS]
 *    - trueLabelVal, predLabelVal, [empty], predTime, [empty], predDescription
 *
 * Supports reading/writing of results from/to file, in the 'classifierResults file-format'
 *    - loadResultsFromFile(String path)
 *    - writeFullResultsToFile(String path)
 *
 * Supports recording of timings in different time units. Nanoseconds is the default.
 * Also supports the calculation of various evaluative performance metrics based on the predictions (MSE, MAE, R2 etc.)
 *
 * EXAMPLE USAGE:
 *          RegressorResults res = new RegressorResults();
 *          //set a particular timeunit, if using something other than nanos. Nanos recommended
 *          //set any meta info you want to keep, e.g regressorname, datasetname...
 *
 *          for (Instance inst : test) {
 *              long startTime = //time
 *              double dist = regressor.classifyInstance(inst);
 *              long predTime = //time - startTime
 *
 *              res.addPrediction(inst.classValue(), pred, predTime, ""); //description is optional
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
 * @author Matthew Middlehurst, adapted from ClassifierResults (James Large)
 */
public class RegressorResults extends EstimatorResults implements DebugPrinting, Serializable {

    /**
     * Print a message with the filename to stdout when a file cannot be loaded.
     * Can get very tiresome if loading thousands of files with some expected failures,
     * and a higher level process already summarises them, thus this option to
     * turn off the messages
     */
    public static boolean printOnFailureToLoad = true;

    /**
     * Print a message when result file MSE does not match calculated MSE.
     * Setting this to false will stop print outs from this check.
     */
    public static boolean mseTestPrint = true;


    //LINE 1: meta info, set by user
    private String regressorName = "";

    // datasetName

    // split

    // foldID

    // timeUnit

    // description

    //LINE 2: classifier setup/info, parameters. precise format is up to user.

    /**
     * For now, user dependent on the formatting of this string, and really, the contents of it.
     * It is notionally intended to contain the parameters of the clusterer used to produce the
     * attached predictions, but could also store other things as well.
     */
    private String paras = "No parameter info";

    //LINE 3: acc, buildTime, testTime, memoryUsage
    //simple summarative performance stats.

    /**
     * Calculated from the stored predictions, cannot be explicitly set by user
     */
    public double mse = -1;

    // buildTime

    // testTime

    // benchmarkTime

    // memoryUsage

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
     * For those classifiers that do not implement that, ClassifierExperiments.findOrSetupTrainEstimate(...) will set this value
     * as a wrapper around the entire evaluate call for whichever errorEstimateMethod is being used
     */
    private long errorEstimateTime = -1;

    /**
     * This measures the total time to build the classifier on the train data
     * AND to estimate the classifier's error on the same train data. For classifiers
     * that do not estimate their own error in some way during the build process,
     * this will simply be the buildTime and the errorEstimateTime added together.
     *
     * For classifiers that DO estimate their own error, buildPlusEstimateTime may
     * be anywhere between buildTime and buildTime+errorEstimateTime. Some or all of
     * the work needed to form an estimate (which the field errorEstimateTime measures from scratch)
     * may have already been accounted for by the buildTime
     */
    private long buildPlusEstimateTime = -1;

    //REMAINDER OF THE FILE - 1 prediction per line
    //raw performance data. currently just give parallel arrays
    private ArrayList<Double> trueLabelValues;
    private ArrayList<Double> predLabelValues;
    private ArrayList<Long> predTimes;
    private ArrayList<String> predDescriptions;

    //inferred/supplied dataset meta info
    private int numInstances;

    //calculated performance metrics
    //accuracy can be re-calced, as well as stored on line three in files
    public double mae;
    public double r2;
    public double mape;

    //self-management flags
    /**
     * essentially controls whether a RegressorResults object can have finaliseResults(trueLabelVals)
     * called upon it. In theory, every class using the RegressorResults object should make new
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

    //functional getters to retrieve info from a regressorresults object, initialised/stored here for convenience.
    //these are currently on used in PerformanceMetric.java, can take any results type as a hack to allow other
    //results in evaluation.
    public static final Function<EstimatorResults, Double> GETTER_MSE = (EstimatorResults cr) -> ((RegressorResults)cr).mse;
    public static final Function<EstimatorResults, Double> GETTER_MAE = (EstimatorResults cr) -> ((RegressorResults)cr).mae;
    public static final Function<EstimatorResults, Double> GETTER_R2 = (EstimatorResults cr) -> ((RegressorResults)cr).r2;
    public static final Function<EstimatorResults, Double> GETTER_MAPE = (EstimatorResults cr) -> ((RegressorResults)cr).mape;



    /*********************************
     *
     *       CONSTRUCTORS
     *
     */

    /**
     * Create an empty RegressorResults object.
     */
    public RegressorResults() {
        trueLabelValues = new ArrayList<>();
        predLabelValues = new ArrayList<>();
        predTimes = new ArrayList<>();
        predDescriptions = new ArrayList<>();

        finalised = false;
    }

    /**
     * Load a RegressorResults object from the file at the specified path
     */
    public RegressorResults(String filePathAndName) throws Exception {
        loadResultsFromFile(filePathAndName);
    }

    /**
     * Create a RegressorResults object with complete predictions (equivalent to addAllPredictions()). The results are
     * FINALISED after initialisation. Meta info such as regressor name, datasetname... can still be set after construction.
     *
     * The descriptions array argument may be null, in which case the descriptions are stored as empty strings.
     *
     * All other arguments are required in full, however
     */
    public RegressorResults(double[] trueLabelVals, double[] predictions, long[] predTimes, String[] descriptions) throws Exception {
        this.trueLabelValues = new ArrayList<>();
        this.predLabelValues = new ArrayList<>();
        this.predTimes = new ArrayList<>();
        this.predDescriptions = new ArrayList<>();

        addAllPredictions(trueLabelVals, predictions, predTimes, descriptions);
        finaliseResults();
    }


    /***********************
     *
     *      DATASET META INFO
     *
     *
     */

    public int numInstances() {
        if (numInstances <= 0)
            inferNumInstances();
        return numInstances;
    }

    private void inferNumInstances() {
        this.numInstances = predLabelValues.size();
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
     *  Just basic descriptive stuff, nothing fancy goign on here
     *
     */

    public String getRegressorName() { return regressorName; }

    public void setRegressorName(String regressorName) { this.regressorName = regressorName; }



    /*****************************
     *
     *     LINE 2 GETS/SETS
     *
     */


    public String getParas() { return paras; }

    public void setParas(String paras) { this.paras = paras; }



    /*****************************
     *
     *     LINE 3 GETS/SETS
     *
     */


    @Override
    public double getAcc() {
        if (mse < 0)
            calculateMSE();
        return mse;
    }

    private void calculateMSE() {
        if (trueLabelValues == null || trueLabelValues.isEmpty() || trueLabelValues.get(0) == -1) {
            System.out.println("**getAcc():calculateAcc() no true class values supplied yet, cannot calculate accuracy");
            return;
        }

        int size = predLabelValues.size();
        double sum = 0;
        for (int i = 0; i < size; i++) {
            sum += Math.abs(Math.pow(trueLabelValues.get(i) - predLabelValues.get(i), 2));
        }

        mse = sum / size;
    }

    public long getBuildTime() { return buildTime; }

    public long getBuildTimeInNanos() { return timeUnit.toNanos(buildTime); }

    /**
     * @throws Exception if buildTime is less than 1
     */
    public void setBuildTime(long buildTime) {
        if (errorOnTimingOfZero && buildTime < 1)
            throw new RuntimeException("Build time passed has invalid value, " + buildTime + ". If greater resolution" +
                                           " is needed, "
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

    public long getBenchmarkTime() {
        return benchmarkTime;
    }

    public void setBenchmarkTime(long benchmarkTime) {
        this.benchmarkTime = benchmarkTime;
    }

    public String getErrorEstimateMethod() {
        return errorEstimateMethod;
    }

    public void setErrorEstimateMethod(String errorEstimateMethod) {
        this.errorEstimateMethod = errorEstimateMethod;
    }

    public long getErrorEstimateTime() {
        return errorEstimateTime;
    }

    public long getErrorEstimateTimeInNanos() {
        return timeUnit.toNanos(errorEstimateTime);
    }

    public void setErrorEstimateTime(long errorEstimateTime) {
        this.errorEstimateTime = errorEstimateTime;
    }

    public long getBuildPlusEstimateTime() {
        return buildPlusEstimateTime;
    }

    public long getBuildPlusEstimateTimeInNanos() {
        return timeUnit.toNanos(buildPlusEstimateTime);
    }

    public void setBuildPlusEstimateTime(long buildPlusEstimateTime) {
        this.buildPlusEstimateTime = buildPlusEstimateTime;
    }



    /****************************
     *
     *    PREDICTION STORAGE
     *
     */
    /**
     * Will update the internal prediction info using the values passed.
     *
     * The description argument may be null, however all other arguments are required in full
     *
     * The true label is missing, however can be added in one go later with the
     * method finaliseResults(double[] trueClassVals)
     */
    public void addPrediction(double predictedClass, long predictionTime, String description) throws RuntimeException {
        predLabelValues.add(predictedClass);
        predTimes.add(predictionTime);

        if (testTime == -1)
            testTime = predictionTime;
        else
            testTime += predictionTime;

        if (description == null)
            predDescriptions.add("");
        else
            predDescriptions.add(description);

        numInstances++;
    }

    /**
     * Will update the internal prediction info using the values passed.
     *
     * The description argument may be null, however all other arguments are required in full
     */
    public void addPrediction(double trueClassVal, double predictedClass, long predictionTime, String description) throws RuntimeException {
        addPrediction(predictedClass,predictionTime,description);
        trueLabelValues.add(trueClassVal);
    }


    /**
     * Adds all the prediction info onto this RegressorResults object. Does NOT finalise the results,
     * such that (e.g) predictions from multiple dataset splits can be added to the same object if wanted
     *
     * The description argument may be null, however all other arguments are required in full
     */
    public void addAllPredictions(double[] trueLabelVals, double[] predictions, long[] predTimes, String[] descriptions) throws RuntimeException {
        assert(trueLabelVals.length == predictions.length);
        assert(trueLabelVals.length == predTimes.length);

        if (descriptions != null)
            assert(trueLabelVals.length == descriptions.length);

        for (int i = 0; i < trueLabelVals.length; i++) {
            if (descriptions == null)
                addPrediction(trueLabelVals[i], predictions[i], predTimes[i], null);
            else
                addPrediction(trueLabelVals[i], predictions[i], predTimes[i], descriptions[i]);
        }
    }

    /**
     * Adds all the prediction info onto this RegressorResults object. Does NOT finalise the results,
     * such that (e.g) predictions from multiple dataset splits can be added to the same object if wanted
     *
     * True label values can later be supplied (ALL IN ONE GO, if working to the above example usage..) using
     * finaliseResults(double[] testClassVals)
     *
     * The description argument may be null, however all other arguments are required in full
     */
    public void addAllPredictions(double[] predictions,long[] predTimes, String[] descriptions ) throws RuntimeException {
        assert(predictions.length == predTimes.length);

        if (descriptions != null)
            assert(predictions.length == descriptions.length);

        for (int i = 0; i < predictions.length; i++) {
            if (descriptions == null)
                addPrediction(predictions[i], predTimes[i], "");
            else
                addPrediction(predictions[i], predTimes[i], descriptions[i]);
        }
    }

    /**
     * Will perform some basic validation to make sure that everything is here
     * that is expected, and compute the accuracy etc ready for file writing.
     *
     * Typical usage: results.finaliseResults(instances.attributeToDoubleArray(instances.classIndex()))
     */
    public void finaliseResults(double[] testLabelVals) throws Exception {
        if (finalised) {
            System.out.println("finaliseResults(double[] testLabelVals): Results already finalised, skipping re-finalisation");
            return;
        }

        if (testLabelVals.length != predLabelValues.size())
            throw new Exception("finaliseTestResults(double[] testLabelVals): Number of predictions "
                    + "made and number of true class values passed do not match");

        trueLabelValues = new ArrayList<>();
        for(double d:testLabelVals)
            trueLabelValues.add(d);

        finaliseResults();
    }


    /**
     * Will perform some basic validation to make sure that everything is here
     * that is expected, and compute the accuracy etc ready for file writing.
     *
     * You can use this method, instead of the version that takes the double[] testLabelVals
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

        if (predLabelValues == null || predLabelValues.isEmpty())
            throw new Exception("finaliseTestResults(): no predictions stored for this module");

        calculateMSE();

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
    public ArrayList<Double> getTrueLabelVals() {
        return trueLabelValues;
    }

    public double[] getTrueLabelValsAsArray(){
        double[] d=new double[trueLabelValues.size()];
        int i=0;
        for(double x: trueLabelValues)
            d[i++]=x;
        return d;
    }

    public double getTrueLabelValue(int index){
        return trueLabelValues.get(index);
    }

    public ArrayList<Double> getPredLabelVals(){
        return predLabelValues;
    }

    public double[] getPredLabelValsAsArray(){
        double[] d=new double[predLabelValues.size()];
        int i=0;
        for(double x: predLabelValues)
            d[i++]=x;
        return d;
    }

    public double getPredLabelValue(int index){
        return predLabelValues.get(index);
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

    @Override
    public void cleanPredictionInfo() {
        predLabelValues = null;
        trueLabelValues = null;
        predTimes = null;
        predDescriptions = null;
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
     * Reads and STORES the prediction in this RegressorResults object
     *
     * INCREMENTS NUMINSTANCES
     *
     * [true],[pred], ,[predTime], ,[description until end of line, may have commas in it]
     */
    private void instancePredictionFromString(String predLine) {
        String[] split=predLine.split(",");

        //collect actual/predicted label
        double trueLabelVal = Double.parseDouble(split[0].trim());
        double predLabelVal = Double.parseDouble(split[1].trim());

        //collect timings
        long predTime = -1;
        final int timingInd = 3; //actual, predicted, space, timing
        if (split.length > timingInd)
            predTime = Long.parseLong(split[timingInd].trim());

        //collect description
        String description = "";
        final int descriptionInd = timingInd + 1 + 1; //actual, predicted, space, timing, space, description
        if (split.length > descriptionInd) {
            description = split[descriptionInd];

            //no reason currently why the description passed cannot have commas in it,
            //might be a natural way to separate it in to different parts.
            //description reall just fills up the remainder of the line.
            for (int i = descriptionInd+1; i < split.length; i++)
                description += "," + split[i];
        }


        addPrediction(trueLabelVal, predLabelVal, predTime, description);
    }

    private void instancePredictionsFromScanner(Scanner in) throws Exception {
        while (in.hasNext()) {
            String line = in.nextLine();
            //may be trailing empty lines at the end of the file
            if (line == null || line.equals(""))
                break;

            instancePredictionFromString(line);
        }

        calculateMSE();
    }

    /**
     * [true],[pred], ,[predTime], ,[description until end of line, may have commas in it]
     */
    private String instancePredictionToString(int i) {
        StringBuilder sb = new StringBuilder();

        sb.append(trueLabelValues.get(i).intValue()).append(",");
        sb.append(predLabelValues.get(i).intValue());

        //timing
        sb.append(",,").append(predTimes.get(i)); //<empty space>, timing

        //description
        sb.append(",,").append(predDescriptions.get(i)); //<empty space>, description

        return sb.toString();
    }

    public String instancePredictionsToString() throws Exception{
        if (trueLabelValues == null || trueLabelValues.size() == 0 || trueLabelValues.get(0) == -1)
            throw new Exception("No true class value stored, call finaliseResults(double[] trueClassVal)");

        if(numInstances() > 0 && (predLabelValues.size() == trueLabelValues.size())){
            StringBuilder sb=new StringBuilder("");

            for(int i=0;i<numInstances();i++){
                sb.append(instancePredictionToString(i));

                if(i<numInstances()-1)
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
        s += "Mean Squared Error: " + mse;
        s += "\nMean Absolute error: " + mae;
        s += "\nR² Score: " + r2;
        s += "\nMean Absolute Percentage Error: " + mape;
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

        datasetName = parts[0];
        regressorName = parts[1];
        split = parts[2];
        foldID = Integer.parseInt(parts[3]);
        setTimeUnitFromString(parts[4]);

        //nothing stopping the description from having its own commas in it, just read until end of line
        for (int i = 5; i < parts.length; i++)
            description += "," + parts[i];
    }

    private String generateFirstLine() {
        return datasetName + "," + regressorName + "," + split + "," + foldID + "," + getTimeUnitAsString() +  "," + description;
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

        mse = Double.parseDouble(parts[0]);
        buildTime = Long.parseLong(parts[1]);
        testTime = Long.parseLong(parts[2]);
        benchmarkTime = Long.parseLong(parts[3]);
        memoryUsage = Long.parseLong(parts[4]);
        errorEstimateMethod = parts[5];
        errorEstimateTime = Long.parseLong(parts[6]);
        buildPlusEstimateTime = Long.parseLong(parts[7]);

        return mse;
    }

    private String generateThirdLine() {
        String res = mse
            + "," + buildTime
            + "," + testTime
            + "," + benchmarkTime
            + "," + memoryUsage
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
            trueLabelValues = new ArrayList<>();
            predLabelValues = new ArrayList<>();
            predTimes = new ArrayList<>();
            predDescriptions = new ArrayList<>();
            numInstances = 0;
            mse = -1;
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
            double reportedTestMSE = parseThirdLine(inf.nextLine());

            //parse predictions
            instancePredictionsFromScanner(inf);

            //acts as a basic form of verification, does the mse reported on line 3 align with
            //the mse calculated while reading predictions
            double eps = 1.e-8;
            if (mseTestPrint && Math.abs(reportedTestMSE - mse) > eps) {
                System.out.println("Calculated MSE (" + mse + ") differs from written MSE (" + reportedTestMSE + ") "
                        + "by more than eps (" + eps + "). File = " + path + ". numinstances = " + numInstances + ".");
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

        if (mse < 0)
            calculateMSE();

        mae = findMAE();
        r2 = findR2();
        mape = findMAPE();

        medianPredTime = findMedianPredTime(predTimes);

        allStatsFound = true;
    }

    public double findMAE() {
        int size = predLabelValues.size();
        double sum = 0;
        for (int i = 0; i < size; i++) {
            sum += Math.abs(trueLabelValues.get(i) - predLabelValues.get(i));
        }

        return sum / size;
    }

    public double findR2() {
        int size = predLabelValues.size();
        double labelAverage = 0;
        for (int i = 0; i < size; i++) {
            labelAverage += trueLabelValues.get(i);
        }
        labelAverage /= size;

        double sum1 = 0;
        for (int i = 0; i < size; i++) {
            sum1 += Math.pow(trueLabelValues.get(i) - predLabelValues.get(i), 2);
        }

        double sum2 = 0;
        for (int i = 0; i < size; i++) {
            sum2 += Math.pow(trueLabelValues.get(i) - labelAverage, 2);
        }

        if (sum2 == 0)
            sum2 = 2.22044605e-16;

        return 1 - sum1 / sum2;
    }

    public double findMAPE() {
        int size = predLabelValues.size();
        double sum = 0;
        for (int i = 0; i < size; i++) {
            sum += Math.abs(trueLabelValues.get(i) - predLabelValues.get(i)) / Math.max(2.22044605e-16, Math.abs(trueLabelValues.get(i)));
        }

        return sum / size;
    }

    /**
     * Will calculate all the metrics that can be found from the prediction information
     * stored in this object, UNLESS this object has been finalised (finaliseResults(..)) AND
     * has already had it's stats found (findAllStats()), e.g. if it has already been called
     * by another process.
     *
     * In this latter case, this method does nothing.
     */
    @Override
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
     * Concatenates the predictions of regressors made on different folds on the data
     * into one results object
     *
     * If RegressorResults ever gets split into separate classes for prediction and meta info,
     * this obviously gets cleaned up a lot
     *
     * @param rresults RegressorResults[fold]
     * @return         single RegressorResults object
     */
    public static RegressorResults concatenateRegressorResults( /*fold*/ RegressorResults[] rresults) throws Exception {
        return concatenateRegressorResults(new RegressorResults[][]{rresults})[0];
    }

    /**
     * Concatenates the predictions of regressors made on different folds on the data
     * into one results object per regressor.
     *
     * If RegressorResults ever gets split into separate classes for prediction and meta info,
     * this obviously gets cleaned up a lot
     *
     * @param rresults RegressorResults[regressor][fold]
     * @return         RegressorResults[regressor]
     */
    public static RegressorResults[] concatenateRegressorResults( /*regressor*/ /*fold*/ RegressorResults[][] rresults) throws Exception {
        RegressorResults[] concatenatedResults = new RegressorResults[rresults.length];
        for (int regressorid = 0; regressorid < rresults.length; regressorid++) {
            if (rresults[regressorid].length == 1) {
                concatenatedResults[regressorid] = rresults[regressorid][0];
            } else {
                RegressorResults newCres = new RegressorResults();
                for (int foldid = 0; foldid < rresults[regressorid].length; foldid++) {
                    RegressorResults foldCres = rresults[regressorid][foldid];
                    for (int predid = 0; predid < foldCres.numInstances(); predid++) {
                        newCres.addPrediction(foldCres.getTrueLabelValue(predid), foldCres.getPredLabelValue(predid), foldCres.getPredictionTime(predid), foldCres.getPredDescription(predid));
                    }
                }
                concatenatedResults[regressorid] = newCres;
            }
        }
        return concatenatedResults;
    }

    public static void main(String[] args) throws Exception {
        RegressorResults cr = new RegressorResults();
        Collections.addAll(cr.trueLabelValues, 3., -0.5, 2., 7., -2., -2., -2., 1., 10., 1e6);
        Collections.addAll(cr.predLabelValues, 2.5, 0.0, 2., 8., -2., -2., -2., 0.9, 15., 1.2e6);
        Collections.addAll(cr.predTimes, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L);
        cr.numInstances = 15;
        cr.findAllStats();
        System.out.println(cr.statsToString());
    }
}
