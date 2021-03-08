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
package tsml.classifiers.interval_based;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.storage.ClassifierResults;
import evaluation.tuning.ParameterSpace;
import experiments.data.DatasetLoading;
import machine_learning.classifiers.ContinuousIntervalTree;
import machine_learning.classifiers.ContinuousIntervalTree.Interval;
import tsml.classifiers.*;
import tsml.data_containers.TSCapabilities;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import tsml.transformers.*;
import utilities.ClassifierTools;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.*;

import java.io.File;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.Function;

import static utilities.ArrayUtilities.sum;
import static utilities.StatisticalUtilities.median;
import static utilities.Utilities.argMax;

/**
 * Implementation of the catch22 Interval Forest algorithm with extra representations and summary stats.
 *
 * @author Matthew Middlehurst
 **/
public class DrCIF extends EnhancedAbstractClassifier implements TechnicalInformationHandler, TrainTimeContractable,
        Checkpointable, Tuneable, MultiThreadable {

    /**
     * Paper defining DrCIF.
     *
     * @return TechnicalInformation for DrCIF
     */
    @Override //TechnicalInformationHandler
    public TechnicalInformation getTechnicalInformation() {
        //TODO update
//        TechnicalInformation result;
//        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
//        result.setValue(TechnicalInformation.Field.AUTHOR, "M. Middlehurst, J. Large and A. Bagnall");
//        result.setValue(TechnicalInformation.Field.TITLE, "The Canonical Interval Forest (CIF) Classifier for " +
//                "Time Series Classifciation");
//        result.setValue(TechnicalInformation.Field.YEAR, "2020");
//        return result;
        return null;
    }

    /** Primary parameters potentially tunable */
    private int numClassifiers = 500;

    /** Amount of attributes to be subsampled and related data storage. */
    private int attSubsampleSize = 10;
    private int numAttributes = 29;
    private int startNumAttributes;
    private ArrayList<int[]> subsampleAtts;

    /** Normalise outlier catch22 features which break on data not normalised */
    private boolean outlierNorm = true;

    /** Use STSF features as well as catch22 features */
    private boolean useSummaryStats = true;

    /** IntervalsFinders sets parameter values in buildClassifier if -1. */
    /** Num intervals selected per representation per tree built */
    private int[] numIntervals;
    private transient Function<Integer,Integer> numIntervalsFinder;

    /** Secondary parameters */
    /** Mainly there to avoid single item intervals, which have no slope or std dev */
    /** Min defaults to 3, Max defaults to m/2 */
    private int[] minIntervalLength;
    private transient Function<Integer,Integer> minIntervalLengthFinder;
    private int[] maxIntervalLength;
    private transient Function<Integer,Integer> maxIntervalLengthFinder;

    /** Ensemble members of base classifier, default to TimeSeriesTree */
    private ArrayList<Classifier> trees;
    private Classifier base= new ContinuousIntervalTree();

    /** for each classifier i representation r attribute a interval j  starts at intervals[i][r][a][j][0] and
     ends  at  intervals[i][r][j][1] */
    private ArrayList<int[][][]> intervals;

    /**Holding variable for test classification in order to retain the header info*/
    private Instances testHolder;

    /** Flags and data required if Bagging **/
    private boolean bagging = false;
    private int[] oobCounts;
    private double[][] trainDistributions;

    /** Flags and data required if Checkpointing **/
    private boolean checkpoint = false;
    private String checkpointPath;
    private long checkpointTime = 0;
    private long lastCheckpointTime = 0;
    private long checkpointTimeDiff = 0;
    private boolean internalContractCheckpointHandling = true;

    /** Flags and data required if Contracting **/
    private boolean trainTimeContract = false;
    private long contractTime = 0;
    private int maxClassifiers = 500;

    /** Multithreading **/
    private int numThreads = 1;
    private boolean multiThread = false;
    private ExecutorService ex;

    /** data information **/
    private int numInstances;

    /** Multivariate **/
    private int numDimensions;
    private ArrayList<int[][]> intervalDimensions;

    /** Transformer used to obtain catch22 features **/
    private transient Catch22 c22;

    /** Transformers used for other representations **/
    private transient Fast_FFT fft;
    private transient Differences di;

    protected static final long serialVersionUID = 1L;

    /**
     * Default constructor for DrCIF. Can estimate own performance.
     */
    public DrCIF(){
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
    }

    /**
     * Set the number of trees to be built.
     *
     * @param t number of trees
     */
    public void setNumTrees(int t){
        numClassifiers = t;
    }

    /**
     * Set the number of attributes to be subsampled per tree.
     *
     * @param a number of attributes sumsampled
     */
    public void setAttSubsampleSize(int a) {
        attSubsampleSize = a;
    }

    /**
     * Set whether to use the original TSF statistics as well as catch22 features.
     *
     * @param b boolean to use summary stats
     */
    public void setUseSummaryStats(boolean b) {
        useSummaryStats = b;
    }

    /**
     * Set a function for finding the number of intervals randomly selected per tree.
     *
     * @param f a function for the number of intervals
     */
    public void setNumIntervalsFinder(Function<Integer,Integer> f){
        numIntervalsFinder = f;
    }

    /**
     * Set a function for finding the min interval length for randomly selected intervals.
     *
     * @param f a function for min interval length
     */
    public void setMinIntervalLengthFinder(Function<Integer,Integer> f){
        minIntervalLengthFinder = f;
    }

    /**
     * Set a function for finding the max interval length for randomly selected intervals.
     *
     * @param f a function for max interval length
     */
    public void setMaxIntervalLengthFinder(Function<Integer,Integer> f){
        maxIntervalLengthFinder = f;
    }

    /**
     * Set whether to normalise the outlier catch22 features.
     *
     * @param b boolean to set outlier normalisation
     */
    public void setOutlierNorm(boolean b) {
        outlierNorm = b;
    }

    /**
     * Sets the base classifier for the ensemble.
     *
     * @param c a base classifier constructed elsewhere and cloned into ensemble
     */
    public void setBaseClassifier(Classifier c){
        base=c;
    }

    /**
     * Set whether to perform bagging with replacement.
     *
     * @param b boolean to set bagging
     */
    public void setBagging(boolean b){
        bagging = b;
    }

    /**
     * Outputs DrCIF parameters information as a String.
     *
     * @return String written to results files
     */
    @Override //SaveParameterInfo
    public String getParameters() {
        int nt = numClassifiers;
        if (trees != null) nt = trees.size();
        String temp=super.getParameters()+",numTrees,"+nt+",attSubsampleSize,"+attSubsampleSize+
                ",outlierNorm,"+outlierNorm+",basicSummaryStats,"+useSummaryStats+",numIntervals,"+
                Arrays.toString(numIntervals).replace(',', ';')+",minIntervalLength,"+
                Arrays.toString(minIntervalLength).replace(',', ';')+",maxIntervalLength,"+
                Arrays.toString(maxIntervalLength).replace(',', ';')+",baseClassifier,"+
                base.getClass().getSimpleName()+",bagging,"+ bagging+",estimator,"+estimator.name()+
                ",contractTime,"+contractTime;
        return temp;
    }

    /**
     * Returns the capabilities for DrCIF. These are that the
     * data must be numeric or relational, with no missing and a nominal class
     *
     * @return the capabilities of DrCIF
     */
    @Override //AbstractClassifier
    public Capabilities getCapabilities(){
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.setMinimumNumberInstances(2);

        // attributes
        result.enable(Capabilities.Capability.RELATIONAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        return result;
    }

    /**
     * Returns the time series capabilities for DrCIF. These are that the
     * data must be equal length, with no missing values
     *
     * @return the time series capabilities of DrCIF
     */
    public TSCapabilities getTSCapabilities(){
        TSCapabilities capabilities = new TSCapabilities();
        capabilities.enable(TSCapabilities.EQUAL_LENGTH)
                .enable(TSCapabilities.MULTI_OR_UNIVARIATE)
                .enable(TSCapabilities.NO_MISSING_VALUES);
        return capabilities;
    }

    /**
     * Build the DrCIF classifier.
     *
     * @param data TimeSeriesInstances object
     * @throws Exception unable to train model
     */
    @Override
    public void buildClassifier(TimeSeriesInstances data) throws Exception {
        /** Build Stage:
         *  Builds the final classifier with or without bagging.
         */
        trainResults = new ClassifierResults();
        rand.setSeed(seed);
        numClasses = data.numClasses();
        trainResults.setClassifierName(getClassifierName());
        trainResults.setBuildTime(System.nanoTime());
        // can classifier handle the data?
        getTSCapabilities().test(data);

        c22 = new Catch22();
        c22.setOutlierNormalise(outlierNorm);

        TimeSeriesInstances[] representations = new TimeSeriesInstances[3];
        representations[0] = data;
        fft = new Fast_FFT();
        fft.nearestPowerOF2(representations[0].getMaxLength());
        representations[1] = fft.transform(representations[0]);
        di = new Differences();
        di.setSubtractFormerValue(true);
        representations[2] = di.transform(representations[0]);

        File file = new File(checkpointPath + "DrCIF" + seed + ".ser");
        //if checkpointing and serialised files exist load said files
        if (checkpoint && file.exists()){
            //path checkpoint files will be saved to
            if(debug)
                System.out.println("Loading from checkpoint file");
            loadFromFile(checkpointPath + "DrCIF" + seed + ".ser");
        }
        //initialise variables
        else {
            numInstances = data.numInstances();
            numDimensions = data.getMaxNumChannels();

            numIntervals = new int[representations.length];
            minIntervalLength = new int[representations.length];
            maxIntervalLength = new int[representations.length];
            for (int r = 0; r < 3; r++) {
                if (numIntervalsFinder == null){
                    numIntervals[r] = (int)(4 + (Math.sqrt(representations[r].getMaxLength()) *
                            Math.sqrt(numDimensions))/3);
                }
                else {
                    numIntervals[r] = numIntervalsFinder.apply(representations[0].getMaxLength());
                }

                if (minIntervalLengthFinder == null) {
                    minIntervalLength[r] = 3;
                } else {
                    minIntervalLength[r] = minIntervalLengthFinder.apply(representations[r].getMaxLength());
                }
                if (minIntervalLength[r] < 3) {
                    minIntervalLength[r] = 3;
                }
                if (representations[r].getMaxLength() <= minIntervalLength[r]) {
                    minIntervalLength[r] = representations[r].getMaxLength() / 2;
                }

                if (maxIntervalLengthFinder == null) {
                    maxIntervalLength[r] = representations[r].getMaxLength() / 2;
                } else {
                    maxIntervalLength[r] = maxIntervalLengthFinder.apply(representations[r].getMaxLength());
                }
                if (maxIntervalLength[r] > representations[r].getMaxLength()) {
                    maxIntervalLength[r] = representations[r].getMaxLength();
                }

                if (maxIntervalLength[r] < minIntervalLength[r]) {
                    maxIntervalLength[r] = minIntervalLength[r];
                }
            }

            if (!useSummaryStats){
                numAttributes = 22;
            }

            startNumAttributes = numAttributes;
            subsampleAtts = new ArrayList<>();

            if (attSubsampleSize < numAttributes) {
                numAttributes = attSubsampleSize;
            }

            //Set up for Bagging if required
            if(bagging && getEstimateOwnPerformance()) {
                trainDistributions = new double[numInstances][numClasses];
                oobCounts = new int[numInstances];
            }

            //cancel loop using time instead of number built.
            if (trainTimeContract){
                numClassifiers = maxClassifiers;
                trees = new ArrayList<>();
                intervals = new ArrayList<>();
            }
            else{
                trees = new ArrayList<>(numClassifiers);
                intervals = new ArrayList<>(numClassifiers);
            }

            intervalDimensions = new ArrayList<>();
        }

        if (multiThread) {
            ex = Executors.newFixedThreadPool(numThreads);
            if (checkpoint) System.out.println("Unable to checkpoint until end of build when multi threading.");
        }

        //Set up instances size and format.
        ArrayList<Attribute> atts=new ArrayList<>();
        String name;
        for(int j = 0; j < sum(numIntervals)*numAttributes; j++){
            name = "F"+j;
            atts.add(new Attribute(name));
        }
        //Get the class values as an array list
        ArrayList<String> vals = new ArrayList<>(numClasses);
        for(int j = 0; j < numClasses; j++)
            vals.add(Integer.toString(j));
        atts.add(new Attribute("cls", vals));
        //create blank instances with the correct class value
        Instances result = new Instances("Tree", atts, numInstances);
        result.setClassIndex(result.numAttributes()-1);
        for(int i = 0; i < numInstances; i++){
            DenseInstance in = new DenseInstance(result.numAttributes());
            in.setValue(result.numAttributes()-1, data.get(i).getLabelIndex());
            result.add(in);
        }

        testHolder = new Instances(result,1);
        DenseInstance in = new DenseInstance(testHolder.numAttributes());
        in.setValue(testHolder.numAttributes()-1, -1);
        testHolder.add(in);

        if (multiThread){
            multiThreadBuildDrCIF(representations, result);
        }
        else{
            buildDrCIF(representations, result);
        }

        if(trees.size() == 0){//Not enough time to build a single classifier
            throw new Exception((" ERROR in DrCIF, no trees built, contract time probably too low. Contract time = "
                    + contractTime));
        }

        if(checkpoint) {
            saveToFile(checkpointPath);
        }

        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        trainResults.setBuildTime(System.nanoTime() - trainResults.getBuildTime() - checkpointTimeDiff
                - trainResults.getErrorEstimateTime());

        if(getEstimateOwnPerformance()){
            long est1 = System.nanoTime();
            estimateOwnPerformance(data);
            long est2 = System.nanoTime();
            trainResults.setErrorEstimateTime(est2 - est1 + trainResults.getErrorEstimateTime());
        }
        trainResults.setBuildPlusEstimateTime(trainResults.getBuildTime() + trainResults.getErrorEstimateTime());
        trainResults.setParas(getParameters());
        printLineDebug("*************** Finished DrCIF Build with " + trees.size() + " Trees built in " +
                trainResults.getBuildTime()/1000000000 + " Seconds  ***************");
    }

    /**
     * Build the DrCIF classifier.
     *
     * @param data weka Instances object
     * @throws Exception unable to train model
     */
    @Override //AbstractClassifier
    public void buildClassifier(Instances data) throws Exception {
        buildClassifier(Converter.fromArff(data));
    }

    /**
     * Build the DrCIF classifier
     * For each base classifier
     *     generate random intervals
     *     do the transfrorms
     *     build the classifier
     *
     * @throws Exception unable to build DrCIF
     */
    public void buildDrCIF(TimeSeriesInstances[] representations, Instances result) throws Exception {
        double[][][][] dimensions =  new double[numInstances][representations.length][][];
        for (int r = 0; r < representations.length; r++){
            double[][][] arr = representations[r].toValueArray();
            for (int n = 0; n < numInstances; n++) {
                dimensions[n][r] = arr[n];
            }
        }

        while(withinTrainContract(trainResults.getBuildTime()) && trees.size() < numClassifiers) {
            int i = trees.size();

            //1. Select random intervals for tree i
            int[][][] interval = new int[representations.length][][];
            for (int r = 0; r < representations.length; r++) {
                interval[r] = new int[numIntervals[r]][2];

                for (int j = 0; j < numIntervals[r]; j++) {
                    if (rand.nextBoolean()) {
                        interval[r][j][0] = rand.nextInt(representations[r].getMaxLength() -
                                minIntervalLength[r]); //Start point

                        int range = Math.min(representations[r].getMaxLength() - interval[r][j][0],
                                maxIntervalLength[r]);
                        int length;
                        if (range - minIntervalLength[r] == 0) length = minIntervalLength[r];
                        else length = rand.nextInt(range - minIntervalLength[r]) + minIntervalLength[r];
                        interval[r][j][1] = interval[r][j][0] + length;
                    } else {
                        interval[r][j][1] = rand.nextInt(representations[r].getMaxLength() -
                                minIntervalLength[r]) + minIntervalLength[r]; //Start point

                        int range = Math.min(interval[r][j][1], maxIntervalLength[r]);
                        int length;
                        if (range - minIntervalLength[r] == 0) length = minIntervalLength[r];
                        else length = rand.nextInt(range - minIntervalLength[r]) + minIntervalLength[r];
                        interval[r][j][0] = interval[r][j][1] - length;
                    }
                }
            }

            //If bagging find instances with replacement
            int[] instInclusions = null;
            boolean[] inBag = null;
            if (bagging) {
                inBag = new boolean[numInstances];
                instInclusions = new int[numInstances];

                for (int n = 0; n < numInstances; n++) {
                    instInclusions[rand.nextInt(numInstances)]++;
                }

                for (int n = 0; n < numInstances; n++) {
                    if (instInclusions[n] > 0) {
                        inBag[n] = true;
                    }
                }
            }

            //find attributes to subsample
            ArrayList<Integer> arrl = new ArrayList<>(startNumAttributes);
            for (int n = 0; n < startNumAttributes; n++){
                arrl.add(n);
            }

            int[] subsampleAtt = new int[numAttributes];
            for (int n = 0; n < numAttributes; n++){
                subsampleAtt[n] = arrl.remove(rand.nextInt(arrl.size()));
            }

            //find dimensions for each interval
            int[][] intervalDimension = new int[representations.length][];
            for (int r = 0; r < representations.length; r++) {
                intervalDimension[r] = new int[numIntervals[r]];
                for (int n = 0; n < numIntervals[r]; n++) {
                    intervalDimension[r][n] = rand.nextInt(numDimensions);
                }
                Arrays.sort(intervalDimension[r]);
            }

            //For bagging
            int instIdx = 0;
            int lastIdx = -1;

            //2. Generate and store attributes
            for (int k = 0; k < numInstances; k++) {
                //For each instance
                if (bagging) {
                    boolean sameInst = false;

                    while (true) {
                        if (instInclusions[instIdx] == 0) {
                            instIdx++;
                        } else {
                            instInclusions[instIdx]--;

                            if (instIdx == lastIdx) {
                                result.set(k, new DenseInstance(result.instance(k - 1)));
                                sameInst = true;
                            } else {
                                lastIdx = instIdx;
                            }

                            break;
                        }
                    }

                    if (sameInst) continue;

                    result.instance(k).setValue(result.classIndex(), representations[0].get(instIdx).getLabelIndex());
                } else {
                    instIdx = k;
                }

                int p = 0;
                for (int r = 0; r < representations.length; r++) {
                    for (int j = 0; j < numIntervals[r]; j++) {
                        //extract the interval
                        double[] series = dimensions[instIdx][r][intervalDimension[r][j]];
                        double[] intervalArray = Arrays.copyOfRange(series, interval[r][j][0], interval[r][j][1] + 1);

                        //process features
                        for (int a = 0; a < numAttributes; a++) {
                            if (subsampleAtt[a] < 22) {
                                result.instance(k).setValue(p,
                                        c22.getSummaryStatByIndex(subsampleAtt[a], j, intervalArray));
                            }
                            else {
                                result.instance(k).setValue(p,
                                        FeatureSet.calcFeatureByIndex(subsampleAtt[a], interval[r][j][0],
                                                interval[r][j][1], series));
                            }

                            p++;
                        }
                    }
                }
            }

            //3. Create and build tree using all the features. Feature selection
            Classifier tree = AbstractClassifier.makeCopy(base);
            if (seedClassifier && tree instanceof Randomizable)
                ((Randomizable) tree).setSeed(seed * (i + 1));

            tree.buildClassifier(result);

            if (bagging && getEstimateOwnPerformance()) {
                long t1 = System.nanoTime();

                if (base instanceof ContinuousIntervalTree){
                    for (int n = 0; n < numInstances; n++) {
                        if (inBag[n])
                            continue;

                        double[] newProbs = ((ContinuousIntervalTree) tree).distributionForInstance(dimensions[n],
                                functions, interval, subsampleAtt, intervalDimension);
                        oobCounts[n]++;
                        for (int k = 0; k < newProbs.length; k++)
                            trainDistributions[n][k] += newProbs[k];
                    }
                }
                else {
                    for (int n = 0; n < numInstances; n++) {
                        if (inBag[n])
                            continue;

                        int p = 0;
                        for (int r = 0; r < representations.length; r++) {
                            for (int j = 0; j < numIntervals[r]; j++) {
                                double[] series = dimensions[n][r][intervalDimension[r][j]];
                                double[] intervalArray = Arrays.copyOfRange(series, interval[r][j][0],
                                        interval[r][j][1] + 1);

                                for (int a = 0; a < numAttributes; a++) {
                                    if (subsampleAtt[a] < 22) {
                                        testHolder.instance(0).setValue(p,
                                                c22.getSummaryStatByIndex(subsampleAtt[a], j, intervalArray));
                                    } else {
                                        testHolder.instance(0).setValue(p,
                                                FeatureSet.calcFeatureByIndex(subsampleAtt[a],
                                                        interval[r][j][0], interval[r][j][1], series));
                                    }

                                    p++;
                                }
                            }
                        }

                        double[] newProbs = tree.distributionForInstance(testHolder.instance(0));
                        oobCounts[n]++;
                        for (int k = 0; k < newProbs.length; k++)
                            trainDistributions[n][k] += newProbs[k];
                    }
                }

                trainResults.setErrorEstimateTime(trainResults.getErrorEstimateTime() + (System.nanoTime() - t1));
            }

            trees.add(tree);
            intervals.add(interval);
            subsampleAtts.add(subsampleAtt);
            intervalDimensions.add(intervalDimension);

            //Timed checkpointing if enabled, else checkpoint every 100 trees
            if(checkpoint && ((checkpointTime>0 && System.nanoTime()-lastCheckpointTime>checkpointTime)
                    || trees.size()%100 == 0)) {
                saveToFile(checkpointPath);
            }
        }
    }

    /**
     * Build the DrCIF classifier using multiple threads.
     * Unable to checkpoint until after the build process while using multiple threads.
     * For each base classifier
     *     generate random intervals
     *     do the transfrorms
     *     build the classifier
     *
     * @param representations TimeSeriesInstances data
     * @param result Instances object formatted for transformed data
     * @throws Exception unable to build DrCIF
     */
    private void multiThreadBuildDrCIF(TimeSeriesInstances[] representations, Instances result) throws Exception {
        double[][][][] dimensions =  new double[representations.length][][][];
        for (int r = 0; r < representations.length; r++){
            dimensions[r] = representations[r].toValueArray();
        }

        int[] classVals = representations[0].getClassIndexes();
        int buildStep = trainTimeContract ? numThreads : numClassifiers;

        while (withinTrainContract(trainResults.getBuildTime()) && trees.size() < numClassifiers) {
            ArrayList<Future<MultiThreadBuildHolder>> futures = new ArrayList<>(buildStep);

            int end = trees.size()+buildStep;
            for (int i = trees.size(); i < end; ++i) {
                Instances resultCopy = new Instances(result, numInstances);
                for(int n = 0; n < numInstances; n++){
                    DenseInstance in = new DenseInstance(result.numAttributes());
                    in.setValue(result.numAttributes()-1, result.instance(n).classValue());
                    resultCopy.add(in);
                }

                futures.add(ex.submit(new TreeBuildThread(i, dimensions, classVals, resultCopy)));
            }

            for (Future<MultiThreadBuildHolder> f : futures) {
                MultiThreadBuildHolder h = f.get();
                trees.add(h.tree);
                intervals.add(h.interval);
                subsampleAtts.add(h.subsampleAtts);
                intervalDimensions.add(h.intervalDimensions);

                if (bagging && getEstimateOwnPerformance()){
                    trainResults.setErrorEstimateTime(trainResults.getErrorEstimateTime() + h.errorTime);
                    for (int n = 0; n < numInstances; n++) {
                        oobCounts[n] += h.oobCounts[n];
                        for (int k = 0; k < numClasses; k++)
                            trainDistributions[n][k] += h.trainDistribution[n][k];
                    }
                }
            }
        }
    }

    /**
     * Estimate accuracy stage: Three scenarios
     * 1. If we bagged the full build (bagging ==true), we estimate using the full build OOB.
     *    If we built on all data (bagging ==false) we estimate either:
     * 2. With a 10 fold CV.
     * 3. Build a bagged model simply to get the estimate.
     *
     * @param data TimeSeriesInstances to estimate with
     * @throws Exception unable to obtain estimate
     */
    private void estimateOwnPerformance(TimeSeriesInstances data) throws Exception {
        if(bagging){
            // Use bag data, counts normalised to probabilities
            double[] preds=new double[data.numInstances()];
            double[] actuals=new double[data.numInstances()];
            long[] predTimes=new long[data.numInstances()];//Dummy variable, need something
            for(int j=0;j<data.numInstances();j++){
                long predTime = System.nanoTime();
                for(int k=0;k<trainDistributions[j].length;k++)
                    trainDistributions[j][k] /= oobCounts[j];
                preds[j] = findIndexOfMax(trainDistributions[j], rand);
                actuals[j] = data.get(j).getLabelIndex();
                predTimes[j] = System.nanoTime()-predTime;
            }
            trainResults.addAllPredictions(actuals,preds, trainDistributions, predTimes, null);
            trainResults.setClassifierName("DrCIFBagging");
            trainResults.setDatasetName(data.getProblemName());
            trainResults.setSplit("train");
            trainResults.setFoldID(seed);
            trainResults.setErrorEstimateMethod("OOB");
            trainResults.finaliseResults(actuals);
        }
        //Either do a CV, or bag and get the estimates
        else if(estimator== EstimatorMethod.CV){
            /** Defaults to 10 or numInstances, whichever is smaller.
             * Interface TrainAccuracyEstimate
             * Could this be handled better? */
            int numFolds=Math.min(data.numInstances(), 10);
            CrossValidationEvaluator cv = new CrossValidationEvaluator();
            if (seedClassifier)
                cv.setSeed(seed*5);
            cv.setNumFolds(numFolds);
            DrCIF cif=new DrCIF();
            cif.copyParameters(this);
            if (seedClassifier)
                cif.setSeed(seed*100);
            cif.setEstimateOwnPerformance(false);
            long tt = trainResults.getBuildTime();
            trainResults=cv.evaluate(cif,Converter.toArff(data));
            trainResults.setBuildTime(tt);
            trainResults.setClassifierName("DrCIFCV");
            trainResults.setErrorEstimateMethod("CV_"+numFolds);
        }
        else if(estimator== EstimatorMethod.OOB || estimator==EstimatorMethod.NONE){
            /** Build a single new DrCIF using Bagging, and extract the estimate from this
             */
            DrCIF cif=new DrCIF();
            cif.copyParameters(this);
            cif.setSeed(seed*5);
            cif.setEstimateOwnPerformance(true);
            cif.bagging=true;
            cif.buildClassifier(data);
            long tt = trainResults.getBuildTime();
            trainResults=cif.trainResults;
            trainResults.setBuildTime(tt);
            trainResults.setClassifierName("DrCIFOOB");
            trainResults.setErrorEstimateMethod("OOB");
        }
    }

    /**
     * Copy the parameters of a DrCIF object to this.
     *
     * @param other A DrCIF object
     */
    private void copyParameters(DrCIF other){
        this.numClassifiers = other.numClassifiers;
        this.attSubsampleSize = other.attSubsampleSize;
        this.outlierNorm = other.outlierNorm;
        this.useSummaryStats = other.useSummaryStats;
        this.numIntervals = other.numIntervals;
        this.numIntervalsFinder = other.numIntervalsFinder;
        this.minIntervalLength = other.minIntervalLength;
        this.minIntervalLengthFinder = other.minIntervalLengthFinder;
        this.maxIntervalLength = other.maxIntervalLength ;
        this.maxIntervalLengthFinder = other.maxIntervalLengthFinder;
        this.base = other.base;
        this.bagging = other.bagging;
        this.trainTimeContract = other.trainTimeContract;
        this.contractTime = other.contractTime;
    }

    /**
     * Find class probabilities of an instance using the trained model.
     *
     * @param ins TimeSeriesInstance object
     * @return array of doubles: probability of each class
     * @throws Exception failure to classify
     */
    @Override //TSClassifier
    public double[] distributionForInstance(TimeSeriesInstance ins) throws Exception {
        double[] d = new double[numClasses];

        double[][][] dimensions = new double[3][][];
        dimensions[0] = ins.toValueArray();
        dimensions[1] = fft.transform(ins).toValueArray();
        dimensions[2] = di.transform(ins).toValueArray();

        if (multiThread){
            ArrayList<Future<MultiThreadPredictionHolder>> futures = new ArrayList<>(trees.size());

            for (int i = 0; i < trees.size(); ++i) {
                Instances testCopy = new Instances(testHolder, 1);
                DenseInstance in = new DenseInstance(testHolder.numAttributes());
                in.setValue(testHolder.numAttributes()-1, -1);
                testCopy.add(in);

                futures.add(ex.submit(new TreePredictionThread(i, dimensions, trees.get(i), testCopy)));
            }

            for (Future<MultiThreadPredictionHolder> f : futures) {
                MultiThreadPredictionHolder h = f.get();
                d[h.c]++;

            }
        }
        else if (base instanceof ContinuousIntervalTree) {
            for (int i = 0; i < trees.size(); i++) {
                int c = (int) ((ContinuousIntervalTree) trees.get(i)).classifyInstance(dimensions, functions,
                            intervals.get(i), subsampleAtts.get(i), intervalDimensions.get(i));
                d[c]++;
            }
        }
        else {
            //Build transformed instance
            for (int i = 0; i < trees.size(); i++) {
                Catch22 c22 = new Catch22();
                c22.setOutlierNormalise(outlierNorm);

                int p = 0;
                for (int r = 0; r < dimensions.length; r++) {
                    for (int j = 0; j < intervals.get(i)[r].length; j++) {
                        double[] series = dimensions[r][intervalDimensions.get(i)[r][j]];
                        double[] intervalArray = Arrays.copyOfRange(series, intervals.get(i)[r][j][0],
                                intervals.get(i)[r][j][1] + 1);

                        for (int a = 0; a < numAttributes; a++) {
                            if (subsampleAtts.get(i)[a] < 22){
                                testHolder.instance(0).setValue(p,
                                        c22.getSummaryStatByIndex(subsampleAtts.get(i)[a], j, intervalArray));
                            }
                            else {
                                testHolder.instance(0).setValue(p,
                                        FeatureSet.calcFeatureByIndex(subsampleAtts.get(i)[a],
                                                intervals.get(i)[r][j][0], intervals.get(i)[r][j][1], series));
                            }

                            p++;
                        }
                    }
                }

                int c = (int) trees.get(i).classifyInstance(testHolder.instance(0));
                d[c]++;
            }
        }

        double sum = 0;
        for(double x: d)
            sum += x ;
        for(int i = 0; i < d.length; i++)
            d[i] = d[i]/sum;

        return d;
    }

    /**
     * Find class probabilities of an instance using the trained model.
     *
     * @param ins weka Instance object
     * @return array of doubles: probability of each class
     * @throws Exception failure to classify
     */
    @Override //AbstractClassifier
    public double[] distributionForInstance(Instance ins) throws Exception {
        return distributionForInstance(Converter.fromArff(ins));
    }

    /**
     * Classify an instance using the trained model.
     *
     * @param ins TimeSeriesInstance object
     * @return predicted class value
     * @throws Exception failure to classify
     */
    @Override //TSClassifier
    public double classifyInstance(TimeSeriesInstance ins) throws Exception {
        double[] probs = distributionForInstance(ins);
        return findIndexOfMax(probs, rand);
    }

    /**
     * Classify an instance using the trained model.
     *
     * @param ins weka Instance object
     * @return predicted class value
     * @throws Exception failure to classify
     */
    @Override //AbstractClassifier
    public double classifyInstance(Instance ins) throws Exception {
        return classifyInstance(Converter.fromArff(ins));
    }

    /**
     * Set the train time limit for a contracted classifier.
     *
     * @param amount contract time in nanoseconds
     */
    @Override //TrainTimeContractable
    public void setTrainTimeLimit(long amount) {
        contractTime = amount;
        trainTimeContract = true;
    }

    /**
     * Check if a contracted classifier is within its train time limit.
     *
     * @param start classifier build start time
     * @return true if within the contract or not contracted, false otherwise.
     */
    @Override //TrainTimeContractable
    public boolean withinTrainContract(long start){
        if(contractTime<=0) return true; //Not contracted
        return System.nanoTime()-start-checkpointTimeDiff < contractTime;
    }

    /**
     * Set the path to save checkpoint files to.
     *
     * @param path string for full path for the directory to store checkpointed files
     * @return true if valid path, false otherwise
     */
    @Override //Checkpointable
    public boolean setCheckpointPath(String path) {
        boolean validPath = Checkpointable.super.createDirectories(path);
        if(validPath){
            checkpointPath = path;
            checkpoint = true;
        }
        return validPath;
    }

    /**
     * Set the time between checkpoints in hours.
     *
     * @param t number of hours between checkpoints
     * @return true
     */
    @Override //Checkpointable
    public boolean setCheckpointTimeHours(int t){
        checkpointTime=TimeUnit.NANOSECONDS.convert(t,TimeUnit.HOURS);
        return true;
    }

    /**
     * Serialises this DrCIF object to the specified path.
     *
     * @param path save path for object
     * @throws Exception object fails to save
     */
    @Override //Checkpointable
    public void saveToFile(String path) throws Exception{
        lastCheckpointTime = System.nanoTime();
        Checkpointable.super.saveToFile(path + "DrCIF" + seed + "temp.ser");
        File file = new File(path + "DrCIF" + seed + "temp.ser");
        File file2 = new File(path + "DrCIF" + seed + ".ser");
        file2.delete();
        file.renameTo(file2);
        if (internalContractCheckpointHandling) checkpointTimeDiff += System.nanoTime()-lastCheckpointTime;
    }

    /**
     * Copies values from a loaded DrCIF object into this object.
     *
     * @param obj a DrCIF object
     * @throws Exception if obj is not an instance of DrCIF
     */
    @Override //Checkpointable
    public void copyFromSerObject(Object obj) throws Exception {
        if (!(obj instanceof DrCIF))
            throw new Exception("The SER file is not an instance of TSF");
        DrCIF saved = ((DrCIF)obj);
        System.out.println("Loading DrCIF" + seed + ".ser");

        try {
            numClassifiers = saved.numClassifiers;
            attSubsampleSize = saved.attSubsampleSize;
            numAttributes = saved.numAttributes;
            startNumAttributes = saved.startNumAttributes;
            subsampleAtts = saved.subsampleAtts;
            outlierNorm = saved.outlierNorm;
            useSummaryStats = saved.useSummaryStats;
            numIntervals = saved.numIntervals;
            //numIntervalsFinder = saved.numIntervalsFinder;
            minIntervalLength = saved.minIntervalLength;
            //minIntervalLengthFinder = saved.minIntervalLengthFinder;
            maxIntervalLength = saved.maxIntervalLength;
            //maxIntervalLengthFinder = saved.maxIntervalLengthFinder;
            trees = saved.trees;
            base = saved.base;
            intervals = saved.intervals;
            //testHolder = saved.testHolder;
            bagging = saved.bagging;
            oobCounts = saved.oobCounts;
            trainDistributions = saved.trainDistributions;
            //checkpoint = saved.checkpoint;
            //checkpointPath = saved.checkpointPath
            //checkpointTime = saved.checkpointTime;
            //lastCheckpointTime = saved.lastCheckpointTime;
            //checkpointTimeDiff = saved.checkpointTimeDiff;
            //internalContractCheckpointHandling = saved.internalContractCheckpointHandling;
            trainTimeContract = saved.trainTimeContract;
            if (internalContractCheckpointHandling) contractTime = saved.contractTime;
            maxClassifiers = saved.maxClassifiers;
            //numThreads = saved.numThreads;
            //multiThread = saved.multiThread;
            //ex = saved.ex;
            numInstances = saved.numInstances;
            numDimensions = saved.numDimensions;
            intervalDimensions = saved.intervalDimensions;
            //c22 = saved.c22;

            trainResults = saved.trainResults;
            if (!internalContractCheckpointHandling) trainResults.setBuildTime(System.nanoTime());
            seedClassifier = saved.seedClassifier;
            seed = saved.seed;
            rand = saved.rand;
            estimateOwnPerformance = saved.estimateOwnPerformance;
            estimator = saved.estimator;
            numClasses = saved.numClasses;

            if (internalContractCheckpointHandling) checkpointTimeDiff = saved.checkpointTimeDiff
                    + (System.nanoTime() - saved.lastCheckpointTime);
            lastCheckpointTime = System.nanoTime();
        }catch(Exception ex){
            System.out.println("Unable to assign variables when loading serialised file");
        }
    }

    /**
     * Returns the default set of possible parameter values for use in setOptions when tuning.
     *
     * @return default parameter space for tuning
     */
    @Override //Tunable
    public ParameterSpace getDefaultParameterSearchSpace(){
        ParameterSpace ps=new ParameterSpace();
        String[] numAtts={"8","16","25"};
        ps.addParameter("-A", numAtts);
        String[] maxIntervalLengths={"0.5","0.75","1"};
        ps.addParameter("-L", maxIntervalLengths);
        return ps;
    }

    /**
     * Parses a given list of options. Valid options are:
     *
     * -A  The number of attributes to subsample as an integer from 1-25.
     * -L  Max interval length as a proportion of series length as a double from 0-1.
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option value is invalid
     */
    @Override //AbstractClassifier
    public void setOptions(String[] options) throws Exception{
        System.out.println(Arrays.toString(options));

        String numAttsString = Utils.getOption("-A", options);
        System.out.println(numAttsString);
        if (numAttsString.length() != 0)
            attSubsampleSize = Integer.parseInt(numAttsString);

        String maxIntervalLengthsString = Utils.getOption("-L", options);
        System.out.println(maxIntervalLengthsString);
        if (maxIntervalLengthsString.length() != 0)
            maxIntervalLengthFinder = (numAtts) -> (int)(numAtts*Double.parseDouble(maxIntervalLengthsString));

        System.out.println(attSubsampleSize + " " + maxIntervalLengthFinder.apply(100));
    }

    /**
     * Enables multi threading with a set number of threads to use.
     *
     * @param numThreads number of threads available for multi threading
     */
    @Override //MultiThreadable
    public void enableMultiThreading(int numThreads) {
        if (numThreads > 1) {
            this.numThreads = numThreads;
            multiThread = true;
        } else {
            this.numThreads = 1;
            multiThread = false;
        }
    }

    /**
     * Nested class to find and store seven simple summary features for an interval
     */
    private static class FeatureSet {
        public static double calcFeatureByIndex(int idx, int start, int end, double[] data) {
            switch (idx){
                case 22: return calcMean(start, end, data);
                case 23: return calcMedian(start, end, data);
                case 24: return calcStandardDeviation(start, end, data);
                case 25: return calcSlope(start, end, data);
                case 26: return calcInterquartileRange(start, end, data);
                case 27: return calcMin(start, end, data);
                case 28: return calcMax(start, end, data);
                default: return Double.NaN;
            }
        }

        public static double calcMean(int start, int end, double[] data){
            double sumY = 0;
            for(int i=start;i<=end;i++) {
                sumY += data[i];
            }

            int length = end-start+1;
            return sumY/length;
        }

        public static double calcMedian(int start, int end, double[] data){
            ArrayList<Double> sortedData = new ArrayList<>(end-start+1);
            for(int i=start;i<=end;i++){
                sortedData.add(data[i]);
            }

            return median(sortedData, false); //sorted in function
        }

        public static double calcStandardDeviation(int start, int end, double[] data){
            double sumY = 0;
            double sumYY = 0;
            for(int i=start;i<=end;i++) {
                sumY += data[i];
                sumYY += data[i] * data[i];
            }

            int length = end-start+1;
            return (sumYY-(sumY*sumY)/length)/(length-1);
        }

        public static double calcSlope(int start, int end, double[] data){
            double sumY = 0;
            double sumX = 0, sumXX = 0, sumXY = 0;
            for(int i=start;i<=end;i++) {
                sumY += data[i];
                sumX+=(i-start);
                sumXX+=(i-start)*(i-start);
                sumXY+=data[i]*(i-start);
            }

            int length = end-start+1;
            double slope=(sumXY-(sumX*sumY)/length);
            double denom=sumXX-(sumX*sumX)/length;
            slope = denom == 0 ? 0 : slope/denom;
            return slope;
        }

        public static double calcInterquartileRange(int start, int end, double[] data){
            ArrayList<Double> sortedData = new ArrayList<>(end-start+1);
            for(int i=start;i<=end;i++){
                sortedData.add(data[i]);
            }
            Collections.sort(sortedData);

            int length = end-start+1;
            ArrayList<Double> left = new ArrayList<>(length / 2 + 1);
            ArrayList<Double> right = new ArrayList<>(length / 2 + 1);
            if (length % 2 == 1) {
                for (int i = 0; i <= length / 2; i++){
                    left.add(sortedData.get(i));
                }
            }
            else {
                for (int i = 0; i < length / 2; i++){
                    left.add(sortedData.get(i));
                }

            }
            for (int i = length / 2; i < sortedData.size(); i++){
                right.add(sortedData.get(i));
            }

            return median(right, false) - median(left, false);
        }

        public static double calcMin(int start, int end, double[] data){
            double min = Double.MAX_VALUE;
            for(int i=start;i<=end;i++){
                if (data[i] < min) min = data[i];
            }
            return min;
        }

        public static double calcMax(int start, int end, double[] data){
            double max = -999999999;
            for(int i=start;i<=end;i++){
                if (data[i] > max) max = data[i];
            }
            return max;
        }
    }

    /**
     * Class to hold data about a DrCIF tree when multi threading.
     */
    private static class MultiThreadBuildHolder {
        int[] subsampleAtts;
        int[][] intervalDimensions;
        Classifier tree;
        int[][][] interval;

        double[][] trainDistribution;
        int[] oobCounts;
        long errorTime;

        public MultiThreadBuildHolder() { }
    }

    /**
     * Class to build a DrCIF tree when multi threading.
     */
    private class TreeBuildThread implements Callable<MultiThreadBuildHolder> {
        int i;
        double[][][][] dimensions;
        int[] classVals;
        Instances result;

        public TreeBuildThread(int i, double[][][][] dimensions, int[] classVals, Instances result){
            this.i = i;
            this.dimensions = dimensions;
            this.classVals = classVals;
            this.result = result;
        }

        /**
         *   generate random intervals
         *   do the transfrorms
         *   build the classifier
         **/
        @Override
        public MultiThreadBuildHolder call() throws Exception{
            MultiThreadBuildHolder h = new MultiThreadBuildHolder();
            Random rand = new Random(seed + i * numClassifiers);

            Catch22 c22 = new Catch22();
            c22.setOutlierNormalise(outlierNorm);

            //1. Select random intervals for tree i
            int[][][] interval = new int[dimensions.length][][];
            for (int r = 0; r < dimensions.length; r++) {
                interval[r] = new int[numIntervals[r]][2];

                for (int j = 0; j < numIntervals[r]; j++) {
                    if (rand.nextBoolean()) {
                        interval[r][j][0] = rand.nextInt(dimensions[r][0][0].length -
                                minIntervalLength[r]); //Start point

                        int range = Math.min(dimensions[r][0][0].length - interval[r][j][0],
                                maxIntervalLength[r]);
                        int length;
                        if (range - minIntervalLength[r] == 0) length = minIntervalLength[r];
                        else length = rand.nextInt(range - minIntervalLength[r]) + minIntervalLength[r];
                        interval[r][j][1] = interval[r][j][0] + length;
                    } else {
                        interval[r][j][1] = rand.nextInt(dimensions[r][0][0].length -
                                minIntervalLength[r]) + minIntervalLength[r]; //Start point

                        int range = Math.min(interval[r][j][1], maxIntervalLength[r]);
                        int length;
                        if (range - minIntervalLength[r] == 0) length = minIntervalLength[r];
                        else length = rand.nextInt(range - minIntervalLength[r]) + minIntervalLength[r];
                        interval[r][j][0] = interval[r][j][1] - length;
                    }
                }
            }

            //If bagging find instances with replacement
            int[] instInclusions = null;
            boolean[] inBag = null;
            if (bagging) {
                inBag = new boolean[numInstances];
                instInclusions = new int[numInstances];

                for (int n = 0; n < numInstances; n++) {
                    instInclusions[rand.nextInt(numInstances)]++;
                }

                for (int n = 0; n < numInstances; n++) {
                    if (instInclusions[n] > 0) {
                        inBag[n] = true;
                    }
                }
            }

            //find attributes to subsample
            ArrayList<Integer> arrl = new ArrayList<>(startNumAttributes);
            for (int n = 0; n < startNumAttributes; n++){
                arrl.add(n);
            }

            int[] subsampleAtts = new int[numAttributes];
            for (int n = 0; n < numAttributes; n++){
                subsampleAtts[n] = arrl.remove(rand.nextInt(arrl.size()));
            }

            //find dimensions for each interval
            int[][] intervalDimensions = new int[dimensions.length][];
            for (int r = 0; r < dimensions.length; r++) {
                intervalDimensions[r] = new int[numIntervals[r]];
                for (int n = 0; n < numIntervals[r]; n++) {
                    intervalDimensions[r][n] = rand.nextInt(numDimensions);
                }
                Arrays.sort(intervalDimensions[r]);
            }

            h.subsampleAtts = subsampleAtts;
            h.intervalDimensions = intervalDimensions;

            //For bagging
            int instIdx = 0;
            int lastIdx = -1;

            //2. Generate and store attributes
            for (int k = 0; k < numInstances; k++) {
                //For each instance
                if (bagging) {
                    boolean sameInst = false;

                    while (true) {
                        if (instInclusions[instIdx] == 0) {
                            instIdx++;
                        } else {
                            instInclusions[instIdx]--;

                            if (instIdx == lastIdx) {
                                result.set(k, new DenseInstance(result.instance(k - 1)));
                                sameInst = true;
                            } else {
                                lastIdx = instIdx;
                            }

                            break;
                        }
                    }

                    if (sameInst) continue;

                    result.instance(k).setValue(result.classIndex(), classVals[instIdx]);
                } else {
                    instIdx = k;
                }

                int p = 0;
                for (int r = 0; r < dimensions.length; r++) {
                    for (int j = 0; j < numIntervals[r]; j++) {
                        //extract the interval
                        double[] series = dimensions[r][instIdx][intervalDimensions[r][j]];
                        double[] intervalArray = Arrays.copyOfRange(series, interval[r][j][0], interval[r][j][1] + 1);

                        //process features
                        for (int a = 0; a < numAttributes; a++) {
                            if (subsampleAtts[a] < 22) {
                                result.instance(k).setValue(p,
                                        c22.getSummaryStatByIndex(subsampleAtts[a], j, intervalArray));
                            }
                            else {
                                result.instance(k).setValue(p,
                                        FeatureSet.calcFeatureByIndex(subsampleAtts[a], interval[r][j][0],
                                                interval[r][j][1], series));
                            }

                            p++;
                        }
                    }
                }
            }

            //3. Create and build tree using all the features. Feature selection
            Classifier tree = AbstractClassifier.makeCopy(base);
            if (seedClassifier && tree instanceof Randomizable)
                ((Randomizable) tree).setSeed(seed * (i + 1));

            tree.buildClassifier(result);

            if (bagging && getEstimateOwnPerformance()) {
                long t1 = System.nanoTime();
                int[] oobCounts = new int[numInstances];
                double[][] trainDistributions = new double[numInstances][numClasses];

                if (base instanceof ContinuousIntervalTree){
                    for (int n = 0; n < numInstances; n++) {
                        if (inBag[n])
                            continue;

                        double[] newProbs = ((ContinuousIntervalTree) tree).distributionForInstance(dimensions[n],
                                functions, interval, subsampleAtts, intervalDimensions);
                        oobCounts[n]++;
                        for (int k = 0; k < newProbs.length; k++)
                            trainDistributions[n][k] += newProbs[k];
                    }
                }
                else {
                    for (int n = 0; n < numInstances; n++) {
                        if (inBag[n])
                            continue;

                        int p = 0;
                        for (int r = 0; r < dimensions.length; r++) {
                            for (int j = 0; j < numIntervals[r]; j++) {
                                double[] series = dimensions[r][n][intervalDimensions[r][j]];
                                double[] intervalArray = Arrays.copyOfRange(series, interval[r][j][0],
                                        interval[r][j][1] + 1);

                                for (int a = 0; a < numAttributes; a++) {
                                    if (subsampleAtts[a] < 22) {
                                        result.instance(0).setValue(p,
                                                c22.getSummaryStatByIndex(subsampleAtts[a], j, intervalArray));
                                    } else {
                                        result.instance(0).setValue(p,
                                                FeatureSet.calcFeatureByIndex(subsampleAtts[a],
                                                        interval[r][j][0], interval[r][j][1], series));
                                    }

                                    p++;
                                }
                            }
                        }

                        double[] newProbs = tree.distributionForInstance(testHolder.instance(0));
                        oobCounts[n]++;
                        for (int k = 0; k < newProbs.length; k++)
                            trainDistributions[n][k] += newProbs[k];
                    }
                }

                h.oobCounts = oobCounts;
                h.trainDistribution = trainDistributions;
                h.errorTime = System.nanoTime() - t1;
            }

            h.tree = tree;
            h.interval = interval;
            return h;
        }
    }

    /**
     * Class to hold data about a DrCIF tree when multi threading.
     */
    private static class MultiThreadPredictionHolder {
        int c;

        public MultiThreadPredictionHolder() { }
    }

    /**
     * Class to make a class prediction using a DrCIF tree when multi threading.
     */
    private class TreePredictionThread implements Callable<MultiThreadPredictionHolder> {
        int i;
        double[][][] dimensions;
        Classifier tree;
        Instances testHolder;

        public TreePredictionThread(int i, double[][][] dimensions, Classifier tree, Instances testHolder){
            this.i = i;
            this.dimensions = dimensions;
            this.tree = tree;
            this.testHolder = testHolder;
        }

        @Override
        public MultiThreadPredictionHolder call() throws Exception{
            MultiThreadPredictionHolder h = new MultiThreadPredictionHolder();

            if (base instanceof ContinuousIntervalTree) {
                h.c = (int) ((ContinuousIntervalTree) trees.get(i)).classifyInstance(dimensions, functions,
                            intervals.get(i), subsampleAtts.get(i), intervalDimensions.get(i));
            }
            else {
                Catch22 c22 = new Catch22();
                c22.setOutlierNormalise(outlierNorm);

                int p = 0;
                for (int r = 0; r < dimensions.length; r++) {
                    for (int j = 0; j < intervals.get(i)[r].length; j++) {
                        double[] series = dimensions[r][intervalDimensions.get(i)[r][j]];
                        double[] intervalArray = Arrays.copyOfRange(series, intervals.get(i)[r][j][0],
                                intervals.get(i)[r][j][1] + 1);

                        for (int a = 0; a < numAttributes; a++) {
                            if (subsampleAtts.get(i)[a] < 22) {
                                testHolder.instance(0).setValue(p,
                                        c22.getSummaryStatByIndex(subsampleAtts.get(i)[a], j, intervalArray));
                            } else {
                                testHolder.instance(0).setValue(p,
                                        FeatureSet.calcFeatureByIndex(subsampleAtts.get(i)[a],
                                                intervals.get(i)[r][j][0], intervals.get(i)[r][j][1], series));
                            }

                            p++;
                        }
                    }
                }

                h.c = (int) tree.classifyInstance(testHolder.instance(0));
            }

            return h;
        }
    }

    /** DrCIF attributes as functions **/
    public Function<Interval, Double>[] functions = new Function[]{c22_0, c22_1, c22_2, c22_3, c22_4, c22_5, c22_6,
            c22_7, c22_8, c22_9, c22_10, c22_11, c22_12, c22_13, c22_14, c22_15, c22_16, c22_17, c22_18, c22_19, c22_20,
            c22_21, mean, median, stdev, slope, iqr, min, max};

    public static final Function<Interval, Double> c22_0 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(0, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_1 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(1, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_2 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(2, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_3 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(3, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_4 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(4, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_5 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(5, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_6 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(6, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_7 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(7, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_8 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(8, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_9 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(9, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_10 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(10, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_11 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(11, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_12 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(12, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_13 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(13, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_14 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(14, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_15 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(15, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_16 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(16, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_17 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(17, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_18 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(18, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_19 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(19, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_20 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(20, intervalArray, true);
    };
    public static final Function<Interval, Double> c22_21 = (Interval i) -> {
        double[] intervalArray = Arrays.copyOfRange(i.series, i.start, i.end + 1);
        return Catch22.getSummaryStatByIndex(21, intervalArray, true);
    };
    public static final Function<Interval, Double> mean = (Interval i) ->
            FeatureSet.calcFeatureByIndex(22, i.start, i.end, i.series);
    public static final Function<Interval, Double> median = (Interval i) ->
            FeatureSet.calcFeatureByIndex(23, i.start, i.end, i.series);
    public static final Function<Interval, Double> stdev = (Interval i) ->
            FeatureSet.calcFeatureByIndex(24, i.start, i.end, i.series);
    public static final Function<Interval, Double> slope = (Interval i) ->
            FeatureSet.calcFeatureByIndex(25, i.start, i.end, i.series);
    public static final Function<Interval, Double> iqr = (Interval i) ->
            FeatureSet.calcFeatureByIndex(26, i.start, i.end, i.series);
    public static final Function<Interval, Double> min = (Interval i) ->
            FeatureSet.calcFeatureByIndex(27, i.start, i.end, i.series);
    public static final Function<Interval, Double> max = (Interval i) ->
            FeatureSet.calcFeatureByIndex(28, i.start, i.end, i.series);


    /**
     * Development tests for the DrCIF classifier.
     *
     * @param arg arguments, unused
     * @throws Exception if tests fail
     */
    public static void main(String[] arg) throws Exception{
        String dataLocation="D:\\CMP Machine Learning\\Datasets\\UnivariateARFF\\";
        String problem="ItalyPowerDemand";
        Instances train= DatasetLoading.loadDataNullable(dataLocation+problem+"\\"+problem+"_TRAIN");
        Instances test= DatasetLoading.loadDataNullable(dataLocation+problem+"\\"+problem+"_TEST");
        DrCIF c = new DrCIF();
        c.setSeed(0);
        c.estimateOwnPerformance = true;
        c.estimator = EstimatorMethod.OOB;
        double a;
        long t1 = System.nanoTime();
        c.buildClassifier(train);
        System.out.println("Train time="+(System.nanoTime()-t1)*1e-9);
        System.out.println("build ok: original atts = "+(train.numAttributes()-1)+" new atts = "
                +(c.testHolder.numAttributes()-1)+" num trees = "+c.trees.size()+" num intervals = "+
                Arrays.toString(c.numIntervals));
        System.out.println("recorded times: train time = "+(c.trainResults.getBuildTime()*1e-9)+" estimate time = "
                +(c.trainResults.getErrorEstimateTime()*1e-9)
                +" both = "+(c.trainResults.getBuildPlusEstimateTime()*1e-9));
        a= ClassifierTools.accuracy(test, c);
        System.out.println("Test Accuracy = "+a);
        System.out.println("Train Accuracy = "+c.trainResults.getAcc());

        //Test Accuracy = 0.9727891156462585
        //Train Accuracy = 0.9701492537313433
    }
}
