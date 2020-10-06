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
package tsml.classifiers.dictionary_based;

import java.util.*;
import tsml.classifiers.*;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import tsml.classifiers.MemoryContractable;
import utilities.*;
import utilities.samplers.*;
import weka.classifiers.functions.GaussianProcesses;
import weka.core.*;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;

import static utilities.InstanceTools.resampleTrainAndTestInstances;
import static utilities.multivariate_tools.MultivariateInstanceTools.*;
import static weka.core.Utils.sum;

/**
 * cBOSS classifier with parameter search and ensembling for univariate and
 * multivariate time series classification.
 * If parameters are known, use the class BOSSIndividual and directly provide them.
 *
 * Options to change the method of ensembling to randomly select parameters with or without a filter.
 * Has the capability to contract train time and checkpoint when using a random ensemble.
 *
 * Alphabetsize fixed to four and maximum wordLength of 16.
 *
 * @author Matthew Middlehurst
 *
 * Implementation based on the algorithm described in getTechnicalInformation()


 */
public class cBOSS extends EnhancedAbstractClassifier implements TrainTimeContractable,
        MemoryContractable, Checkpointable, TechnicalInformationHandler, MultiThreadable {

    private ArrayList<Double>[] paramAccuracy;
    private ArrayList<Double>[] paramTime;
    private ArrayList<Double>[] paramMemory;

    private int ensembleSize = 50;
    private int ensembleSizePerChannel = -1;
    private boolean randomCVAccEnsemble = false;
    private boolean useWeights = false;

    private boolean useFastTrainEstimate = false;
    private int maxEvalPerClass = -1;
    private int maxEval = 500;

    private double maxWinLenProportion = 1;
    private double maxWinSearchProportion = 0.25;

    private boolean reduceTrainInstances = false;
    private double trainProportion = -1;
    private int maxTrainInstances = 1000;
    private boolean stratifiedSubsample = false;

    private boolean cutoff = false;

    private transient LinkedList<IndividualBOSS>[] classifiers;
    private int numSeries;
    private int[] numClassifiers;
    private int currentSeries = 0;
    private boolean isMultivariate = false;

    private final int[] wordLengths = { 16, 14, 12, 10, 8 };
    private final int[] alphabetSize = { 4 };
    private final boolean[] normOptions = { true, false };
    private final double correctThreshold = 0.92;
    private int maxEnsembleSize = 500;

    private boolean bayesianParameterSelection = false;
    private int initialRandomParameters = 20;
    private int[] initialParameterCount;
    private Instances[] parameterPool;
    private Instances[] prevParameters;

    private String checkpointPath;
    private boolean checkpoint = false;
    private long checkpointTime = 0;
    private long checkpointTimeDiff = 0;
    private boolean cleanupCheckpointFiles = false;
    private boolean loadAndFinish = false;

    private long trainContractTimeNanos = 0;
    private boolean trainTimeContract = false;
    private boolean underContractTime = false;

    private long memoryLimit = 0;
    private long bytesUsed = 0;
    private boolean memoryContract = false;
    private boolean underMemoryLimit = true;

    //cBOSS CV acc variables, stored as field for checkpointing.
    private int[] classifiersBuilt;
    private int[] lowestAccIdx;
    private double[] lowestAcc;

    private boolean fullTrainCVEstimate = false;
    private double[][] trainDistributions;
    private int[] idxSubsampleCount;
    private ArrayList<Integer> latestTrainPreds;
    private ArrayList<Integer> latestTrainIdx;
    private ArrayList<ArrayList>[] filterTrainPreds;
    private ArrayList<ArrayList>[] filterTrainIdx;
    private Instances seriesHeader;

    private transient Instances train;
    private double ensembleCvAcc = -1;
    private double[] ensembleCvPreds = null;

    private int numThreads = 1;
    private boolean multiThread = false;
    private ExecutorService ex;

    protected static final long serialVersionUID = 22554L;

    public cBOSS(){
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
        useRecommendedSettings();
    }

    public cBOSS(boolean useRecommendedSettings){
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
        if (useRecommendedSettings) useRecommendedSettings();
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "M. Middlehurst, W. Vickers and A. Bagnall");
        result.setValue(TechnicalInformation.Field.TITLE, "Scalable dictionary classifiers for time series " +
                "classification");
        result.setValue(TechnicalInformation.Field.JOURNAL, "International Conference on Intelligent Data " +
                "Engineering and Automated Learning");
        result.setValue(TechnicalInformation.Field.PAGES, "11-19");
        result.setValue(TechnicalInformation.Field.YEAR, "2020");
        return result;
    }

    @Override
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

    @Override
    public String getParameters() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.getParameters());

        sb.append(",numSeries,").append(numSeries);

        for (int n = 0; n < numSeries; n++) {
            sb.append(",numclassifiers,").append(n).append(",").append(numClassifiers[n]);

            for (int i = 0; i < numClassifiers[n]; ++i) {
                IndividualBOSS boss = classifiers[n].get(i);
                sb.append(",windowSize,").append(boss.getWindowSize()).append(",wordLength,").append(boss.getWordLength());
                sb.append(",alphabetSize,").append(boss.getAlphabetSize()).append(",norm,").append(boss.isNorm());
            }
        }

        return sb.toString();
    }

    private void useRecommendedSettings(){
        ensembleSize = 250;
        maxEnsembleSize = 50;
        randomCVAccEnsemble = true;
        useWeights = true;
        reduceTrainInstances = true;
        trainProportion = 0.7;
        //bayesianParameterSelection = true;
    }

    //pass in an enum of hour, minute, day, and the amount of them.
    @Override
    public void setTrainTimeLimit(long amount){
        printLineDebug(" cBOSS setting contract to "+amount);
        if(amount>0) {
            trainContractTimeNanos = amount;
            trainTimeContract = true;
        }
        else
            trainTimeContract = false;
    }

    @Override
    public void setMemoryLimit(DataUnit unit, long amount){
        try {
            throw new Exception("Memory contract currently unavailable");
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

//        switch (unit){
//            case GIGABYTE:
//                memoryLimit = amount*1073741824;
//                break;
//            case MEGABYTE:
//                memoryLimit = amount*1048576;
//                break;
//            case BYTES:
//                memoryLimit = amount;
//                break;
//            default:
//                throw new InvalidParameterException("Invalid data unit");
//        }
//        memoryContract = true;
    }

    @Override
    public void enableMultiThreading(int numThreads) {
        if (numThreads > 1) {
            this.numThreads = numThreads;
            multiThread = true;
        }
        else{
            this.numThreads = 1;
            multiThread = false;
        }
    }

    //Set the path where checkpointed versions will be stored
    @Override //Checkpointable
    public boolean setCheckpointPath(String path) {
        boolean validPath=Checkpointable.super.createDirectories(path);
        if(validPath){
            checkpointPath = path;
            checkpoint = true;
        }
        return validPath;
    }

    //Define how to copy from a loaded object to this object
    @Override
    public void copyFromSerObject(Object obj) throws Exception{
        if(!(obj instanceof cBOSS))
            throw new Exception("The SER file is not an instance of cBOSS");
        cBOSS saved = ((cBOSS)obj);
        System.out.println("Loading cBOSS" + seed + ".ser");

        //copy over variables from serialised object
        paramAccuracy = saved.paramAccuracy;
        paramTime = saved.paramTime;
        paramMemory = saved.paramMemory;
        ensembleSize = saved.ensembleSize;
        seed = saved.seed;
        ensembleSizePerChannel = saved.ensembleSizePerChannel;
        rand = saved.rand;
        randomCVAccEnsemble = saved.randomCVAccEnsemble;
        useWeights = saved.useWeights;
        useFastTrainEstimate = saved.useFastTrainEstimate;
        maxEvalPerClass = saved.maxEvalPerClass;
        maxEval = saved.maxEval;
        maxWinLenProportion = saved.maxWinLenProportion;
        maxWinSearchProportion = saved.maxWinSearchProportion;
        reduceTrainInstances = saved.reduceTrainInstances;
        trainProportion = saved.trainProportion;
        maxTrainInstances = saved.maxTrainInstances;
        stratifiedSubsample = saved.stratifiedSubsample;
        cutoff = saved.cutoff;
//        loadAndFinish = saved.loadAndFinish;
        numSeries = saved.numSeries;
        numClassifiers = saved.numClassifiers;
        currentSeries = saved.currentSeries;
        isMultivariate = saved.isMultivariate;
//        wordLengths = saved.wordLengths;
//        alphabetSize = saved.alphabetSize;
//        correctThreshold = saved.correctThreshold;
        maxEnsembleSize = saved.maxEnsembleSize;
        bayesianParameterSelection = saved.bayesianParameterSelection;
        initialRandomParameters = saved.initialRandomParameters;
        initialParameterCount = saved.initialParameterCount;
        parameterPool = saved.parameterPool;
        prevParameters = saved.prevParameters;
//        checkpointPath = saved.checkpointPath;
//        serPath = saved.serPath;
//        checkpoint = saved.checkpoint;
        checkpointTime = saved.checkpointTime;
//        checkpointTimeDiff = checkpointTimeDiff;
        cleanupCheckpointFiles = saved.cleanupCheckpointFiles;
        trainContractTimeNanos = saved.trainContractTimeNanos;
        trainTimeContract = saved.trainTimeContract;
        underContractTime = saved.underContractTime;
        memoryLimit = saved.memoryLimit;
        bytesUsed = saved.bytesUsed;
        memoryContract = saved.memoryContract;
        underMemoryLimit = saved.underMemoryLimit;
        classifiersBuilt = saved.classifiersBuilt;
        lowestAccIdx = saved.lowestAccIdx;
        lowestAcc = saved.lowestAcc;
        fullTrainCVEstimate = saved.fullTrainCVEstimate;
        trainDistributions = saved.trainDistributions;
        idxSubsampleCount = saved.idxSubsampleCount;
        latestTrainPreds = saved.latestTrainPreds;
        latestTrainIdx = saved.latestTrainIdx;
        filterTrainPreds = saved.filterTrainPreds;
        filterTrainIdx = saved.filterTrainIdx;
        seriesHeader = saved.seriesHeader;
        trainResults = saved.trainResults;
        ensembleCvAcc = saved.ensembleCvAcc;
        ensembleCvPreds = saved.ensembleCvPreds;
        numThreads = saved.numThreads;
        multiThread = saved.multiThread;

        //load in each serisalised classifier
        classifiers = new LinkedList[numSeries];
        for (int n = 0; n < numSeries; n++) {
            classifiers[n] = new LinkedList();
            for (int i = 0; i < saved.numClassifiers[n]; i++) {
                System.out.println("Loading cBOSSIndividual" + seed + n + "-" + i + ".ser");

                FileInputStream fis = new FileInputStream(checkpointPath + "cBOSSIndividual" + seed + n + "-" + i + ".ser");
                try (ObjectInputStream in = new ObjectInputStream(fis)) {
                    Object indv = in.readObject();

                    if (!(indv instanceof IndividualBOSS))
                        throw new Exception("The SER file " + n + "-" + i + " is not an instance of cBOSSIndividual");
                    IndividualBOSS ser = ((IndividualBOSS) indv);
                    classifiers[n].add(ser);
                }
            }
        }

        checkpointTimeDiff = saved.checkpointTimeDiff + (System.nanoTime() - checkpointTime);
    }

    @Override
    public ClassifierResults getTrainResults(){
//        trainResults.setAcc(ensembleCvAcc);
        return trainResults;
    }

    public void setEnsembleSize(int size) {
        ensembleSize = size;
    }

    public void setMaxEnsembleSize(int size) {
        maxEnsembleSize = size;
    }

    public void setRandomCVAccEnsemble(boolean b){
        randomCVAccEnsemble = b;
    }

    public void useWeights(boolean b) {
        useWeights = b;
    }

    public void setFastTrainEstimate(boolean b){
        useFastTrainEstimate = b;
    }

    public void setMaxEval(int i) {
        maxEval = i;
    }

    public void setMaxEvalPerClass(int i) {
        maxEvalPerClass = i;
    }

    public void setReduceTrainInstances(boolean b){
        reduceTrainInstances = b;
    }

    public void setTrainProportion(double d){
        trainProportion = d;
    }

    public void setMaxTrainInstances(int i){
        maxTrainInstances = i;
    }

    public void setCleanupCheckpointFiles(boolean b) {
        cleanupCheckpointFiles = b;
    }

    public void setFullTrainCVEstimate(boolean b){
        fullTrainCVEstimate = b;
    }

    public void setCutoff(boolean b) {
        cutoff = b;
    }

    public void cleanupCheckpointFiles(boolean b){
        cleanupCheckpointFiles = b;
    }

    public void loadAndFinish(boolean b) {
        loadAndFinish = b;
    }

    public void setMaxWinLenProportion(double d){
        maxWinLenProportion = d;
    }

    public void setMaxWinSearchProportion(double d){
        maxWinSearchProportion = d;
    }

    public void setBayesianParameterSelection(boolean b) {
        bayesianParameterSelection = b;
    }

    @Override
    public void buildClassifier(final Instances data) throws Exception {
        // can classifier handle the data?
        getCapabilities().testWithFail(data);
        trainResults.setBuildTime(System.nanoTime());
        long startTime=System.nanoTime();

        if (data.checkForAttributeType(Attribute.RELATIONAL)) {
            isMultivariate = true;
        }

        //Window length settings
        int seriesLength = isMultivariate ? channelLength(data) - 1 : data.numAttributes() - 1; //minus class attribute
        int minWindow = 10;
        int maxWindow = (int) (seriesLength * maxWinLenProportion);
        if (maxWindow < minWindow) minWindow = maxWindow / 2;
        //whats the max number of window sizes that should be searched through
        double maxWindowSearches = seriesLength * maxWinSearchProportion;
        int winInc = (int) ((maxWindow - minWindow) / maxWindowSearches);
        if (winInc < 1) winInc = 1;

        //path checkpoint files will be saved to
//        checkpointPath = checkpointPath + "/" + checkpointName(data.relationName()) + "/";
        File file = new File(checkpointPath + "cBOSS" + seed + ".ser");
        //if checkpointing and serialised files exist load said files
        if (checkpoint && file.exists()){
            //path checkpoint files will be saved to
            printLineDebug("Loading from checkpoint file");
            loadFromFile(checkpointPath + "cBOSS" + seed + ".ser");
            //               checkpointTimeElapsed -= System.nanoTime()-t1;
        }
        //initialise variables
        else {
            if (data.classIndex() != data.numAttributes() - 1)
                throw new Exception("BOSS_BuildClassifier: Class attribute not set as last attribute in dataset");

            printLineDebug("Building cBOSS  target number of classifiers = " +ensembleSize);

            //Multivariate
            if (isMultivariate) {
                numSeries = numDimensions(data);
                classifiers = new LinkedList[numSeries];

                for (int n = 0; n < numSeries; n++) {
                    classifiers[n] = new LinkedList<>();
                }

                numClassifiers = new int[numSeries];

                if (ensembleSizePerChannel > 0) {
                    ensembleSize = ensembleSizePerChannel * numSeries;
                }
            }
            //Univariate
            else {
                numSeries = 1;
                classifiers = new LinkedList[1];
                classifiers[0] = new LinkedList<>();
                numClassifiers = new int[1];
            }

            if (maxEvalPerClass > 0) {
                maxEval = data.numClasses() * maxEvalPerClass;
            }

            rand = new Random(seed);

            parameterPool = uniqueParameters(minWindow, maxWindow, winInc);

            if (randomCVAccEnsemble){
                classifiersBuilt = new int[numSeries];
                lowestAccIdx = new int[numSeries];
                lowestAcc = new double[numSeries];
                for (int i = 0; i < numSeries; i++) lowestAcc[i] = Double.MAX_VALUE;

                if (getEstimateOwnPerformance()){
                    filterTrainPreds = new ArrayList[numSeries];
                    filterTrainIdx  = new ArrayList[numSeries];
                    for (int n = 0; n < numSeries; n++){
                        filterTrainPreds[n] = new ArrayList();
                        filterTrainIdx[n] = new ArrayList();
                    }
                }
            }
        }

/*        if (memoryContract) {
            try {
                SizeOf.deepSizeOf("test");
            } catch (IllegalStateException e) {
                throw new Exception("Unable to contract memory with SizeOf unavailable, " +
                            "enable by linking to SizeOf.jar in VM options i.e. -javaagent:lib/SizeOf.jar");
            }
        }
*/
        train = data;

        if (getEstimateOwnPerformance()){
            trainDistributions = new double[data.numInstances()][data.numClasses()];
            idxSubsampleCount = new int[data.numInstances()];
        }

        if (multiThread){
            if (numThreads == 1) numThreads = Runtime.getRuntime().availableProcessors();
            if (ex == null) ex = Executors.newFixedThreadPool(numThreads);
        }

        //required to deal with multivariate datasets, each channel is split into its own instances
        Instances[] series;

        //Multivariate
        if (isMultivariate) {
            series = splitMultivariateInstances(data);
            seriesHeader = new Instances(series[0], 0);
        }
        //Univariate
        else{
            series = new Instances[1];
            series[0] = data;
        }

        //Contracting
        if (trainTimeContract){
            ensembleSize = 0;
            underContractTime = true;
        }

        //If checkpointing and flag is set stop building.
        if (!(checkpoint && loadAndFinish)){
            //Randomly selected ensemble with accuracy filter
            if (randomCVAccEnsemble){
                buildRandomCVAccBOSS(series);
            }
            //Randomly selected ensemble
            else {
                buildRandomBOSS(series);
            }
        }

        //end train time in nanoseconds
        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        trainResults.setBuildTime(System.nanoTime() - trainResults.getBuildTime() - checkpointTimeDiff);

        //Estimate train accuracy
        if (getEstimateOwnPerformance()) {
            long start = System.nanoTime();
            ensembleCvAcc = findEnsembleTrainAcc(data);
            long end = System.nanoTime();
            trainResults.setErrorEstimateTime(end - start);
        }
        trainResults.setBuildPlusEstimateTime(trainResults.getBuildTime() + trainResults.getErrorEstimateTime());
        trainResults.setParas(getParameters());

        //delete any serialised files and holding folder for checkpointing on completion
        if (checkpoint && cleanupCheckpointFiles){
            checkpointCleanup();
        }
        trainResults.setParas(getParameters());

        if (randomCVAccEnsemble)
            printLineDebug("*************** Finished cBOSS Build with "+classifiersBuilt[0]+" Base BOSS evaluated " +
                "*************** in "+(System.nanoTime()-startTime)/1000000000+" Seconds. Number retained = " + classifiers[0].size());

    }

    private void buildRandomCVAccBOSS(Instances[] series) throws Exception {
        //build classifiers up to a set size
        while (((underContractTime || sum(classifiersBuilt) < ensembleSize) && underMemoryLimit) && parameterPool[numSeries-1].size() > 0) {
            long indivBuildTime = System.nanoTime();
            boolean checkpointChange = false;
            double[] parameters = selectParameters();
            if (parameters == null) continue;

            IndividualBOSS boss = new IndividualBOSS((int)parameters[0], (int)parameters[1], (int)parameters[2], parameters[3] == 1, multiThread, numThreads, ex);
            Instances data = resampleData(series[currentSeries], boss);
            boss.cleanAfterBuild = true;
            boss.seed = seed;
            boss.buildClassifier(data);
            boss.accuracy = individualTrainAcc(boss, data, numClassifiers[currentSeries] < maxEnsembleSize ? Double.MIN_VALUE : lowestAcc[currentSeries]);

            if (useWeights){
                boss.weight = Math.pow(boss.accuracy, 4);
                if (boss.weight == 0) boss.weight = Double.MIN_VALUE;
            }

            if (bayesianParameterSelection) paramAccuracy[currentSeries].add(boss.accuracy);
            if (trainTimeContract) paramTime[currentSeries].add((double)(System.nanoTime() - indivBuildTime));
//            if (memoryContract) paramMemory[currentSeries].add((double)SizeOf.deepSizeOf(boss));

            if (numClassifiers[currentSeries] < maxEnsembleSize){
                if (boss.accuracy < lowestAcc[currentSeries]){
                    lowestAccIdx[currentSeries] = classifiersBuilt[currentSeries];
                    lowestAcc[currentSeries] = boss.accuracy;
                }
                classifiers[currentSeries].add(boss);
                numClassifiers[currentSeries]++;

                if (getEstimateOwnPerformance()){
                    filterTrainPreds[currentSeries].add(latestTrainPreds);
                    filterTrainIdx[currentSeries].add(latestTrainIdx);
                }
            }
            else if (boss.accuracy > lowestAcc[currentSeries]) {
                double[] newLowestAcc = findMinEnsembleAcc();
                lowestAccIdx[currentSeries] = (int)newLowestAcc[0];
                lowestAcc[currentSeries] = newLowestAcc[1];

                classifiers[currentSeries].remove(lowestAccIdx[currentSeries]);
                classifiers[currentSeries].add(lowestAccIdx[currentSeries], boss);

                if (getEstimateOwnPerformance()){
                    filterTrainPreds[currentSeries].remove(lowestAccIdx[currentSeries]);
                    filterTrainIdx[currentSeries].remove(lowestAccIdx[currentSeries]);
                    filterTrainPreds[currentSeries].add(lowestAccIdx[currentSeries], latestTrainPreds);
                    filterTrainIdx[currentSeries].add(lowestAccIdx[currentSeries], latestTrainIdx);
                }

                checkpointChange = true;
            }

            classifiersBuilt[currentSeries]++;

            int prev = currentSeries;
            if (isMultivariate) {
                nextSeries();
            }

            if (checkpoint) {
                if (classifiersBuilt[currentSeries] <= maxEnsembleSize) {
                    checkpoint(prev, -1, true);
                }
                else{
                    checkpoint(prev, lowestAccIdx[prev], checkpointChange);
                }
            }

            checkContracts();
        }

        if (cutoff){
            for (int n = 0; n < numSeries; n++) {
                double maxAcc = 0;
                for (int i = 0; i < classifiers[n].size(); i++){
                    if (classifiers[n].get(i).accuracy > maxAcc){
                        maxAcc = classifiers[n].get(i).accuracy;
                    }
                }

                for (int i = 0; i < classifiers[n].size(); i++){
                    IndividualBOSS b = classifiers[n].get(i);
                    if (b.accuracy < maxAcc * correctThreshold) {
                        classifiers[currentSeries].remove(i);

                        if (getEstimateOwnPerformance()){
                            filterTrainPreds[n].remove(i);
                            filterTrainIdx[n].remove(i);
                        }

                        numClassifiers[n]--;
                        i--;
                    }
                }
            }
        }

        if (getEstimateOwnPerformance()){
            for (int n = 0; n < numSeries; n++) {
                for (int i = 0; i < filterTrainIdx[n].size(); i++) {
                    ArrayList<Integer> trainIdx = filterTrainIdx[n].get(i);
                    ArrayList<Integer> trainPreds = filterTrainPreds[n].get(i);
                    double weight = classifiers[n].get(i).weight;
                    for (int g = 0; g < trainIdx.size(); g++) {
                        idxSubsampleCount[trainIdx.get(g)] += weight;
                        trainDistributions[trainIdx.get(g)][trainPreds.get(g)] += weight;
                    }
                }
            }

            filterTrainPreds = null;
            filterTrainIdx = null;
            latestTrainPreds = null;
            latestTrainIdx = null;

            for (int i = 0; i < trainDistributions.length; i++){
                if (idxSubsampleCount[i] > 0) {
                    for (int n = 0; n < trainDistributions[i].length; n++) {
                        trainDistributions[i][n] /= idxSubsampleCount[i];
                    }
                    //System.out.println(Arrays.toString(trainDistributions[i]));
                }
            }
        }
    }

    private void buildRandomBOSS(Instances[] series) throws Exception {
        //build classifiers up to a set size
        while ((((underContractTime && numClassifiers[numSeries-1] < maxEnsembleSize)
                || sum(numClassifiers) < ensembleSize) && underMemoryLimit) && parameterPool[numSeries-1].size() > 0) {
            long indivBuildTime = System.nanoTime();
            double[] parameters = selectParameters();
            if (parameters == null) continue;

            IndividualBOSS boss = new IndividualBOSS((int)parameters[0], (int)parameters[1], (int)parameters[2], parameters[3] == 1, multiThread, numThreads, ex);
            Instances data = resampleData(series[currentSeries], boss);
            boss.cleanAfterBuild = true;
            boss.seed = seed;
            boss.buildClassifier(data);
            classifiers[currentSeries].add(boss);
            numClassifiers[currentSeries]++;

            if (useWeights){
                if (boss.accuracy == -1) boss.accuracy = individualTrainAcc(boss, data, Double.MIN_VALUE);
                boss.weight = Math.pow(boss.accuracy, 4);
                if (boss.weight == 0) boss.weight = Double.MIN_VALUE;
            }

            if (bayesianParameterSelection) {
                if (boss.accuracy == -1) boss.accuracy = individualTrainAcc(boss, data, Double.MIN_VALUE);
                paramAccuracy[currentSeries].add(boss.accuracy);
            }
            if (trainTimeContract) paramTime[currentSeries].add((double)(System.nanoTime() - indivBuildTime));
//            if (memoryContract) paramMemory[currentSeries].add((double)SizeOf.deepSizeOf(boss));

            if (getEstimateOwnPerformance()){
                if (boss.accuracy == -1) boss.accuracy = individualTrainAcc(boss, data, Double.MIN_VALUE);
                for (int i = 0; i < latestTrainIdx.size(); i++){
                    idxSubsampleCount[latestTrainIdx.get(i)] += boss.weight;
                    trainDistributions[latestTrainIdx.get(i)][latestTrainPreds.get(i)] += boss.weight;
                }
            }

            int prev = currentSeries;
            if (isMultivariate){
                nextSeries();
            }

            if (checkpoint) {
                checkpoint(prev, -1, true);
            }

            checkContracts();
        }

        if (getEstimateOwnPerformance()){
            latestTrainPreds = null;
            latestTrainIdx = null;

            for (int i = 0; i < trainDistributions.length; i++){
                if (idxSubsampleCount[i] > 0) {
                    for (int n = 0; n < trainDistributions[i].length; n++) {
                        trainDistributions[i][n] /= idxSubsampleCount[i];
                    }
                }
            }
        }
    }

    private void checkpoint(int seriesNo, int classifierNo, boolean saveIndiv){
        if(checkpointPath!=null){
            try{
                File f = new File(checkpointPath);
                if(!f.isDirectory())
                    f.mkdirs();
                //time the checkpoint occured
                checkpointTime = System.nanoTime();

                if (saveIndiv) {
                    if (seriesNo >= 0) {
                        if (classifierNo < 0) classifierNo = classifiers[seriesNo].size() - 1;

                        //save the last build individual classifier
                        IndividualBOSS indiv = classifiers[seriesNo].get(classifierNo);

                        FileOutputStream fos = new FileOutputStream(checkpointPath + "cBOSSIndividual" + seed + seriesNo + "-" + classifierNo + ".ser");
                        try (ObjectOutputStream out = new ObjectOutputStream(fos)) {
                            out.writeObject(indiv);
                            out.close();
                            fos.close();
                        }
                    }
                }

                //dont take into account time spent serialising into build time
                checkpointTimeDiff += System.nanoTime() - checkpointTime;
                checkpointTime = System.nanoTime();

                //save this, classifiers and train data not included
                saveToFile(checkpointPath + "cBOSS" + seed + "temp.ser");

                File file = new File(checkpointPath + "cBOSS" + seed + "temp.ser");
                File file2 = new File(checkpointPath + "cBOSS" + seed + ".ser");
                file2.delete();
                file.renameTo(file2);

                checkpointTimeDiff += System.nanoTime() - checkpointTime;
            }
            catch(Exception e){
                e.printStackTrace();
                System.out.println("Serialisation to "+checkpointPath+" FAILED");
            }
        }
    }

    private void checkpointCleanup(){
        File f = new File(checkpointPath);
        String[] files = f.list();

        for (String file: files){
            File f2 = new File(f.getPath() + "\\" + file);
            boolean b = f2.delete();
        }

        f.delete();
    }

    private String checkpointName(String datasetName){
        String name = datasetName + seed + "cBOSS";

        if (trainTimeContract){
            name += ("TTC" + trainContractTimeNanos);
        }
        else if (isMultivariate && ensembleSizePerChannel > 0){
            name += ("PC" + (ensembleSizePerChannel*numSeries));
        }
        else{
            name += ("S" + ensembleSize);
        }

        if (memoryContract){
            name += ("MC" + memoryLimit);
        }

        if (randomCVAccEnsemble) {
            name += ("M" + maxEnsembleSize);
        }

        if (useWeights){
            name += "W";
        }

        return name;
    }

    public void checkContracts(){
        underContractTime = System.nanoTime() - trainResults.getBuildTime() - checkpointTimeDiff < trainContractTimeNanos;
        underMemoryLimit = !memoryContract || bytesUsed < memoryLimit;
    }

    //[0] = index, [1] = acc
    private double[] findMinEnsembleAcc() {
        double minAcc = Double.MAX_VALUE;
        int minAccInd = 0;
        for (int i = 0; i < classifiers[currentSeries].size(); ++i) {
            double curacc = classifiers[currentSeries].get(i).accuracy;
            if (curacc < minAcc) {
                minAcc = curacc;
                minAccInd = i;
            }
        }

        return new double[] { minAccInd, minAcc };
    }

    private Instances[] uniqueParameters(int minWindow, int maxWindow, int winInc){
        Instances[] parameterPool = new Instances[numSeries];
        ArrayList<double[]> possibleParameters = new ArrayList();

        for (Boolean normalise: normOptions) {
            for (Integer alphSize : alphabetSize) {
                for (int winSize = minWindow; winSize <= maxWindow; winSize += winInc) {
                    for (Integer wordLen : wordLengths) {
                        double[] parameters = {wordLen, alphSize, winSize, normalise ? 1 : 0};
                        possibleParameters.add(parameters);
                    }
                }
            }
        }

        int numAtts = possibleParameters.get(0).length+1;
        ArrayList<Attribute> atts = new ArrayList<>(numAtts);
        for (int i = 0; i < numAtts; i++){
            atts.add(new Attribute("att" + i));
        }

        prevParameters = new Instances[numSeries];
        initialParameterCount = new int[numSeries];

        for (int n = 0; n < numSeries; n++) {
            parameterPool[n] = new Instances("params", atts, possibleParameters.size());
            parameterPool[n].setClassIndex(numAtts-1);
            prevParameters[n] = new Instances(parameterPool[n], 0);
            prevParameters[n].setClassIndex(numAtts-1);

            for (int i = 0; i < possibleParameters.size(); i++) {
                DenseInstance inst = new DenseInstance(1, possibleParameters.get(i));
                inst.insertAttributeAt(numAtts-1);
                parameterPool[n].add(inst);
            }
        }

        if (bayesianParameterSelection){
            paramAccuracy = new ArrayList[numSeries];
            for (int i = 0; i < numSeries; i++){
                paramAccuracy[i] = new ArrayList<>();
            }
        }
        if (trainTimeContract){
            paramTime = new ArrayList[numSeries];
            for (int i = 0; i < numSeries; i++){
                paramTime[i] = new ArrayList<>();
            }
        }
        if (memoryContract){
            paramMemory = new ArrayList[numSeries];
            for (int i = 0; i < numSeries; i++){
                paramMemory[i] = new ArrayList<>();
            }
        }

        return parameterPool;
    }

    private double[] selectParameters() throws Exception {
        Instance params;

        if (trainTimeContract) {
            if (prevParameters[currentSeries].size() > 0) {
                for (int i = 0; i < paramTime[currentSeries].size(); i++) {
                    prevParameters[currentSeries].get(i).setClassValue(paramTime[currentSeries].get(i));
                }

                GaussianProcesses gp = new GaussianProcesses();
                gp.buildClassifier(prevParameters[currentSeries]);
                long remainingTime = trainContractTimeNanos - (System.nanoTime() - trainResults.getBuildTime() - checkpointTimeDiff);

                for (int i = 0; i < parameterPool[currentSeries].size(); i++) {
                    double pred = gp.classifyInstance(parameterPool[currentSeries].get(i));
                    if (pred > remainingTime) {
                        parameterPool[currentSeries].remove(i);
                        i--;
                    }
                }
            }
        }

        if (memoryContract) {
            if (prevParameters[currentSeries].size() > 0) {
                for (int i = 0; i < paramMemory[currentSeries].size(); i++) {
                    prevParameters[currentSeries].get(i).setClassValue(paramMemory[currentSeries].get(i));
                }

                GaussianProcesses gp = new GaussianProcesses();
                gp.buildClassifier(prevParameters[currentSeries]);
                long remainingMemory = memoryLimit - bytesUsed;

                for (int i = 0; i < parameterPool[currentSeries].size(); i++) {
                    double pred = gp.classifyInstance(parameterPool[currentSeries].get(i));
                    if (pred > remainingMemory) {
                        parameterPool[currentSeries].remove(i);
                        i--;
                    }
                }
            }
        }

        if (parameterPool[currentSeries].size() == 0){
            return null;
        }

        if (bayesianParameterSelection) {
            if (initialParameterCount[currentSeries] < initialRandomParameters) {
                initialParameterCount[currentSeries]++;
                params = parameterPool[currentSeries].remove(rand.nextInt(parameterPool[currentSeries].size()));
            } else {
                for (int i = 0; i < paramAccuracy[currentSeries].size(); i++){
                    prevParameters[currentSeries].get(i).setClassValue(paramAccuracy[currentSeries].get(i));
                }

                GaussianProcesses gp = new GaussianProcesses();
                gp.buildClassifier(prevParameters[currentSeries]);
                int bestIndex = 0;
                double bestAcc = -1;

                for (int i = 0; i < parameterPool[currentSeries].numInstances(); i++) {
                    double pred = gp.classifyInstance(parameterPool[currentSeries].get(i));

                    if (pred > bestAcc){
                        bestIndex = i;
                        bestAcc = pred;
                    }
                }

                params = parameterPool[currentSeries].remove(bestIndex);
            }
        }
        else {
            params = parameterPool[currentSeries].remove(rand.nextInt(parameterPool[currentSeries].size()));
        }

        prevParameters[currentSeries].add(params);
        return params.toDoubleArray();
    }

    private Instances resampleData(Instances series, IndividualBOSS boss){
        Instances data;
        int newSize;

        if (trainProportion > 0){
            newSize = (int)(series.numInstances()*trainProportion);
        }
        else{
            newSize = maxTrainInstances;
        }

        if (reduceTrainInstances && series.numInstances() > newSize){
            Sampler sampler;

            if (stratifiedSubsample){
                sampler = new RandomStratifiedIndexSampler(rand);
            }
            else{
                sampler = new RandomIndexSampler(rand);
            }

            sampler.setInstances(series);
            data = new Instances(series, newSize);
            boss.subsampleIndices = new ArrayList<>(newSize);

            for (int i = 0; i < newSize; i++){
                int n = (Integer)sampler.next();
                data.add(series.get(n));
                boss.subsampleIndices.add(n);
            }
        }
        else{
            data = series;
        }

        return data;
    }

    private double individualTrainAcc(IndividualBOSS boss, Instances series, double lowestAcc) throws Exception {
        int[] indicies;

        if (getEstimateOwnPerformance()){
            latestTrainPreds = new ArrayList();
            latestTrainIdx = new ArrayList();
        }

        if (useFastTrainEstimate && maxEval < series.numInstances()){
            RandomRoundRobinIndexSampler sampler = new RandomRoundRobinIndexSampler(rand);
            sampler.setInstances(series);
            indicies = new int[maxEval];

            for (int i = 0; i < maxEval; ++i) {
                int subsampleIndex = sampler.next();
                indicies[i] = subsampleIndex;
            }
        }
        else {
            indicies = new int[series.numInstances()];

            for (int i = 0; i < series.numInstances(); ++i) {
                indicies[i] = i;
            }
        }

        int correct = 0;
        int numInst = indicies.length;
        int requiredCorrect = (int)(lowestAcc*numInst);

        if (multiThread){
            ArrayList<Future<Double>> futures = new ArrayList<>(numInst);

            for (int i = 0; i < numInst; ++i)
                futures.add(ex.submit(boss.new TrainNearestNeighbourThread(i)));

            int idx = 0;
            for (Future<Double> f: futures){
                if (f.get() == series.get(idx).classValue()) {
                    ++correct;
                }
                idx++;
            }
        }
        else {
            for (int i = 0; i < numInst; ++i) {
                if (correct + numInst - i < requiredCorrect) {
                    return -1;
                }

                double c = boss.classifyInstance(indicies[i]); //classify series i, while ignoring its corresponding histogram i
                if (c == series.get(indicies[i]).classValue()) {
                    ++correct;
                }

                if (getEstimateOwnPerformance()){
                    latestTrainPreds.add((int)c);
                    if (boss.subsampleIndices != null) {
                        latestTrainIdx.add(boss.subsampleIndices.get(indicies[i]));
                    }
                    else {
                        latestTrainIdx.add(indicies[i]);
                    }
                }
            }
        }

        return (double) correct / (double) numInst;
    }

    public void nextSeries(){
        if (currentSeries == numSeries-1){
            currentSeries = 0;
        }
        else{
            currentSeries++;
        }
    }

    private double findEnsembleTrainAcc(Instances data) throws Exception {
        this.ensembleCvPreds = new double[data.numInstances()];
        int totalClassifers = sum(numClassifiers);
        double correct = 0;

        trainResults.setClassifierName(getClassifierName());
        trainResults.setDatasetName(data.relationName());
        trainResults.setFoldID(seed);
        trainResults.setSplit("train");
        trainResults.setParas(getParameters());

        if (idxSubsampleCount == null) idxSubsampleCount = new int[train.numInstances()];

        for (int i = 0; i < data.numInstances(); ++i) {
            double[] probs;
            if (idxSubsampleCount[i] > 0 && (!fullTrainCVEstimate || idxSubsampleCount[i] == totalClassifers)){
                probs = trainDistributions[i];
            }
            else {
                probs = distributionForInstance(i);
            }

            int maxClass = findIndexOfMax(probs, rand);
            if (maxClass == data.get(i).classValue())
                ++correct;
            this.ensembleCvPreds[i] = maxClass;

            trainResults.addPrediction(data.get(i).classValue(), probs, maxClass, -1, "");
        }
        trainResults.finaliseResults();

        return correct / data.numInstances();
    }

    public double getTrainAcc(){
        if(ensembleCvAcc >= 0){
            return this.ensembleCvAcc;
        }

        try{
            return this.findEnsembleTrainAcc(train);
        }catch(Exception e){
            e.printStackTrace();
        }

        return -1;
    }

    public double[] getTrainPreds(){
        if(this.ensembleCvPreds == null){
            try{
                this.findEnsembleTrainAcc(train);
            }catch(Exception e){
                e.printStackTrace();
            }
        }

        return this.ensembleCvPreds;
    }

    //potentially scuffed when train set is subsampled, will have to revisit and discuss if this is a viable option
    //for estimation anyway.
    private double[] distributionForInstance(int test) throws Exception {
        int numClasses = train.numClasses();
        double[] classHist = new double[numClasses];

        //get sum of all channels, votes from each are weighted the same.
        double sum = 0;

        for (int n = 0; n < numSeries; n++) {
            for (IndividualBOSS classifier : classifiers[n]) {
                double classification;

                if (classifier.subsampleIndices == null){
                    classification = classifier.classifyInstance(test);
                }
                else if (classifier.subsampleIndices.contains(test)){
                    classification = classifier.classifyInstance(classifier.subsampleIndices.indexOf(test));
                }
                else if (fullTrainCVEstimate) {
                    Instance series = train.get(test);
                    if (isMultivariate){
                        series = splitMultivariateInstance(series)[n];
                        series.setDataset(seriesHeader);
                    }
                    classification = classifier.classifyInstance(series);
                }
                else{
                    continue;
                }

                classHist[(int) classification] += classifier.weight;
                sum += classifier.weight;
            }
        }

        double[] distributions = new double[numClasses];

        if (sum != 0) {
            for (int i = 0; i < classHist.length; ++i)
                distributions[i] += (classHist[i] / sum);
        }
        else{
            for (int i = 0; i < classHist.length; ++i)
                distributions[i] += 1 / numClasses;
        }

        return distributions;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] probs = distributionForInstance(instance);
        return findIndexOfMax(probs, rand);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        int numClasses = train.numClasses();
        double[] classHist = new double[numClasses];

        //get sum of all channels, votes from each are weighted the same.
        double sum = 0;

        Instance[] series;

        //Multivariate
        if (isMultivariate) {
            series = splitMultivariateInstanceWithClassVal(instance);
        }
        //Univariate
        else {
            series = new Instance[1];
            series[0] = instance;
        }

        if (multiThread){
            ArrayList<Future<Double>>[] futures = new ArrayList[numSeries];

            for (int n = 0; n < numSeries; n++) {
                futures[n] = new ArrayList<>(numClassifiers[n]);
                for (IndividualBOSS classifier : classifiers[n]) {
                    futures[n].add(ex.submit(classifier.new TestNearestNeighbourThread(instance)));
                }
            }

            for (int n = 0; n < numSeries; n++) {
                int idx = 0;
                for (Future<Double> f : futures[n]) {
                    double weight = classifiers[n].get(idx).weight;
                    classHist[f.get().intValue()] += weight;
                    sum += weight;
                    idx++;
                }
            }
        }
        else {
            for (int n = 0; n < numSeries; n++) {
                for (IndividualBOSS classifier : classifiers[n]) {
                    double classification = classifier.classifyInstance(series[n]);
                    classHist[(int) classification] += classifier.weight;
                    sum += classifier.weight;
                }
            }
        }

        double[] distributions = new double[instance.numClasses()];

        if (sum != 0) {
            for (int i = 0; i < classHist.length; ++i)
                distributions[i] += classHist[i] / sum;
        }
        else{
            for (int i = 0; i < classHist.length; ++i)
                distributions[i] += 1 / numClasses;
        }

        return distributions;
    }

    public static void main(String[] args) throws Exception{
        int fold = 0;

        //Minimum working example
        String dataset = "ItalyPowerDemand";
        Instances train = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\"+dataset+"\\"+dataset+"_TRAIN.arff");
        Instances test = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\"+dataset+"\\"+dataset+"_TEST.arff");
        Instances[] data = resampleTrainAndTestInstances(train, test, fold);
        train = data[0];
        test = data[1];

        String dataset2 = "ERing";
        Instances train2 = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Multivariate_arff\\"+dataset2+"\\"+dataset2+"_TRAIN.arff");
        Instances test2 = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Multivariate_arff\\"+dataset2+"\\"+dataset2+"_TEST.arff");
        Instances[] data2 = resampleMultivariateTrainAndTestInstances(train2, test2, fold);
        train2 = data2[0];
        test2 = data2[1];

        cBOSS c;
        double accuracy;

        c = new cBOSS(false);
        c.useRecommendedSettings();
        c.setSeed(fold);
        c.setEstimateOwnPerformance(true);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("CVAcc CAWPE BOSS accuracy on " + dataset + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new cBOSS(false);
        c.useRecommendedSettings();
        c.setSeed(fold);
        c.setEstimateOwnPerformance(true);
        c.buildClassifier(train2);
        accuracy = ClassifierTools.accuracy(test2, c);

        System.out.println("CVAcc CAWPE BOSS accuracy on " + dataset2 + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new cBOSS(false);
        c.useRecommendedSettings();
        c.bayesianParameterSelection = true;
        c.setSeed(fold);
        c.setEstimateOwnPerformance(true);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("Bayesian CVAcc CAWPE BOSS accuracy on " + dataset + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new cBOSS(false);
        c.useRecommendedSettings();
        c.bayesianParameterSelection = true;
        c.setSeed(fold);
        c.setEstimateOwnPerformance(true);
        c.buildClassifier(train2);
        accuracy = ClassifierTools.accuracy(test2, c);

        System.out.println("Bayesian CVAcc CAWPE BOSS accuracy on " + dataset2 + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new cBOSS(false);
        c.ensembleSize = 250;
        c.setMaxEnsembleSize(50);
        c.setRandomCVAccEnsemble(true);
        c.setSeed(fold);
        c.useFastTrainEstimate = true;
        c.reduceTrainInstances = true;
        c.setMaxEvalPerClass(50);
        c.setMaxTrainInstances(500);
        c.setEstimateOwnPerformance(true);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("FastMax CVAcc BOSS accuracy on " + dataset + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new cBOSS(false);
        c.ensembleSize = 250;
        c.setMaxEnsembleSize(50);
        c.setRandomCVAccEnsemble(true);
        c.setSeed(fold);
        c.useFastTrainEstimate = true;
        c.reduceTrainInstances = true;
        c.setMaxEvalPerClass(50);
        c.setMaxTrainInstances(500);
        c.setEstimateOwnPerformance(true);
        c.buildClassifier(train2);
        accuracy = ClassifierTools.accuracy(test2, c);

        System.out.println("FastMax CVAcc BOSS accuracy on " + dataset2 + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new cBOSS(false);
        c.ensembleSize = 100;
        c.useWeights(true);
        c.setSeed(fold);
        c.setReduceTrainInstances(true);
        c.setTrainProportion(0.7);
        c.setEstimateOwnPerformance(true);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("CAWPE Subsample BOSS accuracy on " + dataset + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new cBOSS(false);
        c.ensembleSize = 100;
        c.useWeights(true);
        c.setSeed(fold);
        c.setReduceTrainInstances(true);
        c.setTrainProportion(0.7);
        c.setEstimateOwnPerformance(true);
        c.buildClassifier(train2);
        accuracy = ClassifierTools.accuracy(test2, c);

        System.out.println("CAWPE Subsample BOSS accuracy on " + dataset2 + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new cBOSS(false);
        c.setTrainTimeLimit(TimeUnit.MINUTES, 1);
        c.setCleanupCheckpointFiles(true);
        c.setCheckpointPath("D:\\");
        c.setSeed(fold);
        c.setEstimateOwnPerformance(true);
        long startTime = System.nanoTime();
        c.buildClassifier(train);
        long endTime =System.nanoTime() - startTime;
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("Contract 1 Min Checkpoint BOSS accuracy on " + dataset + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers) + " in " + endTime*1e-9 + " seconds");

        c = new cBOSS(false);
        c.setTrainTimeLimit(TimeUnit.MINUTES, 1);
        c.setCleanupCheckpointFiles(true);
        c.setCheckpointPath("D:\\");
        c.setSeed(fold);
        c.setEstimateOwnPerformance(true);
        long startTime2 = System.nanoTime();
        c.buildClassifier(train2);
        long endTime2 = System.nanoTime() - startTime2;
        accuracy = ClassifierTools.accuracy(test2, c);

        System.out.println("Contract 1 Min Checkpoint BOSS accuracy on " + dataset2 + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers)  + " in " + endTime2*1e-9 + " seconds");

//        c = new cBOSS(false);
//        c.setMemoryLimit(DataUnit.MEGABYTE, 500);
//        c.setSeed(fold);
//        c.setEstimateOwnPerformance(true);
//        c.buildClassifier(train);
//        accuracy = ClassifierTools.accuracy(test, c);
//
//        System.out.println("Contract 500MB BOSS accuracy on " + dataset + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));
//
//        c = new cBOSS(false);
//        c.setMemoryLimit(DataUnit.MEGABYTE, 500);
//        c.setSeed(fold);
//        c.setEstimateOwnPerformance(true);
//        c.buildClassifier(train2);
//        accuracy = ClassifierTools.accuracy(test2, c);

        System.out.println("Contract 500MB BOSS accuracy on " + dataset2 + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        //Output 19/11/19
        /*
        JAVAGENT: call premain instrumentation for class SizeOf
        CVAcc CAWPE BOSS accuracy on ItalyPowerDemand fold 0 = 0.923226433430515 numClassifiers = [50]
        CVAcc CAWPE BOSS accuracy on ERing fold 0 = 0.8962962962962963 numClassifiers = [50, 50, 50, 50]
        Bayesian CVAcc CAWPE BOSS accuracy on ItalyPowerDemand fold 0 = 0.9300291545189504 numClassifiers = [50]
        Bayesian CVAcc CAWPE BOSS accuracy on ERing fold 0 = 0.8962962962962963 numClassifiers = [50, 50, 50, 50]
        FastMax CVAcc BOSS accuracy on ItalyPowerDemand fold 0 = 0.8415937803692906 numClassifiers = [50]
        FastMax CVAcc BOSS accuracy on ERing fold 0 = 0.725925925925926 numClassifiers = [50, 50, 50, 50]
        CAWPE Subsample BOSS accuracy on ItalyPowerDemand fold 0 = 0.9271137026239067 numClassifiers = [80]
        CAWPE Subsample BOSS accuracy on ERing fold 0 = 0.8481481481481481 numClassifiers = [25, 25, 25, 25]
        Contract 1 Min Checkpoint BOSS accuracy on ItalyPowerDemand fold 0 = 0.6958211856171039 numClassifiers = [80] in 2.2928520000000003 seconds
        Contract 1 Min Checkpoint BOSS accuracy on ERing fold 0 = 0.5259259259259259 numClassifiers = [190, 190, 190, 190] in 27.359452200000003 seconds
        Contract 500MB BOSS accuracy on ItalyPowerDemand fold 0 = 0.7094266277939747 numClassifiers = [50]
        Contract 500MB BOSS accuracy on ERing fold 0 = 0.4777777777777778 numClassifiers = [13, 13, 12, 12]
        */
    }
}
