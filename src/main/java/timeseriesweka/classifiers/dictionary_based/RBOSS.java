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
package timeseriesweka.classifiers.dictionary_based;

import java.security.InvalidParameterException;
import java.util.*;

import net.sourceforge.sizeof.SizeOf;
import timeseriesweka.classifiers.*;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

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
 * BOSS classifier with parameter search and ensembling for univariate and
 * multivariate time series classification.
 * If parameters are known, use the nested class BOSSIndividual and directly provide them.
 *
 * Options to change the method of ensembling to randomly select parameters instead of searching.
 * Has the capability to contract train time and checkpoint when using a random ensemble.
 *
 * Alphabetsize fixed to four and maximum wordLength of 16.
 *
 * @author James Large, updated by Matthew Middlehurst
 *
 * Implementation based on the algorithm described in getTechnicalInformation()
 */
public class RBOSS extends AbstractClassifierWithTrainingInfo implements TrainAccuracyEstimator, TrainTimeContractable,
        MemoryContractable, Checkpointable, TechnicalInformationHandler, MultiThreadable {

    private ArrayList<Double>[] paramAccuracy;
    private ArrayList<Double>[] paramTime;
    private ArrayList<Double>[] paramMemory;

    private int ensembleSize = 50;
    private int ensembleSizePerChannel = -1;
    private int seed = 0;
    private Random rand;
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

    private transient LinkedList<BOSSIndividual>[] classifiers;
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
    private String serPath;
    private boolean checkpoint = false;
    private long checkpointTime = 0;
    private long checkpointTimeDiff = 0;
    private boolean cleanupCheckpointFiles = true;
    private boolean loadAndFinish = false;

    private long contractTime = 0;
    private boolean trainTimeContract = false;
    private boolean underContractTime = false;

    private long memoryLimit = 0;
    private long bytesUsed = 0;
    private boolean memoryContract = false;
    private boolean underMemoryLimit = true;

    //RBOSS CV acc variables, stored as field for checkpointing.
    private int[] classifiersBuilt;
    private int[] lowestAccIdx;
    private double[] lowestAcc;

    private String trainCVPath;
    private boolean trainCV = false;

    private transient Instances train;
    private double ensembleCvAcc = -1;
    private double[] ensembleCvPreds = null;

    private int numThreads = 1;
    private boolean multiThread = false;
    private ExecutorService ex;

    protected static final long serialVersionUID = 22554L;

    public RBOSS() {}

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "P. Schafer");
        result.setValue(TechnicalInformation.Field.TITLE, "The BOSS is concerned with time series classification in the presence of noise");
        result.setValue(TechnicalInformation.Field.JOURNAL, "Data Mining and Knowledge Discovery");
        result.setValue(TechnicalInformation.Field.VOLUME, "29");
        result.setValue(TechnicalInformation.Field.NUMBER,"6");
        result.setValue(TechnicalInformation.Field.PAGES, "1505-1530");
        result.setValue(TechnicalInformation.Field.YEAR, "2015");
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
                BOSSIndividual boss = classifiers[n].get(i);
                sb.append(",windowSize,").append(boss.getWindowSize()).append(",wordLength,").append(boss.getWordLength());
                sb.append(",alphabetSize,").append(boss.getAlphabetSize()).append(",norm,").append(boss.isNorm());
            }
        }

        return sb.toString();
    }

    public void useRecommendedSettingsRBOSS(){
        ensembleSize = 250;
        maxEnsembleSize = 50;
        randomCVAccEnsemble = true;
        useWeights = true;
        reduceTrainInstances = true;
        trainProportion = 0.7;
        bayesianParameterSelection = true;
    }

    //pass in an enum of hour, minute, day, and the amount of them.
    @Override
    public void setTrainTimeLimit(TimeUnit time, long amount){
        switch (time){
            case DAYS:
                contractTime = (long)(8.64e+13)*amount;
                break;
            case HOURS:
                contractTime = (long)(3.6e+12)*amount;
                break;
            case MINUTES:
                contractTime = (long)(6e+10)*amount;
                break;
            case SECONDS:
                contractTime = (long)(1e+9)*amount;
                break;
            case NANOSECONDS:
                contractTime = amount;
                break;
            default:
                throw new InvalidParameterException("Invalid time unit");
        }
        trainTimeContract = true;
    }

    @Override
    public void setMemoryLimit(DataUnit unit, long amount){
        switch (unit){
            case GIGABYTE:
                memoryLimit = amount*1073741824;
                break;
            case MEGABYTE:
                memoryLimit = amount*1048576;
                break;
            case BYTES:
                memoryLimit = amount;
                break;
            default:
                throw new InvalidParameterException("Invalid data unit");
        }
        memoryContract = true;
    }

    @Override
    public void setThreadAllowance(int numThreads) {
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
    @Override
    public void setSavePath(String path){
        checkpointPath = path;
        checkpoint = true;
    }

    //Define how to copy from a loaded object to this object
    @Override
    public void copyFromSerObject(Object obj) throws Exception{
        if(!(obj instanceof RBOSS))
            throw new Exception("The SER file is not an instance of BOSS");
        RBOSS saved = ((RBOSS)obj);
        System.out.println("Loading BOSS.ser");

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
        contractTime = saved.contractTime;
        trainTimeContract = saved.trainTimeContract;
        underContractTime = saved.underContractTime;
        memoryLimit = saved.memoryLimit;
        bytesUsed = saved.bytesUsed;
        memoryContract = saved.memoryContract;
        underMemoryLimit = saved.underMemoryLimit;
        classifiersBuilt = saved.classifiersBuilt;
        lowestAccIdx = saved.lowestAccIdx;
        lowestAcc = saved.lowestAcc;
        trainCVPath = saved.trainCVPath;
        trainCV = saved.trainCV;
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
                System.out.println("Loading BOSSIndividual" + n + "-" + i + ".ser");

                FileInputStream fis = new FileInputStream(serPath + "BOSSIndividual" + n + "-" + i + ".ser");
                try (ObjectInputStream in = new ObjectInputStream(fis)) {
                    Object indv = in.readObject();

                    if (!(indv instanceof BOSSIndividual))
                        throw new Exception("The SER file " + n + "-" + i + " is not an instance of BOSSIndividual");
                    BOSSIndividual ser = ((BOSSIndividual) indv);
                    classifiers[n].add(ser);
                }
            }
        }

        checkpointTimeDiff = saved.checkpointTimeDiff + (System.nanoTime() - checkpointTime);
    }

    @Override
    public void writeTrainEstimatesToFile(String outputPathAndName){
        trainCVPath = outputPathAndName;
        trainCV = true;
    }

    @Override
    public void setFindTrainAccuracyEstimate(boolean setCV){
        trainCV = setCV;
    }

    @Override
    public boolean findsTrainAccuracyEstimate(){ return trainCV; }

    @Override
    public ClassifierResults getTrainResults(){
        trainResults.setAcc(ensembleCvAcc);
        return trainResults;
    }

    public void setEnsembleSize(int size) {
        ensembleSize = size;
    }

    public void setMaxEnsembleSize(int size) {
        maxEnsembleSize = size;
    }

    public void setSeed(int i) {
        seed = i;
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
        trainResults.setBuildTime(System.nanoTime());

        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        if(data.checkForAttributeType(Attribute.RELATIONAL)){
            isMultivariate = true;
        }

        //path checkpoint files will be saved to
        serPath = checkpointPath + "/" + checkpointName(data.relationName()) + "/";
        File f = new File(serPath + "BOSS.ser");

        //Window length settings
        int seriesLength = isMultivariate ? channelLength(data)-1 : data.numAttributes()-1; //minus class attribute
        int minWindow = 10;
        int maxWindow = (int)(seriesLength*maxWinLenProportion);
        if (maxWindow < minWindow) minWindow = maxWindow/2;
        //whats the max number of window sizes that should be searched through
        double maxWindowSearches = seriesLength*maxWinSearchProportion;
        int winInc = (int)((maxWindow - minWindow) / maxWindowSearches);
        if (winInc < 1) winInc = 1;

        //if checkpointing and serialised files exist load said files
        if (checkpoint && f.exists()){
            long time = System.nanoTime();
            loadFromFile(serPath + "BOSS.ser");
            System.out.println("Spent " + (System.nanoTime() - time) + "nanoseconds loading files");
        }
        //initialise variables
        else {
            if (data.classIndex() != data.numAttributes()-1)
                throw new Exception("BOSS_BuildClassifier: Class attribute not set as last attribute in dataset");

            //Multivariate
            if (isMultivariate) {
                numSeries = numChannels(data);
                classifiers = new LinkedList[numSeries];

                for (int n = 0; n < numSeries; n++){
                    classifiers[n] = new LinkedList<>();
                }

                numClassifiers = new int[numSeries];

                if (ensembleSizePerChannel > 0){
                    ensembleSize = ensembleSizePerChannel*numSeries;
                }
            }
            //Univariate
            else{
                numSeries = 1;
                classifiers = new LinkedList[1];
                classifiers[0] = new LinkedList<>();
                numClassifiers = new int[1];
            }

            if (maxEvalPerClass > 0){
                maxEval = data.numClasses()*maxEvalPerClass;
            }

            rand = new Random(seed);

            parameterPool = uniqueParameters(minWindow, maxWindow, winInc);
        }

        try{
            SizeOf.deepSizeOf("test");
        }
        catch (IllegalStateException e){
            if (memoryContract) {
                throw new Exception("Unable to contract memory with SizeOf unavailable, " +
                        "enable by linking to SizeOf.jar in VM options i.e. -javaagent:lib/SizeOf.jar");
            }
        }

        this.train = data;

        if (multiThread && numThreads == 1){
            numThreads = Runtime.getRuntime().availableProcessors();
        }

        //required to deal with multivariate datasets, each channel is split into its own instances
        Instances[] series;

        //Multivariate
        if (isMultivariate) {
            series = splitMultivariateInstances(data);
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
        trainResults.setBuildTime(System.nanoTime() - trainResults.getBuildTime() - checkpointTimeDiff);

        //Estimate train accuracy
        if (trainCV) {
            trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
            trainResults.setClassifierName("RBOSS");
            trainResults.setDatasetName(data.relationName());
            trainResults.setFoldID(seed);
            trainResults.setSplit("train");
            trainResults.setParas(getParameters());
            double result = findEnsembleTrainAcc(data);
            trainResults.finaliseResults();
            trainResults.writeFullResultsToFile(trainCVPath);

            System.out.println("CV acc ="+result);

            trainCV = false;
        }

        //delete any serialised files and holding folder for checkpointing on completion
        if (checkpoint && cleanupCheckpointFiles){
            checkpointCleanup();
        }
    }

    private void buildRandomCVAccBOSS(Instances[] series) throws Exception {
        classifiersBuilt = new int[numSeries];
        lowestAccIdx = new int[numSeries];
        lowestAcc = new double[numSeries];
        for (int i = 0; i < numSeries; i++) lowestAcc[i] = Double.MAX_VALUE;

        //build classifiers up to a set size
        while (((underContractTime || sum(classifiersBuilt) < ensembleSize) && underMemoryLimit) && parameterPool[numSeries-1].size() > 0) {
            long indivBuildTime = System.nanoTime();
            boolean checkpointChange = false;
            double[] parameters = selectParameters();
            if (parameters == null) continue;

            BOSSIndividual boss = new BOSSIndividual((int)parameters[0], (int)parameters[1], (int)parameters[2], parameters[3] == 1, multiThread, numThreads, ex);
            Instances data = resampleData(series[currentSeries], boss);
            boss.cleanAfterBuild = true;
            boss.seed = seed;
            boss.buildClassifier(data);
            boss.accuracy = individualTrainAcc(boss, data, numClassifiers[currentSeries] < maxEnsembleSize ? Double.MIN_VALUE : lowestAcc[currentSeries]);

            if (useWeights){
                boss.weight = Math.pow(boss.accuracy, 4);
                if (boss.weight == 0) boss.weight = 1;
            }

            if (bayesianParameterSelection) paramAccuracy[currentSeries].add(boss.accuracy);
            if (trainTimeContract) paramTime[currentSeries].add((double)(System.nanoTime() - indivBuildTime));
            if (memoryContract) paramMemory[currentSeries].add((double)SizeOf.deepSizeOf(boss));

            if (numClassifiers[currentSeries] < maxEnsembleSize){
                if (boss.accuracy < lowestAcc[currentSeries]){
                    lowestAccIdx[currentSeries] = classifiersBuilt[currentSeries];
                    lowestAcc[currentSeries] = boss.accuracy;
                }
                classifiers[currentSeries].add(boss);
                numClassifiers[currentSeries]++;
            }
            else if (boss.accuracy > lowestAcc[currentSeries]) {
                double[] newLowestAcc = findMinEnsembleAcc();
                lowestAccIdx[currentSeries] = (int)newLowestAcc[0];
                lowestAcc[currentSeries] = newLowestAcc[1];

                classifiers[currentSeries].remove(lowestAccIdx[currentSeries]);
                classifiers[currentSeries].add(lowestAccIdx[currentSeries], boss);
                checkpointChange = true;
            }

            classifiersBuilt[currentSeries]++;

            int prev = currentSeries;
            if (isMultivariate) {
                nextSeries();
            }

            if (checkpoint) {
                if (numClassifiers[currentSeries] < maxEnsembleSize) {
                    checkpoint(prev, -1);
                }
                else if (checkpointChange){
                    checkpoint(prev, lowestAccIdx[prev]);
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
                    BOSSIndividual b = classifiers[n].get(i);
                    if (b.accuracy < maxAcc * correctThreshold) {
                        classifiers[currentSeries].remove(i);
                        numClassifiers[n]--;
                        i--;
                    }
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

            BOSSIndividual boss = new BOSSIndividual((int)parameters[0], (int)parameters[1], (int)parameters[2], parameters[3] == 1, multiThread, numThreads, ex);
            Instances data = resampleData(series[currentSeries], boss);
            boss.cleanAfterBuild = true;
            boss.seed = seed;
            boss.buildClassifier(data);
            classifiers[currentSeries].add(boss);
            numClassifiers[currentSeries]++;

            if (useWeights){
                if (boss.accuracy == -1) boss.accuracy = individualTrainAcc(boss, data, Double.MIN_VALUE);
                boss.weight = Math.pow(boss.accuracy, 4);
                if (boss.weight == 0) boss.weight = 1;
            }

            if (bayesianParameterSelection) {
                if (boss.accuracy == -1) boss.accuracy = individualTrainAcc(boss, data, Double.MIN_VALUE);
                paramAccuracy[currentSeries].add(boss.accuracy);
            }
            if (trainTimeContract) paramTime[currentSeries].add((double)(System.nanoTime() - indivBuildTime));
            if (memoryContract) paramMemory[currentSeries].add((double)SizeOf.deepSizeOf(boss));

            int prev = currentSeries;
            if (isMultivariate){
                nextSeries();
            }

            if (checkpoint) {
                checkpoint(prev, -1);
            }

            checkContracts();
        }
    }

    private void checkpoint(int seriesNo, int classifierNo){
        if(checkpointPath!=null){
            try{
                File f = new File(serPath);
                if(!f.isDirectory())
                    f.mkdirs();
                //time the checkpoint occured
                checkpointTime = System.nanoTime();

                if (seriesNo >= 0) {
                    if (classifierNo < 0) classifierNo = classifiers[seriesNo].size() - 1;

                    //save the last build individual classifier
                    BOSSIndividual indiv = classifiers[seriesNo].get(classifierNo);

                    FileOutputStream fos = new FileOutputStream(serPath + "BOSSIndividual" + seriesNo + "-" + classifierNo + ".ser");
                    try (ObjectOutputStream out = new ObjectOutputStream(fos)) {
                        out.writeObject(indiv);
                        out.close();
                        fos.close();
                    }
                }

                //dont take into account time spent serialising into build time
                checkpointTimeDiff += System.nanoTime() - checkpointTime;
                checkpointTime = System.nanoTime();

                //save this, classifiers and train data not included
                saveToFile(serPath + "RandomBOSStemp.ser");

                File file = new File(serPath + "RandomBOSStemp.ser");
                File file2 = new File(serPath + "BOSS.ser");
                file2.delete();
                file.renameTo(file2);

                checkpointTimeDiff += System.nanoTime() - checkpointTime;
            }
            catch(Exception e){
                e.printStackTrace();
                System.out.println("Serialisation to "+serPath+" FAILED");
            }
        }
    }

    private void checkpointCleanup(){
        File f = new File(serPath);
        String[] files = f.list();

        for (String file: files){
            File f2 = new File(f.getPath() + "\\" + file);
            boolean b = f2.delete();
        }

        f.delete();
    }

    private String checkpointName(String datasetName){
        String name = datasetName + seed + "BOSS";

        if (trainTimeContract){
            name += ("TTC" + contractTime);
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
        underContractTime = System.nanoTime() - trainResults.getBuildTime() - checkpointTimeDiff < contractTime;
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
                long remainingTime = contractTime - (System.nanoTime() - trainResults.getBuildTime() - checkpointTimeDiff);

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

    private Instances resampleData(Instances series, BOSSIndividual boss){
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

    private double individualTrainAcc(BOSSIndividual boss, Instances series, double lowestAcc) throws Exception {
        int[] indicies;

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
            ex = Executors.newFixedThreadPool(numThreads);
            ArrayList<BOSSIndividual.TrainNearestNeighbourThread> threads = new ArrayList<>(sum(numClassifiers));

            for (int i = 0; i < numInst; ++i) {
                BOSSIndividual.TrainNearestNeighbourThread t = boss.new TrainNearestNeighbourThread(indicies[i]);
                threads.add(t);
                ex.execute(t);
            }

            ex.shutdown();
            while (!ex.isTerminated());

            for (BOSSIndividual.TrainNearestNeighbourThread t: threads){
                if (t.nn == series.get(t.testIndex).classValue()) {
                    ++correct;
                }
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

        double correct = 0;
        for (int i = 0; i < data.numInstances(); ++i) {
            long predTime = System.nanoTime();
            double[] probs = distributionForInstance(i, data.numClasses());
            predTime = System.nanoTime() - predTime;

            double c = 0;
            for (int j = 1; j < probs.length; j++)
                if (probs[j] > probs[(int) c])
                    c = j;

            //No need to do it againclassifyInstance(i, data.numClasses()); //classify series i, while ignoring its corresponding histogram i
            if (c == data.get(i).classValue())
                ++correct;

            this.ensembleCvPreds[i] = c;

            trainResults.addPrediction(data.get(i).classValue(), probs, c, predTime, "");
        }

        double result = correct / data.numInstances();

        return result;
    }

    public double getTrainAcc(){
        if(ensembleCvAcc>=0){
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
        if(this.ensembleCvPreds==null){
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
    private double[] distributionForInstance(int test, int numclasses) throws Exception {
        double[][] classHist = new double[numSeries][numclasses];

        //get sum of all channels, votes from each are weighted the same.
        double sum[] = new double[numSeries];

        for (int n = 0; n < numSeries; n++) {
            for (BOSSIndividual classifier : classifiers[n]) {
                double classification;

                if (classifier.subsampleIndices == null){
                    classification = classifier.classifyInstance(test);
                }
                else if (classifier.subsampleIndices.contains(test)){
                    classification = classifier.classifyInstance(classifier.subsampleIndices.indexOf(test));
                }
                else{
                    classification = classifier.classifyInstance(train.get(test));
                }

                classHist[n][(int) classification] += classifier.weight;
                sum[n] += classifier.weight;
            }
        }

        double[] distributions = new double[numclasses];

        for (int n = 0; n < numSeries; n++){
            if (sum[n] != 0)
                for (int i = 0; i < classHist[n].length; ++i)
                    distributions[i] += (classHist[n][i] / sum[n]) / numSeries;
        }

        return distributions;
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] dist = distributionForInstance(instance);

        double maxFreq=dist[0], maxClass=0;
        for (int i = 1; i < dist.length; ++i) {
            if (dist[i] > maxFreq) {
                maxFreq = dist[i];
                maxClass = i;
            }
            else if (dist[i] == maxFreq){
                if (rand.nextBoolean()){
                    maxClass = i;
                }
            }
        }

        return maxClass;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[][] classHist = new double[numSeries][instance.numClasses()];

        //get sum of all channels, votes from each are weighted the same.
        double sum[] = new double[numSeries];

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
            ex = Executors.newFixedThreadPool(numThreads);
            ArrayList<BOSSIndividual.TestNearestNeighbourThread> threads = new ArrayList<>(sum(numClassifiers));

            for (int n = 0; n < numSeries; n++) {
                for (BOSSIndividual classifier : classifiers[n]) {
                    BOSSIndividual.TestNearestNeighbourThread t = classifier.new TestNearestNeighbourThread(instance, classifier.weight, n);
                    threads.add(t);
                    ex.execute(t);
                }
            }

            ex.shutdown();
            while (!ex.isTerminated());

            for (BOSSIndividual.TestNearestNeighbourThread t: threads){
                classHist[t.series][(int)t.nn] += t.weight;
                sum[t.series] += t.weight;
            }
        }
        else {
            for (int n = 0; n < numSeries; n++) {
                for (BOSSIndividual classifier : classifiers[n]) {
                    double classification = classifier.classifyInstance(series[n]);
                    classHist[n][(int) classification] += classifier.weight;
                    sum[n] += classifier.weight;
                }
            }
        }

        double[] distributions = new double[instance.numClasses()];

        for (int n = 0; n < numSeries; n++){
            if (sum[n] != 0)
                for (int i = 0; i < classHist[n].length; ++i)
                    distributions[i] += (classHist[n][i] / sum[n]) / numSeries;
        }

        return distributions;
    }

    public static void main(String[] args) throws Exception{
        int fold = 0;

        //Minimum working example
        String dataset = "ItalyPowerDemand";
        Instances train = DatasetLoading.loadDataNullable("Z:\\Data\\TSCProblems2018\\"+dataset+"\\"+dataset+"_TRAIN.arff");
        Instances test = DatasetLoading.loadDataNullable("Z:\\Data\\TSCProblems2018\\"+dataset+"\\"+dataset+"_TEST.arff");
        Instances[] data = resampleTrainAndTestInstances(train, test, fold);
        train = data[0];
        test = data[1];

        String dataset2 = "ERing";
        Instances train2 = DatasetLoading.loadDataNullable("Z:\\Data\\MultivariateTSCProblems\\"+dataset2+"\\"+dataset2+"_TRAIN.arff");
        Instances test2 = DatasetLoading.loadDataNullable("Z:\\Data\\MultivariateTSCProblems\\"+dataset2+"\\"+dataset2+"_TEST.arff");
        Instances[] data2 = resampleMultivariateTrainAndTestInstances(train2, test2, fold);
        train2 = data2[0];
        test2 = data2[1];

        RBOSS c;
        double accuracy;

        c = new RBOSS();
        c.useRecommendedSettingsRBOSS();
        c.bayesianParameterSelection = false;
        c.setSeed(fold);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("CVAcc CAWPE BOSS accuracy on " + dataset + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new RBOSS();
        c.useRecommendedSettingsRBOSS();
        c.bayesianParameterSelection = false;
        c.setSeed(fold);
        c.buildClassifier(train2);
        accuracy = ClassifierTools.accuracy(test2, c);

        System.out.println("CVAcc CAWPE BOSS accuracy on " + dataset2 + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new RBOSS();
        c.useRecommendedSettingsRBOSS();
        c.setBayesianParameterSelection(true);
        c.setSeed(fold);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("Bayesian CVAcc CAWPE BOSS accuracy on " + dataset + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new RBOSS();
        c.useRecommendedSettingsRBOSS();
        c.setBayesianParameterSelection(true);
        c.setSeed(fold);
        c.buildClassifier(train2);
        accuracy = ClassifierTools.accuracy(test2, c);

        System.out.println("Bayesian CVAcc CAWPE BOSS accuracy on " + dataset2 + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new RBOSS();
        c.ensembleSize = 250;
        c.setMaxEnsembleSize(50);
        c.setRandomCVAccEnsemble(true);
        c.setSeed(fold);
        c.useFastTrainEstimate = true;
        c.reduceTrainInstances = true;
        c.setMaxEvalPerClass(50);
        c.setMaxTrainInstances(500);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("FastMax CVAcc BOSS accuracy on " + dataset + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new RBOSS();
        c.ensembleSize = 250;
        c.setMaxEnsembleSize(50);
        c.setRandomCVAccEnsemble(true);
        c.setSeed(fold);
        c.useFastTrainEstimate = true;
        c.reduceTrainInstances = true;
        c.setMaxEvalPerClass(50);
        c.setMaxTrainInstances(500);
        c.buildClassifier(train2);
        accuracy = ClassifierTools.accuracy(test2, c);

        System.out.println("FastMax CVAcc BOSS accuracy on " + dataset2 + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new RBOSS();
        c.ensembleSize = 100;
        c.useWeights(true);
        c.setSeed(fold);
        c.setReduceTrainInstances(true);
        c.setTrainProportion(0.7);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("CAWPE Subsample BOSS accuracy on " + dataset + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new RBOSS();
        c.ensembleSize = 100;
        c.useWeights(true);
        c.setSeed(fold);
        c.setReduceTrainInstances(true);
        c.setTrainProportion(0.7);
        c.buildClassifier(train2);
        accuracy = ClassifierTools.accuracy(test2, c);

        System.out.println("CAWPE Subsample BOSS accuracy on " + dataset2 + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new RBOSS();
        c.setTrainTimeLimit(TimeUnit.MINUTES, 1);
        c.setCleanupCheckpointFiles(true);
        c.setSavePath("D:\\");
        c.setSeed(fold);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("Contract 1 Min Checkpoint BOSS accuracy on " + dataset + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new RBOSS();
        c.setTrainTimeLimit(TimeUnit.MINUTES, 1);
        c.setCleanupCheckpointFiles(true);
        c.setSavePath("D:\\");
        c.setSeed(fold);
        c.buildClassifier(train2);
        accuracy = ClassifierTools.accuracy(test2, c);

        System.out.println("Contract 1 Min Checkpoint BOSS accuracy on " + dataset2 + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new RBOSS();
        c.setMemoryLimit(DataUnit.MEGABYTE, 500);
        c.setSeed(fold);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("Contract 500MB BOSS accuracy on " + dataset + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        c = new RBOSS();
        c.setMemoryLimit(DataUnit.MEGABYTE, 500);
        c.setSeed(fold);
        c.buildClassifier(train2);
        accuracy = ClassifierTools.accuracy(test2, c);

        System.out.println("Contract 500MB BOSS accuracy on " + dataset2 + " fold " + fold + " = " + accuracy + " numClassifiers = " + Arrays.toString(c.numClassifiers));

        //Output 22/07/19
        /*
        CVAcc CAWPE BOSS accuracy on ItalyPowerDemand fold 0 = 0.923226433430515 numClassifiers = [50]
        CVAcc CAWPE BOSS accuracy on ERing fold 0 = 0.8851851851851852 numClassifiers = [50, 50, 50, 50]
        Bayesian CVAcc CAWPE BOSS accuracy on ItalyPowerDemand fold 0 = 0.9300291545189504 numClassifiers = [50]
        Bayesian CVAcc CAWPE BOSS accuracy on ERing fold 0 = 0.8851851851851852 numClassifiers = [50, 50, 50, 50]
        FastMax CVAcc BOSS accuracy on ItalyPowerDemand fold 0 = 0.8415937803692906 numClassifiers = [50]
        FastMax CVAcc BOSS accuracy on ERing fold 0 = 0.725925925925926 numClassifiers = [50, 50, 50, 50]
        CAWPE Subsample BOSS accuracy on ItalyPowerDemand fold 0 = 0.9271137026239067 numClassifiers = [80]
        CAWPE Subsample BOSS accuracy on ERing fold 0 = 0.8592592592592593 numClassifiers = [25, 25, 25, 25]
        Contract 1 Min Checkpoint BOSS accuracy on ItalyPowerDemand fold 0 = 0.6958211856171039 numClassifiers = [80]
        Contract 1 Min Checkpoint BOSS accuracy on ERing fold 0 = 0.5259259259259259 numClassifiers = [190, 190, 190, 190]
        Contract 500MB BOSS accuracy on ItalyPowerDemand fold 0 = 0.7103984450923226 numClassifiers = [50]
        Contract 500MB BOSS accuracy on ERing fold 0 = 0.4740740740740741 numClassifiers = [13, 13, 12, 12]
        */
    }
}