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

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import tsml.classifiers.*;
import utilities.ClassifierTools;
import utilities.samplers.RandomIndexSampler;
import utilities.samplers.Sampler;
import weka.classifiers.functions.GaussianProcesses;
import weka.core.*;

import java.io.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static utilities.InstanceTools.resampleTrainAndTestInstances;
import static utilities.Utilities.argMax;
import static utilities.multivariate_tools.MultivariateInstanceTools.*;
import static weka.core.Utils.sum;

/**
 * TDE classifier with parameter search and ensembling for univariate and
 * multivariate time series classification.
 * If parameters are known, use the class IndividualTDE and directly provide them.
 *
 * Has the capability to contract train time and checkpoint.
 *
 * Alphabetsize fixed to four and maximum wordLength of 16.
 *
 * Implementation based on the algorithm described in getTechnicalInformation()
 */
public class TDE extends EnhancedAbstractClassifier implements TrainTimeContractable,
        Checkpointable, TechnicalInformationHandler, MultiThreadable {

    private int parametersConsidered = 250;
    private int parametersConsideredPerChannel = -1;
    private int maxEnsembleSize = 100;

    private boolean histogramIntersection = true;
    private boolean useBigrams = true;

    private double trainProportion = 0.7;

    private boolean bayesianParameterSelection = true;
    private int initialRandomParameters = 50;
    private int[] initialParameterCount;
    private Instances[] parameterPool;
    private Instances[] prevParameters;
    private int[] parametersRemaining;

    private final int[] wordLengths = {16, 14, 12, 10, 8};
    private final int[] alphabetSize = {4};
    private final boolean[] normOptions = {true, false};
    private final Integer[] levels = {1, 2, 3};
    private final boolean[] useIGB = {true, false};

    private double maxWinLenProportion = 1;
    private double maxWinSearchProportion = 0.25;

    private boolean cutoff = false;
    private double cutoffThreshold = 0.7;

    private transient LinkedList<IndividualTDE>[] classifiers;
    private int numSeries;
    private int[] numClassifiers;

    private int currentSeries = 0;
    private boolean isMultivariate = false;
    private Instances seriesHeader;

    private String checkpointPath;
    private boolean checkpoint = false;
    private long checkpointTime = 0;
    private long checkpointTimeDiff = 0;
    private ArrayList<Integer>[] checkpointIDs;
    private boolean internalContractCheckpointHandling = true;
    private boolean cleanupCheckpointFiles = false;
    private boolean loadAndFinish = false;

    private long trainContractTimeNanos = 0;
    private boolean trainTimeContract = false;
    private boolean underContractTime = false;

    private ArrayList<Double>[] paramAccuracy;
    private ArrayList<Double>[] paramTime;

    private transient Instances train;

    private int numThreads = 1;
    private boolean multiThread = false;
    private ExecutorService ex;

    //Classifier build data, stored as field for checkpointing.
    private int[] classifiersBuilt;
    private int[] lowestAccIdx;
    private double[] lowestAcc;
    private double maxAcc;

    protected static final long serialVersionUID = 1L;

    public TDE() {
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        //TODO update
//        TechnicalInformation result;
//        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
//        result.setValue(TechnicalInformation.Field.AUTHOR, "P. Schafer");
//        result.setValue(TechnicalInformation.Field.TITLE, "The BOSS is concerned with time series classification in the presence of noise");
//        result.setValue(TechnicalInformation.Field.JOURNAL, "Data Mining and Knowledge Discovery");
//        result.setValue(TechnicalInformation.Field.VOLUME, "29");
//        result.setValue(TechnicalInformation.Field.NUMBER, "6");
//        result.setValue(TechnicalInformation.Field.PAGES, "1505-1530");
//        result.setValue(TechnicalInformation.Field.YEAR, "2015");
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
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
            sb.append(",seriesNo,").append(n).append(",numClassifiers,").append(numClassifiers[n]);

            for (int i = 0; i < numClassifiers[n]; ++i) {
                IndividualTDE indiv = classifiers[n].get(i);
                sb.append(",windowSize,").append(indiv.getWindowSize()).append(",wordLength,");
                sb.append(indiv.getWordLength()).append(",alphabetSize,").append(indiv.getAlphabetSize());
                sb.append(",norm,").append(indiv.getNorm()).append(",levels,").append(indiv.getLevels());
                sb.append(",IGB,").append(indiv.getIGB());
            }
        }

        return sb.toString();
    }

    //pass in an enum of hour, minute, day, and the amount of them.
    @Override
    public void setTrainTimeLimit(long amount) {
        trainContractTimeNanos = amount;
        trainTimeContract = true;
    }

    @Override
    public void enableMultiThreading(int numThreads) {
        if (numThreads > 1) {
            this.numThreads = numThreads;
            multiThread = true;
        } else {
            this.numThreads = 1;
            multiThread = false;
        }
    }

    //Set the path where checkpointed versions will be stored
    @Override //Checkpointable
    public boolean setCheckpointPath(String path) {
        boolean validPath = Checkpointable.super.createDirectories(path);
        if(validPath){
            checkpointPath = path;
            checkpoint = true;
        }
        return validPath;
    }

    //Define how to copy from a loaded object to this object
    @Override
    public void copyFromSerObject(Object obj) throws Exception {
        if (!(obj instanceof TDE))
            throw new Exception("The SER file is not an instance of TDE");
        TDE saved = ((TDE) obj);
        System.out.println("Loading TDE.ser");

        //copy over variables from serialised object
        parametersConsidered = saved.parametersConsidered;
        parametersConsideredPerChannel = saved.parametersConsideredPerChannel;
        maxEnsembleSize = saved.maxEnsembleSize;
        histogramIntersection = saved.histogramIntersection;
        useBigrams = saved.useBigrams;
        trainProportion = saved.trainProportion;
        bayesianParameterSelection = saved.bayesianParameterSelection;
        initialRandomParameters = saved.initialRandomParameters;
        initialParameterCount = saved.initialParameterCount;
        parameterPool = saved.parameterPool;
        prevParameters = saved.prevParameters;
        parametersRemaining = saved.parametersRemaining;
        //wordLengths = saved.wordLengths;
        //alphabetSize = saved.alphabetSize;
        //normOptions = saved.normOptions;
        //levels = saved.levels;
        //useIGB = saved.useIGB;
        maxWinLenProportion = saved.maxWinLenProportion;
        maxWinSearchProportion = saved.maxWinSearchProportion;
        cutoff = saved.cutoff;
        cutoffThreshold = saved.cutoffThreshold;
        //classifiers = saved.classifier;
        numSeries = saved.numSeries;
        numClassifiers = saved.numClassifiers;
        currentSeries = saved.currentSeries;
        isMultivariate = saved.isMultivariate;
        seriesHeader = saved.seriesHeader;
        //checkpointPath = saved.checkpointPath;
        //checkpoint = saved.checkpoint;
        //checkpointTime = saved.checkpointTime;
        //checkpointTimeDiff = saved.checkpointTimeDiff;
        checkpointIDs = saved.checkpointIDs;
        //internalContractCheckpointHandling = saved.internalContractCheckpointHandling;
        //cleanupCheckpointFiles = saved.cleanupCheckpointFiles;
        //loadAndFinish = saved.loadAndFinish;
        if (internalContractCheckpointHandling) trainContractTimeNanos = saved.trainContractTimeNanos;
        trainTimeContract = saved.trainTimeContract;
        underContractTime = saved.underContractTime;
        paramAccuracy = saved.paramAccuracy;
        paramTime = saved.paramTime;
        //train = saved.train;
        numThreads = saved.numThreads;
        multiThread = saved.multiThread;
        ex = saved.ex;
        classifiersBuilt = saved.classifiersBuilt;
        lowestAccIdx = saved.lowestAccIdx;
        lowestAcc = saved.lowestAcc;
        maxAcc = saved.maxAcc;

        trainResults = saved.trainResults;
        if (!internalContractCheckpointHandling) trainResults.setBuildTime(System.nanoTime());
        seedClassifier = saved.seedClassifier;
        seed = saved.seed;
        rand = saved.rand;
        estimateOwnPerformance = saved.estimateOwnPerformance;
        estimator = saved.estimator;

        //load in each serisalised classifier
        classifiers = new LinkedList[numSeries];
        for (int n = 0; n < numSeries; n++) {
            classifiers[n] = new LinkedList();
            System.out.println(checkpointIDs[n]);
            for (int i = 0; i < maxEnsembleSize; i++) {
                if (!checkpointIDs[n].contains(i)) {
                    System.out.println("Loading IndividualTDE" + n + "-" + i + ".ser");

                    FileInputStream fis = new FileInputStream(checkpointPath + "IndividualTDE" + n + "-" + i + ".ser");
                    try (ObjectInputStream in = new ObjectInputStream(fis)) {
                        Object indv = in.readObject();

                        if (!(indv instanceof IndividualTDE))
                            throw new Exception("The SER file " + n + "-" + i + " is not an instance of IndividualTDE");
                        IndividualTDE ser = ((IndividualTDE) indv);
                        classifiers[n].add(ser);
                    }
                }
            }
        }

        if (internalContractCheckpointHandling) checkpointTimeDiff = saved.checkpointTimeDiff
                + (System.nanoTime() - checkpointTime);
        checkContracts();
    }

    @Override
    public ClassifierResults getTrainResults() {
        return trainResults;
    }

    public void setParametersConsidered(int size) {
        parametersConsidered = size;
    }

    public void setParametersConsideredPerChannel(int size) {
        parametersConsideredPerChannel = size;
    }

    public void setMaxEnsembleSize(int size) {
        maxEnsembleSize = size;
    }

    public void setTrainProportion(double d) {
        trainProportion = d;
    }

    public void setCleanupCheckpointFiles(boolean b) {
        cleanupCheckpointFiles = b;
    }

    public void cleanupCheckpointFiles(boolean b) {
        cleanupCheckpointFiles = b;
    }

    public void loadAndFinish(boolean b) {
        loadAndFinish = b;
    }

    public void setMaxWinLenProportion(double d) {
        maxWinLenProportion = d;
    }

    public void setMaxWinSearchProportion(double d) {
        maxWinSearchProportion = d;
    }

    public void setBayesianParameterSelection(boolean b) {
        bayesianParameterSelection = b;
    }

    public void setUseBigrams(boolean b) { useBigrams = b; }

    public void setCutoff(boolean b) { cutoff = b; }

    public void setCutoffThreshold(double d) { cutoffThreshold = d; }

    @Override
    public void buildClassifier(final Instances data) throws Exception {
        super.buildClassifier(data);
        trainResults.setBuildTime(System.nanoTime());
        // can classifier handle the data?
        getCapabilities().testWithFail(data);

        if (data.checkForAttributeType(Attribute.RELATIONAL)) {
            isMultivariate = true;
        }

        train = data;

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
        checkpointPath = checkpointPath + "/" + checkpointName(data.relationName()) + "/";
        File f = new File(checkpointPath + "TDE.ser");

        //if checkpointing and serialised files exist load said files
        if (checkpoint && f.exists()) {
            if (debug)
                System.out.println("Loading from checkpoint file");
            long time = System.nanoTime();
            loadFromFile(checkpointPath + "TDE.ser");
            if (debug)
                System.out.println("Spent " + (System.nanoTime() - time) + "nanoseconds loading ser files");
        }
        //initialise variables
        else {
            if (data.classIndex() != data.numAttributes() - 1)
                throw new Exception("TDE_BuildClassifier: Class attribute not set as last attribute in dataset");

            //Multivariate
            if (isMultivariate) {
                numSeries = numDimensions(data);
                classifiers = new LinkedList[numSeries];
                for (int n = 0; n < numSeries; n++) {
                    classifiers[n] = new LinkedList<>();
                }

                numClassifiers = new int[numSeries];

                if (parametersConsideredPerChannel > 0) {
                    parametersConsidered = parametersConsideredPerChannel * numSeries;
                }

                if (checkpoint){
                    checkpointIDs = new ArrayList[numSeries];
                    for (int n = 0; n < numSeries; n++) {
                        checkpointIDs[n] = new ArrayList<>();
                        for (int i = 0; i < maxEnsembleSize; i++){
                            checkpointIDs[n].add(i);
                        }
                    }
                }
            }
            //Univariate
            else {
                numSeries = 1;
                classifiers = new LinkedList[1];
                classifiers[0] = new LinkedList<>();
                numClassifiers = new int[1];

                if (checkpoint){
                    checkpointIDs = new ArrayList[1];
                    checkpointIDs[0] = new ArrayList<>();
                    for (int i = 0; i < maxEnsembleSize; i++){
                        checkpointIDs[0].add(i);
                    }
                }
            }

            parameterPool = uniqueParameters(minWindow, maxWindow, winInc);

            classifiersBuilt = new int[numSeries];
            lowestAccIdx = new int[numSeries];
            lowestAcc = new double[numSeries];
            for (int i = 0; i < numSeries; i++) lowestAcc[i] = Double.MAX_VALUE;
            maxAcc = 0;

            if (multiThread) {
                if (numThreads == 1) numThreads = Runtime.getRuntime().availableProcessors();
                if (ex == null) ex = Executors.newFixedThreadPool(numThreads);
            }
        }

        //required to deal with multivariate datasets, each channel is split into its own instances
        Instances[] series;

        //Multivariate
        if (isMultivariate) {
            series = splitMultivariateInstances(data);
            seriesHeader = new Instances(series[0], 0);
        }
        //Univariate
        else {
            series = new Instances[1];
            series[0] = data;
        }

        //Contracting
        if (trainTimeContract) {
            parametersConsidered = 0;
            underContractTime = true;
        }

        //Build ensemble if not set to just load ser files
        if (!(checkpoint && loadAndFinish)) {
            buildTDE(series);
        }

        //end train time in nanoseconds
        trainResults.setBuildTime(System.nanoTime() - trainResults.getBuildTime() - checkpointTimeDiff);

        //Estimate train accuracy
        if (getEstimateOwnPerformance()) {
            long start = System.nanoTime();
            findEnsembleTrainEstimate();
            long end = System.nanoTime();
            trainResults.setErrorEstimateTime(end - start);
            trainResults.setBuildPlusEstimateTime(trainResults.getBuildTime() + trainResults.getErrorEstimateTime());
        }

        trainResults.setParas(getParameters());

        //delete any serialised files and holding folder for checkpointing on completion
        if (checkpoint && cleanupCheckpointFiles) {
            checkpointCleanup();
        }
    }

    private void buildTDE(Instances[] series) throws Exception {
        //build classifiers up to a set size
        while ((underContractTime || sum(classifiersBuilt) < parametersConsidered)
                && sum(parametersRemaining) > 0) {

            long indivBuildTime = System.nanoTime();
            boolean checkpointChange = false;
            double[] parameters = selectParameters();
            if (parameters == null) {
                nextSeries();
                checkContracts();
                continue;
            }

            IndividualTDE indiv = new IndividualTDE((int) parameters[0], (int) parameters[1], (int) parameters[2],
                    parameters[3] == 1, (int) parameters[4], parameters[5] == 1,
                    multiThread, numThreads, ex);
            Instances data = trainProportion < 1 && trainProportion > 0 ? resampleData(series[currentSeries], indiv)
                    : series[currentSeries];
            indiv.buildClassifier(data);
            indiv.setCleanAfterBuild(true);
            indiv.setHistogramIntersection(histogramIntersection);
            indiv.setSeed(seed);

            double accuracy = individualTrainAcc(indiv, data, numClassifiers[currentSeries] < maxEnsembleSize
                    ? Double.MIN_VALUE : lowestAcc[currentSeries]);
            indiv.setAccuracy(accuracy);
            if (accuracy == 0) indiv.setWeight(Double.MIN_VALUE);
            else indiv.setWeight(Math.pow(accuracy, 4));

            if (bayesianParameterSelection) paramAccuracy[currentSeries].add(accuracy);
            if (trainTimeContract) paramTime[currentSeries].add((double) (System.nanoTime() - indivBuildTime));

            if (cutoff && indiv.getAccuracy() > maxAcc) {
                maxAcc = indiv.getAccuracy();
                for (int n = 0; n < numSeries; n++) {
                    //get rid of any extras that dont fall within the new max threshold
                    Iterator<IndividualTDE> it = classifiers[n].iterator();
                    while (it.hasNext()) {
                        IndividualTDE b = it.next();
                        if (b.getAccuracy() < maxAcc * cutoffThreshold) {
                            it.remove();
                            numClassifiers[n]--;

                            if (checkpoint){
                                checkpointIDs[n].add(b.getEnsembleID());
                            }
                        }
                    }
                }
            }

            if (!cutoff || indiv.getAccuracy() >= maxAcc * cutoffThreshold) {
                if (numClassifiers[currentSeries] < maxEnsembleSize) {
                    if (accuracy < lowestAcc[currentSeries]) {
                        lowestAccIdx[currentSeries] = classifiers[currentSeries].size();
                        lowestAcc[currentSeries] = accuracy;
                    }
                    classifiers[currentSeries].add(indiv);
                    numClassifiers[currentSeries]++;

                    if (checkpoint){
                        indiv.setEnsembleID(checkpointIDs[currentSeries].remove(0));
                        checkpointChange = true;
                    }
                } else if (accuracy > lowestAcc[currentSeries]) {
                    double[] newLowestAcc = findMinEnsembleAcc();
                    lowestAccIdx[currentSeries] = (int) newLowestAcc[0];
                    lowestAcc[currentSeries] = newLowestAcc[1];

                    IndividualTDE rm = classifiers[currentSeries].remove(lowestAccIdx[currentSeries]);
                    classifiers[currentSeries].add(lowestAccIdx[currentSeries], indiv);

                    if (checkpoint){
                        indiv.setEnsembleID(rm.getEnsembleID());
                        checkpointChange = true;
                    }
                }
            }

            classifiersBuilt[currentSeries]++;

            int prev = currentSeries;
            if (isMultivariate) {
                nextSeries();
            }

            if (checkpoint) {
                checkpoint(prev, indiv, checkpointChange);
            }

            checkContracts();
        }
    }

    private void checkpoint(int seriesNo, IndividualTDE classifier, boolean saveIndiv) {
        try {
            File f = new File(checkpointPath);
            if (!f.isDirectory())
                f.mkdirs();
            //time the checkpoint occured
            checkpointTime = System.nanoTime();

            if (saveIndiv) {
                FileOutputStream fos = new FileOutputStream(checkpointPath + "IndividualTDE"
                        + seriesNo + "-" + classifier.getEnsembleID() + ".ser");
                try (ObjectOutputStream out = new ObjectOutputStream(fos)) {
                    out.writeObject(classifier);
                    out.close();
                    fos.close();
                }
            }

            //dont take into account time spent serialising into build time
            if (internalContractCheckpointHandling) checkpointTimeDiff += System.nanoTime() - checkpointTime;
            checkpointTime = System.nanoTime();

            //save this, classifiers and train data not included
            saveToFile(checkpointPath + "TDEtemp.ser");

            File file = new File(checkpointPath + "TDEtemp.ser");
            File file2 = new File(checkpointPath + "TDE.ser");
            file2.delete();
            file.renameTo(file2);

            if (internalContractCheckpointHandling) checkpointTimeDiff += System.nanoTime() - checkpointTime;
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Serialisation to " + checkpointPath + " FAILED");
        }
    }

    private void checkpointCleanup() {
        File f = new File(checkpointPath);
        String[] files = f.list();

        for (String file : files) {
            File f2 = new File(f.getPath() + "\\" + file);
            boolean b = f2.delete();
        }

        f.delete();
    }

    private String checkpointName(String datasetName) {
        String name = datasetName + seed + "TDE";

        if (trainTimeContract) {
            name += ("TTC" + trainContractTimeNanos);
        } else if (isMultivariate && parametersConsideredPerChannel > 0) {
            name += ("PC" + (parametersConsideredPerChannel * numSeries));
        } else {
            name += ("S" + parametersConsidered);
        }

        name += ("M" + maxEnsembleSize);

        return name;
    }

    public void checkContracts() {
        underContractTime = System.nanoTime() - trainResults.getBuildTime() - checkpointTimeDiff
                < trainContractTimeNanos;
    }

    //[0] = index, [1] = acc
    private double[] findMinEnsembleAcc() {
        double minAcc = Double.MAX_VALUE;
        int minAccInd = 0;
        for (int i = 0; i < classifiers[currentSeries].size(); ++i) {
            double curacc = classifiers[currentSeries].get(i).getAccuracy();
            if (curacc < minAcc) {
                minAcc = curacc;
                minAccInd = i;
            }
        }

        return new double[]{minAccInd, minAcc};
    }

    private Instances[] uniqueParameters(int minWindow, int maxWindow, int winInc) {
        Instances[] parameterPool = new Instances[numSeries];
        ArrayList<double[]> possibleParameters = new ArrayList();

        for (Boolean normalise : normOptions) {
            for (Integer alphSize : alphabetSize) {
                for (int winSize = minWindow; winSize <= maxWindow; winSize += winInc) {
                    for (Integer wordLen : wordLengths) {
                        for (Integer level : levels) {
                            for (Boolean igb : useIGB) {
                                double[] parameters = {wordLen, alphSize, winSize, normalise ? 1 : 0, level,
                                        igb ? 1 : 0};
                                possibleParameters.add(parameters);
                            }
                        }
                    }
                }
            }
        }

        int numAtts = possibleParameters.get(0).length + 1;
        ArrayList<Attribute> atts = new ArrayList<>(numAtts);
        for (int i = 0; i < numAtts; i++) {
            atts.add(new Attribute("att" + i));
        }

        prevParameters = new Instances[numSeries];
        parametersRemaining = new int[numSeries];
        initialParameterCount = new int[numSeries];

        for (int n = 0; n < numSeries; n++) {
            parameterPool[n] = new Instances("params", atts, possibleParameters.size());
            parameterPool[n].setClassIndex(numAtts - 1);
            prevParameters[n] = new Instances(parameterPool[n], 0);
            prevParameters[n].setClassIndex(numAtts - 1);
            parametersRemaining[n] = possibleParameters.size();

            for (int i = 0; i < possibleParameters.size(); i++) {
                DenseInstance inst = new DenseInstance(1, possibleParameters.get(i));
                inst.insertAttributeAt(numAtts - 1);
                parameterPool[n].add(inst);
            }
        }

        if (bayesianParameterSelection) {
            paramAccuracy = new ArrayList[numSeries];
            for (int i = 0; i < numSeries; i++) {
                paramAccuracy[i] = new ArrayList<>();
            }
        }

        if (trainTimeContract) {
            paramTime = new ArrayList[numSeries];
            for (int i = 0; i < numSeries; i++) {
                paramTime[i] = new ArrayList<>();
            }
        }

        return parameterPool;
    }

    private double[] selectParameters() throws Exception {
        Instance params;

        if (trainTimeContract && System.nanoTime() - trainResults.getBuildTime() - checkpointTimeDiff
                > trainContractTimeNanos / 2) {
            if (prevParameters[currentSeries].size() > 0) {
                for (int i = 0; i < paramTime[currentSeries].size(); i++) {
                    prevParameters[currentSeries].get(i).setClassValue(paramTime[currentSeries].get(i));
                }

                GaussianProcesses gp = new GaussianProcesses();
                gp.buildClassifier(prevParameters[currentSeries]);
                long remainingTime = trainContractTimeNanos - (System.nanoTime() - trainResults.getBuildTime()
                        - checkpointTimeDiff);

                for (int i = 0; i < parameterPool[currentSeries].size(); i++) {
                    double pred = gp.classifyInstance(parameterPool[currentSeries].get(i));
                    if (pred > remainingTime) {
                        parameterPool[currentSeries].remove(i);
                        i--;
                    }
                }
            }
        }

        if (parameterPool[currentSeries].size() == 0) {
            return null;
        }

        if (bayesianParameterSelection) {
            if (initialParameterCount[currentSeries] < initialRandomParameters) {
                initialParameterCount[currentSeries]++;
                params = parameterPool[currentSeries].remove(rand.nextInt(parameterPool[currentSeries].size()));
            } else {
                for (int i = 0; i < paramAccuracy[currentSeries].size(); i++) {
                    prevParameters[currentSeries].get(i).setClassValue(paramAccuracy[currentSeries].get(i));
                }

                GaussianProcesses gp = new GaussianProcesses();
                gp.buildClassifier(prevParameters[currentSeries]);
                int bestIndex = 0;
                double bestAcc = -1;

                for (int i = 0; i < parameterPool[currentSeries].numInstances(); i++) {
                    double pred = gp.classifyInstance(parameterPool[currentSeries].get(i));

                    if (pred > bestAcc) {
                        bestIndex = i;
                        bestAcc = pred;
                    }
                }

                params = parameterPool[currentSeries].remove(bestIndex);
            }
        } else {
            params = parameterPool[currentSeries].remove(rand.nextInt(parameterPool[currentSeries].size()));
        }

        prevParameters[currentSeries].add(params);
        parametersRemaining[currentSeries] = parameterPool[currentSeries].size();
        return params.toDoubleArray();
    }

    private Instances resampleData(Instances series, IndividualTDE indiv) {
        int newSize = (int) (series.numInstances() * trainProportion);
        Instances data = new Instances(series, newSize);

        Sampler sampler = new RandomIndexSampler(rand);
        sampler.setInstances(series);

        ArrayList<Integer> subsampleIndices = new ArrayList<>(newSize);
        for (int i = 0; i < newSize; i++) {
            int n = (Integer) sampler.next();
            data.add(series.get(n));
            subsampleIndices.add(n);
        }
        indiv.setSubsampleIndices(subsampleIndices);

        return data;
    }

    private double individualTrainAcc(IndividualTDE indiv, Instances series, double lowestAcc) throws Exception {
        if (getEstimateOwnPerformance() && estimator == EstimatorMethod.NONE) {
            indiv.setTrainPreds(new ArrayList<>());
        }

        int correct = 0;
        int requiredCorrect = (int) (lowestAcc * series.numInstances());

        if (multiThread) {
            ArrayList<Future<Double>> futures = new ArrayList<>(series.numInstances());

            for (int i = 0; i < series.numInstances(); ++i)
                futures.add(ex.submit(indiv.new TrainNearestNeighbourThread(i)));

            int idx = 0;
            for (Future<Double> f : futures) {
                if (f.get() == series.get(idx).classValue()) {
                    ++correct;
                }
                idx++;

                if (getEstimateOwnPerformance() && estimator == EstimatorMethod.NONE) {
                    indiv.getTrainPreds().add(f.get().intValue());
                }
            }
        } else {
            for (int i = 0; i < series.numInstances(); ++i) {
                if (correct + series.numInstances() - i < requiredCorrect) {
                    return -1;
                }

                double c = indiv.classifyInstance(i); //classify series i, while ignoring its corresponding histogram i
                if (c == series.get(i).classValue()) {
                    ++correct;
                }

                if (getEstimateOwnPerformance() && estimator == EstimatorMethod.NONE) {
                    indiv.getTrainPreds().add((int) c);
                }
            }
        }

        return (double) correct / (double) series.numInstances();
    }

    public void nextSeries() {
        if (currentSeries == numSeries - 1) {
            currentSeries = 0;
        } else {
            currentSeries++;
        }
    }

    private void findEnsembleTrainEstimate() throws Exception {
        if (estimator == EstimatorMethod.OOB && trainProportion < 1){
            for (int i = 0; i < train.numInstances(); ++i) {
                Instance[] series;
                if (isMultivariate) {
                    series = splitMultivariateInstanceWithClassVal(train.get(i));
                }
                else{
                    series = new Instance[]{ train.get(i) };
                }

                double[] probs = new double[train.numClasses()];
                double sum = 0;

                for (int n = 0; n < numSeries; n++) {
                    for (int j = 0; j < numClassifiers[n]; j++) {
                        IndividualTDE classifier = classifiers[n].get(j);

                        if (!classifier.getSubsampleIndices().contains(i)){
                           probs[(int)classifier.classifyInstance(series[n])] += classifier.getWeight();
                           sum += classifier.getWeight();
                        }
                    }
                }

                if (sum != 0) {
                    for (int j = 0; j < probs.length; ++j)
                        probs[j] = (probs[j] / sum);
                }
                else{
                    Arrays.fill(probs, 1.0 / train.numClasses());
                }

                trainResults.addPrediction(train.get(i).classValue(), probs, findIndexOfMax(probs, rand), -1, "");
            }

            trainResults.setClassifierName("TDEOOB");
            trainResults.setErrorEstimateMethod("OOB");
        }
        else {
            double[][] trainDistributions = new double[train.numInstances()][train.numClasses()];
            int[] idxSubsampleCount = new int[train.numInstances()];

            if (estimator == EstimatorMethod.NONE) {
                for (int n = 0; n < numSeries; n++) {
                    for (int i = 0; i < numClassifiers[n]; i++) {
                        ArrayList<Integer> trainIdx = classifiers[n].get(i).getSubsampleIndices();
                        ArrayList<Integer> trainPreds = classifiers[n].get(i).getTrainPreds();
                        double weight = classifiers[n].get(i).getWeight();
                        for (int g = 0; g < trainIdx.size(); g++) {
                            idxSubsampleCount[trainIdx.get(g)] += weight;
                            trainDistributions[trainIdx.get(g)][trainPreds.get(g)] += weight;
                        }
                    }
                }

                for (int i = 0; i < trainDistributions.length; i++) {
                    if (idxSubsampleCount[i] > 0) {
                        for (int n = 0; n < trainDistributions[i].length; n++) {
                            trainDistributions[i][n] /= idxSubsampleCount[i];
                        }
                    }
                }

                trainResults.setClassifierName("TDESubsampleLOO");
                trainResults.setErrorEstimateMethod("SubsampleLOOCV");
            }
            else{
                trainResults.setClassifierName("TDELOO");
                trainResults.setErrorEstimateMethod("LOOCV");
            }

            for (int i = 0; i < train.numInstances(); ++i) {
                double[] probs;

                if (idxSubsampleCount[i] > 0 && estimator == EstimatorMethod.NONE) {
                    probs = trainDistributions[i];
                } else {
                    probs = distributionForInstance(i);
                }

                trainResults.addPrediction(train.get(i).classValue(), probs, findIndexOfMax(probs, rand), -1, "");
            }
        }

        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        trainResults.setDatasetName(train.relationName());
        trainResults.setFoldID(seed);
        trainResults.setSplit("train");
        trainResults.finaliseResults();
    }

    private double[] distributionForInstance(int test) throws Exception {
        double[] probs = new double[train.numClasses()];

        //get sum of all channels, votes from each are weighted the same.
        double sum = 0;

        for (int n = 0; n < numSeries; n++) {
            for (IndividualTDE classifier : classifiers[n]) {
                double classification;

                if (classifier.getSubsampleIndices() == null){
                    classification = classifier.classifyInstance(test);
                }
                else if (classifier.getSubsampleIndices().contains(test)){
                    classification = classifier.classifyInstance(classifier.getSubsampleIndices().indexOf(test));
                }
                else if (estimator == EstimatorMethod.CV) {
                    Instance series = train.get(test);
                    if (isMultivariate){
                        series = splitMultivariateInstanceWithClassVal(series)[n];
                        series.setDataset(seriesHeader);
                    }
                    classification = classifier.classifyInstance(series);
                }
                else{
                    continue;
                }

                probs[(int) classification] += classifier.getWeight();
                sum += classifier.getWeight();
            }
        }

        if (sum != 0) {
            for (int i = 0; i < probs.length; ++i)
                probs[i] = (probs[i] / sum);
        }
        else{
            Arrays.fill(probs, 1.0 / train.numClasses());
        }

        return probs;
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
                for (IndividualTDE classifier : classifiers[n]) {
                    futures[n].add(ex.submit(classifier.new TestNearestNeighbourThread(instance)));
                }
            }

            for (int n = 0; n < numSeries; n++) {
                int idx = 0;
                for (Future<Double> f : futures[n]) {
                    double weight = classifiers[n].get(idx).getWeight();
                    classHist[f.get().intValue()] += weight;
                    sum += weight;
                    idx++;
                }
            }
        }
        else {
            for (int n = 0; n < numSeries; n++) {
                for (IndividualTDE classifier : classifiers[n]) {
                    double classification = classifier.classifyInstance(series[n]);
                    classHist[(int) classification] += classifier.getWeight();
                    sum += classifier.getWeight();
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
        int fold =0;

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

        TDE c;
        double accuracy;

        c = new TDE();
        c.setSeed(fold);
        c.setEstimateOwnPerformance(true);
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("TDE accuracy on " + dataset + " fold " + fold + " = " + accuracy);
        System.out.println("Train accuracy on " + dataset + " fold " + fold + " = " + c.trainResults.getAcc());

        c = new TDE();
        c.setSeed(fold);
        c.setCutoff(true);
        c.setEstimateOwnPerformance(true);
        c.buildClassifier(train2);
        accuracy = ClassifierTools.accuracy(test2, c);

        System.out.println("TDE accuracy on " + dataset2 + " fold " + fold + " = " + accuracy);
        System.out.println("Train accuracy on " + dataset2 + " fold " + fold + " = " + c.trainResults.getAcc());

        c = new TDE();
        c.setSeed(fold);
        c.setTrainTimeLimit(TimeUnit.MINUTES, 1);
        c.setCleanupCheckpointFiles(true);
        c.setCheckpointPath("D:\\");
        c.buildClassifier(train);
        accuracy = ClassifierTools.accuracy(test, c);

        System.out.println("Contract 1 Min Checkpoint TDE accuracy on " + dataset + " fold " + fold + " = "
                + accuracy);
        System.out.println("Build time on " + dataset + " fold " + fold + " = " +
                TimeUnit.SECONDS.convert(c.trainResults.getBuildTime(), TimeUnit.NANOSECONDS) + " seconds");


        c = new TDE();
        c.setSeed(fold);
        c.setTrainTimeLimit(TimeUnit.MINUTES, 1);
        c.setCleanupCheckpointFiles(true);
        c.setCheckpointPath("D:\\");
        c.setCutoff(true);
        c.buildClassifier(train2);
        accuracy = ClassifierTools.accuracy(test2, c);

        System.out.println("Contract 1 Min Checkpoint TDE accuracy on " + dataset2 + " fold " + fold + " = "
                + accuracy);
        System.out.println("Build time on " + dataset2 + " fold " + fold + " = " +
                TimeUnit.SECONDS.convert(c.trainResults.getBuildTime(), TimeUnit.NANOSECONDS) + " seconds");

        //Output 24/06/20
        /*
            TDE accuracy on ItalyPowerDemand fold 0 = 0.9484936831875608
            Train accuracy on ItalyPowerDemand fold 0 = 0.9552238805970149
            TDE accuracy on ERing fold 0 = 0.9481481481481482
            Train accuracy on ERing fold 0 = 0.9
            Contract 1 Min Checkpoint TDE accuracy on ItalyPowerDemand fold 0 = 0.9523809523809523
            Build time on ItalyPowerDemand fold 0 = 7 seconds
            Contract 1 Min Checkpoint TDE accuracy on ERing fold 0 = 0.9629629629629629
            Build time on ERing fold 0 = 60 seconds
        */
    }
}
