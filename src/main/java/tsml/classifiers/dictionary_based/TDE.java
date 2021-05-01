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

package tsml.classifiers.dictionary_based;

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import fileIO.OutFile;
import tsml.classifiers.*;
import tsml.classifiers.dictionary_based.bitword.BitWord;
import tsml.classifiers.dictionary_based.bitword.BitWordInt;
import tsml.classifiers.dictionary_based.bitword.BitWordLong;
import tsml.data_containers.TSCapabilities;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import utilities.ClassifierTools;
import utilities.generic_storage.SerialisableComparablePair;
import weka.classifiers.functions.GaussianProcesses;
import weka.core.*;

import java.io.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static utilities.Utilities.argMax;

/**
 * TDE classifier with parameter search and ensembling for univariate and
 * multivariate time series classification.
 * If parameters are known, use the class IndividualTDE and directly provide them.
 * <p>
 * Has the capability to contract train time and checkpoint.
 * <p>
 * Alphabetsize fixed to four and maximum wordLength of 16.
 * <p>
 * Implementation based on the algorithm described in getTechnicalInformation()
 *
 * @author Matthew Middlehurst
 */
public class TDE extends EnhancedAbstractClassifier implements TrainTimeContractable,
        Checkpointable, TechnicalInformationHandler, MultiThreadable, Visualisable, Interpretable {

    /**
     * Paper defining TDE.
     *
     * @return TechnicalInformation for TDE
     */
    @Override //TechnicalInformationHandler
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;
        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "M. Middlehurst, J. Large, G. Cawley and A. Bagnall");
        result.setValue(TechnicalInformation.Field.TITLE, "The Temporal Dictionary Ensemble (TDE) Classifier for " +
                "Time Series Classification");
        result.setValue(TechnicalInformation.Field.JOURNAL, "The European Conference on Machine Learning and " +
                "Principles and Practice of Knowledge Discovery in Databases");
        result.setValue(TechnicalInformation.Field.YEAR, "2020");
        return result;
    }

    private int parametersConsidered = 250;
    private int maxEnsembleSize = 50;

    private boolean histogramIntersection = true;
    private Boolean useBigrams; //defaults to true if univariate, false if multivariate
    private boolean useFeatureSelection = false;

    private double trainProportion = 0.7;

    private double dimensionCutoffThreshold = 0.85;
    private int maxNoDimensions = 20;

    private boolean bayesianParameterSelection = true;
    private int initialRandomParameters = 50;
    private int initialParameterCount;
    private Instances parameterPool;
    private Instances prevParameters;
    private int parametersRemaining;

    private final int[] wordLengths = {16, 14, 12, 10, 8};
    private final int[] alphabetSize = {4};
    private final boolean[] normOptions = {true, false};
    private final Integer[] levels = {1, 2, 3};
    private final boolean[] useIGB = {true, false};

    private double maxWinLenProportion = 1;
    private double maxWinSearchProportion = 0.25;

    private boolean cutoff = false;
    private double cutoffThreshold = 0.7;

    private transient LinkedList<IndividualTDE> classifiers;

    private String checkpointPath;
    private boolean checkpoint = false;
    private long checkpointTime = 0;
    private long checkpointTimeDiff = 0;
    private ArrayList<Integer> checkpointIDs;
    private boolean internalContractCheckpointHandling = true;
    private boolean cleanupCheckpointFiles = false;
    private boolean loadAndFinish = false;

    private long trainContractTimeNanos = 0;
    private boolean trainTimeContract = false;
    private boolean underContractTime = true;

    private ArrayList<Double> paramAccuracy;
    private ArrayList<Double> paramTime;

    private transient TimeSeriesInstances train;

    private int numThreads = 1;
    private boolean multiThread = false;
    private ExecutorService ex;

    //Classifier build data, stored as field for checkpointing.
    private int classifiersBuilt;
    private int lowestAccIdx;
    private double lowestAcc;
    private double maxAcc;

    private String visSavePath;
    private String interpSavePath;
    private ArrayList<Integer> interpData;
    private ArrayList<Integer> interpPreds;
    private int interpCount = 0;
    private double[] interpSeries;
    private int interpPred;

    protected static final long serialVersionUID = 1L;

    /**
     * Default constructor for TDE. Can estimate own performance.
     */
    public TDE() {
        super(CAN_ESTIMATE_OWN_PERFORMANCE);
    }

    /**
     * Set the amount of parameter sets considered for the ensemble.
     *
     * @param size number of parameters considered
     */
    public void setParametersConsidered(int size) {
        parametersConsidered = size;
    }

    /**
     * Max number of classifiers for the ensemble.
     *
     * @param size max ensemble size
     */
    public void setMaxEnsembleSize(int size) {
        maxEnsembleSize = size;
    }

    /**
     * Proportion of train set to be randomly subsampled for each classifier.
     *
     * @param d train subsample proportion
     */
    public void setTrainProportion(double d) {
        trainProportion = d;
    }

    /**
     * Dimension accuracy cutoff threshold for multivariate time series.
     *
     * @param d dimension cutoff threshold
     */
    public void setDimensionCutoffThreshold(double d) {
        dimensionCutoffThreshold = d;
    }

    /**
     * Maximum number of dimensions kept for multivariate time series.
     *
     * @param d Max number of dimensions
     */
    public void setMaxNoDimensions(int d) {
        maxNoDimensions = d;
    }

    /**
     * Whether to delete checkpoint files after building has finished.
     *
     * @param b clean up checkpoint files
     */
    public void setCleanupCheckpointFiles(boolean b) {
        cleanupCheckpointFiles = b;
    }

    /**
     * Whether to load checkpoint files and finish building from the loaded state.
     *
     * @param b load ser files and finish building
     */
    public void loadAndFinish(boolean b) {
        loadAndFinish = b;
    }

    /**
     * Max window length as proportion of the series length.
     *
     * @param d max window length proportion
     */
    public void setMaxWinLenProportion(double d) {
        maxWinLenProportion = d;
    }

    /**
     * Max number of window lengths to search through as proportion of the series length.
     *
     * @param d window length search proportion
     */
    public void setMaxWinSearchProportion(double d) {
        maxWinSearchProportion = d;
    }

    /**
     * Whether to use GP parameter selection for IndividualBOSS classifiers.
     *
     * @param b use GP parameter selection
     */
    public void setBayesianParameterSelection(boolean b) {
        bayesianParameterSelection = b;
    }

    /**
     * Wether to use bigrams in IndividualBOSS classifiers.
     *
     * @param b use bigrams
     */
    public void setUseBigrams(Boolean b) {
        useBigrams = b;
    }

    /**
     * Wether to use feature selection in IndividualBOSS classifiers.
     *
     * @param b use feature selection
     */
    public void setUseFeatureSelection(boolean b) {
        useFeatureSelection = b;
    }

    /**
     * Whether to remove ensemble members below a proportion of the highest accuracy.
     *
     * @param b use ensemble cutoff
     */
    public void setCutoff(boolean b) {
        cutoff = b;
    }

    /**
     * Ensemble accuracy cutoff as proportion of highest ensemble memeber accuracy.
     *
     * @param d cutoff proportion
     */
    public void setCutoffThreshold(double d) {
        cutoffThreshold = d;
    }

    /**
     * Outputs TDE and IndivdiualTDE parameters as a String.
     *
     * @return String written to results files
     */
    @Override //SaveParameterInfo
    public String getParameters() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.getParameters());
        sb.append(",numClassifiers,").append(classifiers.size()).append(",contractTime,")
                .append(trainContractTimeNanos);

        for (int i = 0; i < classifiers.size(); ++i) {
            IndividualTDE indiv = classifiers.get(i);
            sb.append(",windowSize,").append(indiv.getWindowSize()).append(",wordLength,");
            sb.append(indiv.getWordLength()).append(",alphabetSize,").append(indiv.getAlphabetSize());
            sb.append(",norm,").append(indiv.getNorm()).append(",levels,").append(indiv.getLevels());
            sb.append(",IGB,").append(indiv.getIGB());
        }

        return sb.toString();
    }

    /**
     * Returns the capabilities for TDE. These are that the
     * data must be numeric or relational, with no missing and a nominal class
     *
     * @return the capabilities of TDE
     */
    @Override //AbstractClassifier
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

    /**
     * Returns the time series capabilities for TDE. These are that the
     * data must be equal length, with no missing values
     *
     * @return the time series capabilities of TDE
     */
    public TSCapabilities getTSCapabilities() {
        TSCapabilities capabilities = new TSCapabilities();
        capabilities.enable(TSCapabilities.EQUAL_LENGTH)
                .enable(TSCapabilities.MULTI_OR_UNIVARIATE)
                .enable(TSCapabilities.NO_MISSING_VALUES);
        return capabilities;
    }

    /**
     * Build the TDE classifier.
     *
     * @param data TimeSeriesInstances object
     * @throws Exception unable to train model
     */
    @Override //TSClassifier
    public void buildClassifier(final TimeSeriesInstances data) throws Exception {
        trainResults = new ClassifierResults();
        rand.setSeed(seed);
        numClasses = data.numClasses();
        trainResults.setClassifierName(getClassifierName());
        trainResults.setBuildTime(System.nanoTime());
        // can classifier handle the data?
        getTSCapabilities().test(data);

        train = data;

        //Window length settings
        int minWindow = 10;
        int maxWindow = (int) (data.getMaxLength() * maxWinLenProportion);
        if (maxWindow < minWindow) minWindow = maxWindow / 2;
        //whats the max number of window sizes that should be searched through
        double maxWindowSearches = data.getMaxLength() * maxWinSearchProportion;
        int winInc = (int) ((maxWindow - minWindow) / maxWindowSearches);
        if (winInc < 1) winInc = 1;

        //path checkpoint files will be saved to
        checkpointPath = checkpointPath + "/" + checkpointName(data.getProblemName()) + "/";
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
            classifiers = new LinkedList<>();

            if (checkpoint) {
                checkpointIDs = new ArrayList<>();
                for (int i = 0; i < maxEnsembleSize; i++) {
                    checkpointIDs.add(i);
                }
            }

            useBigrams = !data.isMultivariate() && useBigrams == null;

            parameterPool = uniqueParameters(minWindow, maxWindow, winInc);

            classifiersBuilt = 0;
            lowestAccIdx = 0;
            lowestAcc = Double.MAX_VALUE;
            maxAcc = 0;
        }

        if (multiThread) {
            ex = Executors.newFixedThreadPool(numThreads);
        }

        //Contracting
        if (trainTimeContract) {
            parametersConsidered = Integer.MAX_VALUE;
        }

        //Build ensemble if not set to just load ser files
        if (!(checkpoint && loadAndFinish)) {
            buildTDE(data);
        }

        if (checkpoint) {
            checkpoint(null, false);
        }

        //end train time in nanoseconds
        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        trainResults.setBuildTime(System.nanoTime() - trainResults.getBuildTime() - checkpointTimeDiff);

        //Estimate train accuracy
        if (getEstimateOwnPerformance()) {
            long start = System.nanoTime();
            findEnsembleTrainEstimate();
            long end = System.nanoTime();
            trainResults.setErrorEstimateTime(end - start);
        }
        trainResults.setBuildPlusEstimateTime(trainResults.getBuildTime() + trainResults.getErrorEstimateTime());
        trainResults.setParas(getParameters());

        //delete any serialised files and holding folder for checkpointing on completion
        if (checkpoint && cleanupCheckpointFiles) {
            checkpointCleanup();
        }
        printLineDebug("*************** Finished TDE Build with "+classifiersBuilt+" classifiers built in train time " +
                (trainResults.getBuildTime()/1000000000/60/60.0) + " hours, Train+Estimate time = "+(trainResults.getBuildPlusEstimateTime()/1000000000/60/60.0)+" hours ***************");

    }

    /**
     * Build the TDE classifier.
     *
     * @param data weka Instances object
     * @throws Exception unable to train model
     */
    @Override //AbstractClassifier
    public void buildClassifier(final Instances data) throws Exception {
        buildClassifier(Converter.fromArff(data));
    }

    /**
     * Builds parametersConsidered IndividualTDE classifiers or contiues building until the train time contract
     * finishes.
     * Randomly subsamples the train set for each classifier and selects parameters using a GP model.
     * Keeps the top maxEnsembleSize classifiers with the highest accuracy found using LOOCV for the ensemble.
     *
     * @param series TimeSeriesInstances object
     * @throws Exception unable to train model
     */
    private void buildTDE(TimeSeriesInstances series) throws Exception {
        //build classifiers up to a set size
        while (underContractTime && classifiersBuilt < parametersConsidered && parametersRemaining > 0) {
            long indivBuildTime = System.nanoTime();
            boolean checkpointChange = false;
            double[] parameters = selectParameters();
            if (parameters == null) break;

            IndividualTDE indiv;
            if (series.isMultivariate()) {
                indiv = new MultivariateIndividualTDE((int) parameters[0], (int) parameters[1], (int) parameters[2],
                        parameters[3] == 1, (int) parameters[4], parameters[5] == 1,
                        multiThread, numThreads, ex);
                ((MultivariateIndividualTDE) indiv).setDimensionCutoffThreshold(dimensionCutoffThreshold);
                ((MultivariateIndividualTDE) indiv).setMaxNoDimensions(maxNoDimensions);
            } else {
                indiv = new IndividualTDE((int) parameters[0], (int) parameters[1], (int) parameters[2],
                        parameters[3] == 1, (int) parameters[4], parameters[5] == 1,
                        multiThread, numThreads, ex);
            }
            indiv.setCleanAfterBuild(true);
            indiv.setHistogramIntersection(histogramIntersection);
            indiv.setUseBigrams(useBigrams);
            indiv.setUseFeatureSelection(useFeatureSelection);
            indiv.setSeed(seed);

            TimeSeriesInstances data = trainProportion < 1 && trainProportion > 0 ? subsampleData(series, indiv)
                    : series;
            indiv.buildClassifier(data);

            double accuracy = individualTrainAcc(indiv, data, classifiers.size() < maxEnsembleSize
                    ? -99999999 : lowestAcc);
            indiv.setAccuracy(accuracy);
            if (accuracy == 0) indiv.setWeight(Double.MIN_VALUE);
            else indiv.setWeight(Math.pow(accuracy, 4));

            if (bayesianParameterSelection) paramAccuracy.add(accuracy);
            if (trainTimeContract) paramTime.add((double) (System.nanoTime() - indivBuildTime));

            if (cutoff && indiv.getAccuracy() > maxAcc) {
                maxAcc = indiv.getAccuracy();
                //get rid of any extras that dont fall within the new max threshold
                Iterator<IndividualTDE> it = classifiers.iterator();
                while (it.hasNext()) {
                    IndividualTDE b = it.next();
                    if (b.getAccuracy() < maxAcc * cutoffThreshold) {
                        it.remove();

                        if (checkpoint) {
                            checkpointIDs.add(b.getEnsembleID());
                        }
                    }
                }
            }

            if (!cutoff || indiv.getAccuracy() >= maxAcc * cutoffThreshold) {
                if (classifiers.size() < maxEnsembleSize) {
                    if (accuracy < lowestAcc) {
                        lowestAccIdx = classifiers.size();
                        lowestAcc = accuracy;
                    }
                    classifiers.add(indiv);

                    if (checkpoint) {
                        indiv.setEnsembleID(checkpointIDs.remove(0));
                        checkpointChange = true;
                    }
                } else if (accuracy > lowestAcc) {
                    double[] newLowestAcc = findMinEnsembleAcc();
                    lowestAccIdx = (int) newLowestAcc[0];
                    lowestAcc = newLowestAcc[1];

                    IndividualTDE rm = classifiers.remove(lowestAccIdx);
                    classifiers.add(lowestAccIdx, indiv);

                    if (checkpoint) {
                        indiv.setEnsembleID(rm.getEnsembleID());
                        checkpointChange = true;
                    }
                }
            }

            classifiersBuilt++;

            if (checkpoint) {
                checkpoint(indiv, checkpointChange);
            }

            underContractTime = withinTrainContract(trainResults.getBuildTime());
        }
    }

    /**
     * Saved the current state of the classifier to file, if the ensemble has changed save the new classifier as well
     * as meta info.
     *
     * @param classifier last build IndividualTDE classifier
     * @param saveIndiv  whether to save the new IndividualClassifier
     */
    private void checkpoint(IndividualTDE classifier, boolean saveIndiv) {
        try {
            File f = new File(checkpointPath);
            if (!f.isDirectory())
                f.mkdirs();
            //time the checkpoint occured
            checkpointTime = System.nanoTime();

            if (saveIndiv) {
                FileOutputStream fos = new FileOutputStream(checkpointPath + "IndividualTDE-" +
                        classifier.getEnsembleID() + ".ser");
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

    /**
     * Remove any checkpoint files used.
     */
    private void checkpointCleanup() {
        File f = new File(checkpointPath);
        String[] files = f.list();

        for (String file : files) {
            File f2 = new File(f.getPath() + "\\" + file);
            boolean b = f2.delete();
        }

        f.delete();
    }

    /**
     * Checkpoint classifier name differing by dataset and parameters used to prevent overlap.
     *
     * @param datasetName name of the dataset
     * @return checkpoint file name
     */
    private String checkpointName(String datasetName) {
        String name = datasetName + seed + "TDE";

        if (trainTimeContract) {
            name += ("TTC" + trainContractTimeNanos);
        } else {
            name += ("S" + parametersConsidered);
        }

        name += ("M" + maxEnsembleSize);

        return name;
    }

    /**
     * Finds the index and accuracy of the ensemble member with least accuracy
     *
     * @return index and accuracy of ensemble member with least accuracy, [0] = index, [1] = acc
     */
    private double[] findMinEnsembleAcc() {
        double minAcc = Double.MAX_VALUE;
        int minAccInd = 0;
        for (int i = 0; i < classifiers.size(); ++i) {
            double curacc = classifiers.get(i).getAccuracy();
            if (curacc < minAcc) {
                minAcc = curacc;
                minAccInd = i;
            }
        }

        return new double[]{minAccInd, minAcc};
    }

    /**
     * Finds and returns all possible parameter sets for IndividualTDE classifiers.
     *
     * @param minWindow min window size
     * @param maxWindow max windoew size
     * @param winInc    window size increment
     * @return possible parameters as an Instances object
     */
    private Instances uniqueParameters(int minWindow, int maxWindow, int winInc) {
        ArrayList<double[]> possibleParameters = new ArrayList<>();

        for (Boolean normalise : normOptions) {
            for (Integer alphSize : alphabetSize) {
                for (int winSize = minWindow; winSize <= maxWindow; winSize += winInc) {
                    for (Integer wordLen : wordLengths) {
                        for (Integer level : levels) {
                            for (Boolean igb : useIGB) {
                                double[] parameters = {wordLen, alphSize, winSize, normalise ? 1 : 0, level,
                                        igb ? 1 : 0, -1};
                                possibleParameters.add(parameters);
                            }
                        }
                    }
                }
            }
        }

        int numAtts = possibleParameters.get(0).length;
        ArrayList<Attribute> atts = new ArrayList<>(numAtts);
        for (int i = 0; i < numAtts; i++) {
            atts.add(new Attribute("att" + i));
        }

        Instances parameterPool = new Instances("params", atts, possibleParameters.size());
        parameterPool.setClassIndex(numAtts - 1);
        prevParameters = new Instances(parameterPool, 0);
        prevParameters.setClassIndex(numAtts - 1);
        parametersRemaining = possibleParameters.size();

        for (double[] possibleParameter : possibleParameters) {
            DenseInstance inst = new DenseInstance(1, possibleParameter);
            parameterPool.add(inst);
        }

        if (bayesianParameterSelection) {
            paramAccuracy = new ArrayList<>();
        }

        if (trainTimeContract) {
            paramTime = new ArrayList<>();
        }

        return parameterPool;
    }

    /**
     * Finds a parameter set for an IndividualTDE classifier using GP parameter selection.
     * If contracting remove any parameter sets estimated to go past the contract time after 90% of the contract has
     * elapsed.
     *
     * @return IndividualTDE parameters
     * @throws Exception unable to select parameters
     */
    private double[] selectParameters() throws Exception {
        Instance params;

        if (trainTimeContract && System.nanoTime() - trainResults.getBuildTime() - checkpointTimeDiff
                > trainContractTimeNanos / 10 * 9) {
            if (prevParameters.size() > 0) {
                for (int i = 0; i < paramTime.size(); i++) {
                    prevParameters.get(i).setClassValue(paramTime.get(i));
                }

                GaussianProcesses gp = new GaussianProcesses();
                gp.buildClassifier(prevParameters);
                long remainingTime = trainContractTimeNanos - (System.nanoTime() - trainResults.getBuildTime()
                        - checkpointTimeDiff);

                for (int i = 0; i < parameterPool.size(); i++) {
                    double pred = gp.classifyInstance(parameterPool.get(i));
                    if (pred > remainingTime) {
                        parameterPool.remove(i);
                        i--;
                    }
                }

                if (parameterPool.size() == 0) return null;
            }
        }

        if (bayesianParameterSelection) {
            if (initialParameterCount < initialRandomParameters) {
                initialParameterCount++;
                params = parameterPool.remove(rand.nextInt(parameterPool.size()));
            } else {
                for (int i = 0; i < paramAccuracy.size(); i++) {
                    prevParameters.get(i).setClassValue(paramAccuracy.get(i));
                }

                GaussianProcesses gp = new GaussianProcesses();
                gp.buildClassifier(prevParameters);
                int bestIndex = 0;
                double bestAcc = -1;

                for (int i = 0; i < parameterPool.numInstances(); i++) {
                    double pred = gp.classifyInstance(parameterPool.get(i));

                    if (pred > bestAcc) {
                        bestIndex = i;
                        bestAcc = pred;
                    }
                }

                params = parameterPool.remove(bestIndex);
            }
        } else {
            params = parameterPool.remove(rand.nextInt(parameterPool.size()));
        }

        prevParameters.add(params);
        parametersRemaining = parameterPool.size();
        return params.toDoubleArray();
    }

    /**
     * Randomly subsample the train set.
     *
     * @param series data to be subsampled
     * @param indiv  classifier being subsampled for
     * @return subsampled data
     */
    private TimeSeriesInstances subsampleData(TimeSeriesInstances series, IndividualTDE indiv) {
        int newSize = (int) (series.numInstances() * trainProportion);
        ArrayList<TimeSeriesInstance> data = new ArrayList<>();

        ArrayList<Integer> indices = new ArrayList<>(series.numInstances());
        for (int n = 0; n < series.numInstances(); n++) {
            indices.add(n);
        }

        ArrayList<Integer> subsampleIndices = new ArrayList<>(series.numInstances());
        while (subsampleIndices.size() < newSize) {
            subsampleIndices.add(indices.remove(rand.nextInt(indices.size())));
        }

        for (int i = 0; i < newSize; i++) {
            data.add(series.get(subsampleIndices.get(i)));
        }
        indiv.setSubsampleIndices(subsampleIndices);

        return new TimeSeriesInstances(data, series.getClassLabels());
    }

    /**
     * Estimate the accruacy of an IndividualTDE using LOO CV for the subsampled data. Early exit if it is impossible
     * to meet the required accuracy.
     *
     * @param indiv     classifier to evaluate
     * @param series    subsampled data
     * @param lowestAcc lowest accuracy in the ensemble
     * @return estimated accuracy
     * @throws Exception unable to estimate accuracy
     */
    private double individualTrainAcc(IndividualTDE indiv, TimeSeriesInstances series, double lowestAcc)
            throws Exception {
        if (getEstimateOwnPerformance() && trainEstimateMethod == TrainEstimateMethod.NONE) {
            indiv.setTrainPreds(new ArrayList<>());
        }

        int correct = 0;
        int requiredCorrect = (int) (lowestAcc * series.numInstances());

        if (multiThread) {
            ArrayList<Future<Double>> futures = new ArrayList<>(series.numInstances());

            for (int i = 0; i < series.numInstances(); ++i) {
                if (series.isMultivariate())
                    futures.add(ex.submit(((MultivariateIndividualTDE) indiv).new TrainNearestNeighbourThread(i)));
                else
                    futures.add(ex.submit(indiv.new TrainNearestNeighbourThread(i)));
            }

            int idx = 0;
            for (Future<Double> f : futures) {
                if (f.get() == series.get(idx).getLabelIndex()) {
                    ++correct;
                }
                idx++;

                if (getEstimateOwnPerformance() && trainEstimateMethod == TrainEstimateMethod.NONE) {
                    indiv.getTrainPreds().add(f.get().intValue());
                }
            }
        } else {
            for (int i = 0; i < series.numInstances(); ++i) {
                if (correct + series.numInstances() - i < requiredCorrect) {
                    return -1;
                }

                double c = indiv.classifyInstance(i); //classify series i, while ignoring its corresponding histogram i
                if (c == series.get(i).getLabelIndex()) {
                    ++correct;
                }

                if (getEstimateOwnPerformance() && trainEstimateMethod == TrainEstimateMethod.NONE) {
                    indiv.getTrainPreds().add((int) c);
                }
            }
        }

        return (double) correct / (double) series.numInstances();
    }

    /**
     * Estimate accuracy stage: Three scenarios
     * 1. Subsampled LOO CV using the transformed instances for each classifier.
     * 2. Full LOO CV.
     * 3. Using the out of bag instances for each classifier.
     *
     * @throws Exception unable to obtain estimate
     */
    private void findEnsembleTrainEstimate() throws Exception {
        if (trainEstimateMethod == TrainEstimateMethod.OOB && trainProportion < 1) {
            for (int i = 0; i < train.numInstances(); ++i) {
                double[] probs = new double[train.numClasses()];
                double sum = 0;

                for (IndividualTDE classifier : classifiers) {
                    if (!classifier.getSubsampleIndices().contains(i)) {
                        probs[(int) classifier.classifyInstance(train.get(i))] += classifier.getWeight();
                        sum += classifier.getWeight();
                    }
                }

                if (sum != 0) {
                    for (int j = 0; j < probs.length; ++j)
                        probs[j] = (probs[j] / sum);
                } else {
                    Arrays.fill(probs, 1.0 / train.numClasses());
                }

                trainResults.addPrediction(train.get(i).getLabelIndex(), probs, findIndexOfMax(probs, rand),
                        -1, "");
            }

            trainResults.setClassifierName("TDEOOB");
            trainResults.setErrorEstimateMethod("OOB");
        } else {
            double[][] trainDistributions = new double[train.numInstances()][train.numClasses()];
            double[] idxSubsampleCount = new double[train.numInstances()];

            if ((trainEstimateMethod == TrainEstimateMethod.NONE || trainEstimateMethod == TrainEstimateMethod.TRAIN)
                    && trainProportion < 1) {
                for (IndividualTDE classifier : classifiers) {
                    ArrayList<Integer> trainIdx = classifier.getSubsampleIndices();
                    ArrayList<Integer> trainPreds = classifier.getTrainPreds();
                    double weight = classifier.getWeight();
                    for (int g = 0; g < trainIdx.size(); g++) {
                        idxSubsampleCount[trainIdx.get(g)] += weight;
                        trainDistributions[trainIdx.get(g)][trainPreds.get(g)] += weight;
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
            } else {
                trainResults.setClassifierName("TDELOO");
                trainResults.setErrorEstimateMethod("LOOCV");
            }

            for (int i = 0; i < train.numInstances(); ++i) {
                double[] probs;

                if (idxSubsampleCount[i] > 0 && (trainEstimateMethod == TrainEstimateMethod.NONE
                        || trainEstimateMethod == TrainEstimateMethod.TRAIN)) {
                    probs = trainDistributions[i];
                } else {
                    probs = distributionForInstance(i);
                }

                trainResults.addPrediction(train.get(i).getLabelIndex(), probs, findIndexOfMax(probs, rand),
                        -1, "");
            }
        }

        trainResults.setDatasetName(train.getProblemName());
        trainResults.setFoldID(seed);
        trainResults.setSplit("train");
        trainResults.finaliseResults();
    }

    /**
     * Find class probabilities of a training instance using the trained model, removing said instance if present.
     *
     * @param test train instance index
     * @return array of doubles: probability of each class
     * @throws Exception failure to classify
     */
    private double[] distributionForInstance(int test) throws Exception {
        double[] probs = new double[train.numClasses()];

        //get sum of all channels, votes from each are weighted the same.
        double sum = 0;

        for (IndividualTDE classifier : classifiers) {
            double classification;

            if (classifier.getSubsampleIndices() == null) {
                classification = classifier.classifyInstance(test);
            } else if (classifier.getSubsampleIndices().contains(test)) {
                classification = classifier.classifyInstance(classifier.getSubsampleIndices().indexOf(test));
            } else if (trainEstimateMethod == TrainEstimateMethod.CV) {
                TimeSeriesInstance series = train.get(test);
                classification = classifier.classifyInstance(series);
            } else {
                continue;
            }

            probs[(int) classification] += classifier.getWeight();
            sum += classifier.getWeight();
        }

        if (sum != 0) {
            for (int i = 0; i < probs.length; ++i)
                probs[i] = (probs[i] / sum);
        } else {
            Arrays.fill(probs, 1.0 / train.numClasses());
        }

        return probs;
    }

    /**
     * Find class probabilities of an instance using the trained model.
     *
     * @param instance TimeSeriesInstance object
     * @return array of doubles: probability of each class
     * @throws Exception failure to classify
     */
    @Override //TSClassifier
    public double[] distributionForInstance(TimeSeriesInstance instance) throws Exception {
        double[] classHist = new double[numClasses];

        //get sum of all channels, votes from each are weighted the same.
        double sum = 0;

        if (interpSavePath != null) {
            interpData = new ArrayList<>();
            interpPreds = new ArrayList<>();
        }

        if (multiThread) {
            ArrayList<Future<Double>> futures = new ArrayList<>(classifiers.size());

            for (IndividualTDE classifier : classifiers) {
                if (train.isMultivariate())
                    futures.add(ex.submit(((MultivariateIndividualTDE) classifier)
                            .new TestNearestNeighbourThread(instance)));
                else
                    futures.add(ex.submit(classifier.new TestNearestNeighbourThread(instance)));
            }

            int idx = 0;
            for (Future<Double> f : futures) {
                double weight = classifiers.get(idx).getWeight();
                classHist[f.get().intValue()] += weight;
                sum += weight;
                idx++;
            }
        } else {
            for (IndividualTDE classifier : classifiers) {
                double classification = classifier.classifyInstance(instance);
                classHist[(int) classification] += classifier.getWeight();
                sum += classifier.getWeight();

                if (interpSavePath != null) {
                    interpData.add(classifier.getLastNNIdx());
                    interpPreds.add((int) classification);
                }
            }
        }

        double[] distributions = new double[numClasses];

        if (sum != 0) {
            for (int i = 0; i < classHist.length; ++i)
                distributions[i] += classHist[i] / sum;
        } else {
            for (int i = 0; i < classHist.length; ++i)
                distributions[i] += 1.0 / numClasses;
        }

        if (interpSavePath != null) {
            interpSeries = instance.toValueArray()[0];
            interpPred = argMax(distributions, rand);
        }

        return distributions;
    }

    /**
     * Find class probabilities of an instance using the trained model.
     *
     * @param instance weka Instance object
     * @return array of doubles: probability of each class
     * @throws Exception failure to classify
     */
    @Override //AbstractClassifier
    public double[] distributionForInstance(Instance instance) throws Exception {
        return distributionForInstance(Converter.fromArff(instance));
    }

    /**
     * Classify an instance using the trained model.
     *
     * @param instance TimeSeriesInstance object
     * @return predicted class value
     * @throws Exception failure to classify
     */
    @Override //TSClassifier
    public double classifyInstance(TimeSeriesInstance instance) throws Exception {
        double[] probs = distributionForInstance(instance);
        return findIndexOfMax(probs, rand);
    }

    /**
     * Classify an instance using the trained model.
     *
     * @param instance weka Instance object
     * @return predicted class value
     * @throws Exception failure to classify
     */
    @Override //AbstractClassifier
    public double classifyInstance(Instance instance) throws Exception {
        return classifyInstance(Converter.fromArff(instance));
    }

    /**
     * Set the train time limit for a contracted classifier.
     *
     * @param amount contract time in nanoseconds
     */
    @Override //TrainTimeContractable
    public void setTrainTimeLimit(long amount) {
        trainContractTimeNanos = amount;
        trainTimeContract = true;
    }

    /**
     * Check if a contracted classifier is within its train time limit.
     *
     * @param start classifier build start time
     * @return true if within the contract or not contracted, false otherwise.
     */
    @Override //TrainTimeContractable
    public boolean withinTrainContract(long start) {
        if (trainContractTimeNanos <= 0) return true; //Not contracted
        if (getEstimateOwnPerformance() && trainEstimateMethod == TrainEstimateMethod.OOB)
            return System.nanoTime() - start - checkpointTimeDiff < trainContractTimeNanos -
                    (20000000*train.numInstances() * classifiers.size());
        return System.nanoTime() - start - checkpointTimeDiff < trainContractTimeNanos;
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
        if (validPath) {
            checkpointPath = path;
            checkpoint = true;
        }
        return validPath;
    }


    /**
     * Copies values from a loaded TDE object and IndividualTDE objects into this object.
     *
     * @param obj a TDE object
     * @throws Exception if obj is not an instance of TDE
     */
    @Override //Checkpointable
    public void copyFromSerObject(Object obj) throws Exception {
        if (!(obj instanceof TDE))
            throw new Exception("The SER file is not an instance of TDE");
        TDE saved = ((TDE) obj);
        System.out.println("Loading TDE.ser");

        //copy over variables from serialised object
        parametersConsidered = saved.parametersConsidered;
        maxEnsembleSize = saved.maxEnsembleSize;
        histogramIntersection = saved.histogramIntersection;
        useBigrams = saved.useBigrams;
        useFeatureSelection = saved.useFeatureSelection;
        trainProportion = saved.trainProportion;
        dimensionCutoffThreshold = saved.dimensionCutoffThreshold;
        maxNoDimensions = saved.maxNoDimensions;
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
        //numThreads = saved.numThreads;
        //multiThread = saved.multiThread;
        //ex = saved.ex;
        classifiersBuilt = saved.classifiersBuilt;
        lowestAccIdx = saved.lowestAccIdx;
        lowestAcc = saved.lowestAcc;
        maxAcc = saved.maxAcc;
        visSavePath = saved.visSavePath;
        interpSavePath = saved.interpSavePath;
        //interpData = saved.interpData;
        //interpPreds = saved.interpPreds;
        //interpCount = saved.interpCount;
        //interpSeries = saved.interpSeries;
        //interpPred = saved.interpPred;

        trainResults = saved.trainResults;
        if (!internalContractCheckpointHandling) trainResults.setBuildTime(System.nanoTime());
        seedClassifier = saved.seedClassifier;
        seed = saved.seed;
        rand = saved.rand;
        estimateOwnPerformance = saved.estimateOwnPerformance;
        trainEstimateMethod = saved.trainEstimateMethod;

        //load in each serisalised classifier
        classifiers = new LinkedList<>();
        for (int i = 0; i < maxEnsembleSize; i++) {
            if (!checkpointIDs.contains(i)) {
                System.out.println("Loading IndividualTDE-" + i + ".ser");

                FileInputStream fis = new FileInputStream(checkpointPath + "IndividualTDE-" + i + ".ser");
                try (ObjectInputStream in = new ObjectInputStream(fis)) {
                    Object indv = in.readObject();

                    if (!(indv instanceof IndividualTDE))
                        throw new Exception("The SER file " + i + " is not an instance of IndividualTDE");
                    IndividualTDE ser = ((IndividualTDE) indv);
                    classifiers.add(ser);
                }
            }
        }

        if (internalContractCheckpointHandling) checkpointTimeDiff = saved.checkpointTimeDiff
                + (System.nanoTime() - saved.checkpointTime);
        underContractTime = withinTrainContract(trainResults.getBuildTime());
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

    @Override
    public boolean setInterpretabilitySavePath(String path) {
        boolean validPath = Interpretable.super.createInterpretabilityDirectories(path);
        if (validPath) {
            interpSavePath = path;
        }
        return validPath;
    }

    @Override
    public boolean lastClassifiedInterpretability() throws Exception {
        if (interpSavePath == null) {
            System.err.println("TDE interpretability output save path not set.");
            return false;
        }

        if (train.isMultivariate()) {
            System.err.println("TDE interpretability only available for univariate series.");
            return false;
        }

        TreeMap<Integer, Double> topNeighbours = new TreeMap<>(Collections.reverseOrder());
        for (int i = 0; i < interpData.size(); i++) {
            if (train.get(interpData.get(i)).getLabelIndex() == interpPred) {
                Double val = topNeighbours.get(interpData.get(i));
                if (val == null) val = 0.0;
                topNeighbours.put(interpData.get(i), val + classifiers.get(i).getWeight());
            }
        }

        int topNeighbour = 0;
        double topInstanceWeight = Double.MIN_VALUE;
        for (Map.Entry<Integer, Double> entry : topNeighbours.entrySet()) {
            if (entry.getValue() > topInstanceWeight) {
                topNeighbour = entry.getKey();
                topInstanceWeight = entry.getValue();
            }
        }

        int topClassifier = 0;
        double topClassifierWeight = Double.MIN_VALUE;
        for (int i = 0; i < interpData.size(); i++) {
            if (interpData.get(i) == topNeighbour && classifiers.get(i).getWeight() > topClassifierWeight) {
                topClassifier = i;
                topClassifierWeight = classifiers.get(i).getWeight();
            }
        }

        IndividualTDE tde = classifiers.get(topClassifier);
        double[] nearestSeries = train.get(topNeighbour).toValueArray()[0];

        IndividualTDE.Bag histogram = tde.getLastNNBag();
        IndividualTDE.Bag nearestHistogram = tde.getBags().get(tde.getSubsampleIndices().indexOf(topNeighbour));

        TreeSet<SerialisableComparablePair<Byte, String>> keys = new TreeSet<>((obj1, obj2) -> {
            int c1 = obj1.var1 - obj2.var1;
            if (c1 != 0) {
                return c1;
            } else {
                int c2 = obj1.var2.length() - obj2.var2.length();
                if (c2 != 0) {
                    return c2;
                } else {
                    return obj1.var2.compareTo(obj2.var2);
                }
            }
        });

        HashMap<SerialisableComparablePair<Byte, String>, Integer> histWords = new HashMap<>();
        for (Map.Entry<SerialisableComparablePair<BitWord, Byte>, Integer> entry : histogram.entrySet()) {
            String word = entry.getKey().var2 == -1 ? ((BitWordLong) entry.getKey().var1).toStringBigram()
                    : ((BitWordInt) entry.getKey().var1).toStringUnigram();

            keys.add(new SerialisableComparablePair<>(entry.getKey().var2, word));
            histWords.put(new SerialisableComparablePair<>(entry.getKey().var2, word), entry.getValue());
        }

        HashMap<SerialisableComparablePair<Byte, String>, Integer> nearestWords = new HashMap<>();
        for (Map.Entry<SerialisableComparablePair<BitWord, Byte>, Integer> entry : nearestHistogram.entrySet()) {
            String word = entry.getKey().var2 == -1 ? ((BitWordLong) entry.getKey().var1).toStringBigram()
                    : ((BitWordInt) entry.getKey().var1).toStringUnigram();
            keys.add(new SerialisableComparablePair<>(entry.getKey().var2, word));
            nearestWords.put(new SerialisableComparablePair<>(entry.getKey().var2, word), entry.getValue());
        }

        int numLevels = 1;
        for (int i = 0; i < tde.getLevels(); i++) {
            numLevels += Math.pow(2, i);
        }

        ArrayList<Integer>[][] counts = new ArrayList[numLevels][2];
        ArrayList<String>[] words = new ArrayList[numLevels];
        for (int i = 0; i < numLevels; i++) {
            words[i] = new ArrayList<>();
            for (int n = 0; n < 2; n++) {
                counts[i][n] = new ArrayList<>();
            }
        }

        for (SerialisableComparablePair<Byte, String> key : keys) {
            int idx = key.var1 == -1 ? numLevels - 1 : key.var1;

            words[idx].add(key.var2);

            Integer val = histWords.get(key);
            if (val == null) val = 0;
            counts[idx][0].add(val);

            Integer val2 = nearestWords.get(key);
            if (val2 == null) val2 = 0;
            counts[idx][1].add(val2);
        }

        OutFile of = new OutFile(interpSavePath + "/pred" + seed + "-" + interpCount
                + ".txt");
        of.writeLine(Arrays.toString(interpSeries));
        for (int i = 0; i < numLevels; i++) {
            of.writeLine(words[i].toString());
            of.writeLine(counts[i][0].toString());
        }
        of.writeLine(Arrays.toString(nearestSeries));
        for (int i = 0; i < numLevels; i++) {
            of.writeLine(words[i].toString());
            of.writeLine(counts[i][1].toString());
        }

        Process p = Runtime.getRuntime().exec("py src/main/python/interpretabilityTDE.py \"" +
                interpSavePath.replace("\\", "/") + "\" " + seed + " " + interpCount
                + " " + tde.getLevels() + " " + interpPred + " " + train.get(topNeighbour).getLabelIndex());

        interpCount++;

        if (debug) {
            System.out.println("TDE interp python output:");
            BufferedReader out = new BufferedReader(new InputStreamReader(p.getInputStream()));
            BufferedReader err = new BufferedReader(new InputStreamReader(p.getErrorStream()));
            System.out.println("output : ");
            String outLine = out.readLine();
            while (outLine != null) {
                System.out.println(outLine);
                outLine = out.readLine();
            }
            System.out.println("error : ");
            String errLine = err.readLine();
            while (errLine != null) {
                System.out.println(errLine);
                errLine = err.readLine();
            }
        }

        return true;
    }

    @Override
    public int getPredID() {
        return interpCount;
    }

    @Override
    public boolean setVisualisationSavePath(String path) {
        boolean validPath = Visualisable.super.createVisualisationDirectories(path);
        if (validPath) {
            visSavePath = path;
        }
        return validPath;
    }

    @Override
    public boolean createVisualisation() throws Exception {
        if (visSavePath == null) {
            System.err.println("TDE visualisation save path not set.");
            return false;
        }

        if (train.isMultivariate()) {
            System.err.println("TDE visualisation only available for univariate series.");
            return false;
        }

        HashMap<Integer, Double> wordLengthCounts = new HashMap<>(wordLengths.length);
        HashMap<Boolean, Double> normCounts = new HashMap<>(normOptions.length);
        HashMap<Integer, Double> levelsCounts = new HashMap<>(levels.length);
        HashMap<Boolean, Double> IGBCounts = new HashMap<>(useIGB.length);
        ArrayList<Integer> windowLengths = new ArrayList<>(classifiers.size());
        double weightSum = 0;
        for (int i = 0; i < classifiers.size(); i++) {
            IndividualTDE cls = classifiers.get(i);

            Double val = wordLengthCounts.get(cls.getWordLength());
            if (val == null) val = 0.0;
            wordLengthCounts.put(cls.getWordLength(), val + cls.getWeight());

            Double val2 = normCounts.get(cls.getNorm());
            if (val2 == null) val2 = 0.0;
            normCounts.put(cls.getNorm(), val2 + cls.getWeight());

            Double val3 = levelsCounts.get(cls.getLevels());
            if (val3 == null) val3 = 0.0;
            levelsCounts.put(cls.getLevels(), val3 + cls.getWeight());

            Double val4 = IGBCounts.get(cls.getIGB());
            if (val4 == null) val4 = 0.0;
            IGBCounts.put(cls.getIGB(), val4 + cls.getWeight());

            windowLengths.add(cls.getWindowSize());

            weightSum += cls.getWeight();
        }

        int maxWordLength = -1;
        double maxWeight1 = -1;
        for (Map.Entry<Integer, Double> ent : wordLengthCounts.entrySet()) {
            if (ent.getValue() > maxWeight1 || (ent.getValue() == maxWeight1 && rand.nextBoolean())) {
                maxWordLength = ent.getKey();
                maxWeight1 = ent.getValue();
            }
        }

        int maxLevels = -1;
        double maxWeight2 = -1;
        for (Map.Entry<Integer, Double> ent : levelsCounts.entrySet()) {
            if (ent.getValue() > maxWeight2 || (ent.getValue() == maxWeight2 && rand.nextBoolean())) {
                maxLevels = ent.getKey();
                maxWeight2 = ent.getValue();
            }
        }

        Collections.sort(windowLengths);
        int medianWindowLength;
        if (windowLengths.size() % 2 == 1)
            medianWindowLength = windowLengths.get(windowLengths.size() / 2);
        else
            medianWindowLength = (windowLengths.get(windowLengths.size() / 2 - 1) +
                    windowLengths.get(windowLengths.size() / 2)) / 2;

        ArrayList<IndividualTDE> sortedClassifiers = new ArrayList<>(classifiers);
        Collections.sort(sortedClassifiers, Collections.reverseOrder());
        IndividualTDE tde = null;
        int rank = 1;
        for (IndividualTDE indiv : sortedClassifiers) {
            if (indiv.getWordLength() == maxWordLength && indiv.getLevels() == maxLevels) {
                tde = indiv;
                break;
            }
            rank++;
        }

        if (tde == null) {
            System.out.println("No TDE classifier with word length: " + maxWordLength + ", levels: " + maxLevels
                    + ", using top weighted classifier.");
            tde = sortedClassifiers.get(0);
            rank = 1;
        }

        HashMap<SerialisableComparablePair<Byte, String>, Integer>[] classCounts = new HashMap[getNumClasses()];
        for (int i = 0; i < getNumClasses(); i++) {
            classCounts[i] = new HashMap<>();
        }

        int[] classCount = new int[getNumClasses()];
        for (IndividualTDE.Bag bag : tde.getBags()) {
            int cls = bag.getClassVal();
            if (classCount[cls] >= 1) continue;
            classCount[cls]++;

            for (Map.Entry<SerialisableComparablePair<BitWord, Byte>, Integer> entry : bag.entrySet()) {
                SerialisableComparablePair<BitWord, Byte> key = entry.getKey();

                String word = key.var2 == -1 ? ((BitWordLong) key.var1).toStringBigram()
                        : ((BitWordInt) key.var1).toStringUnigram();

                SerialisableComparablePair<Byte, String> newKey = new SerialisableComparablePair<>(key.var2, word);
                Integer val = classCounts[cls].get(newKey);
                if (val == null) val = 0;
                classCounts[cls].put(newKey, val + entry.getValue());
            }
        }

        TreeSet<SerialisableComparablePair<Byte, String>> keys = new TreeSet<>((obj1, obj2) -> {
            int c1 = obj1.var1 - obj2.var1;
            if (c1 != 0) {
                return c1;
            } else {
                int c2 = obj1.var2.length() - obj2.var2.length();
                if (c2 != 0) {
                    return c2;
                } else {
                    return obj1.var2.compareTo(obj2.var2);
                }
            }
        });
        for (HashMap<SerialisableComparablePair<Byte, String>, Integer> map : classCounts) {
            keys.addAll(map.keySet());
        }

        int numLevels = 1;
        for (int i = 0; i < tde.getLevels(); i++) {
            numLevels += Math.pow(2, i);
        }

        ArrayList<Integer>[][] counts = new ArrayList[numLevels][getNumClasses()];
        ArrayList<String>[] words = new ArrayList[numLevels];
        for (int i = 0; i < numLevels; i++) {
            words[i] = new ArrayList<>();
            for (int n = 0; n < getNumClasses(); n++) {
                counts[i][n] = new ArrayList<>();
            }
        }

        for (SerialisableComparablePair<Byte, String> key : keys) {
            int idx = key.var1 == -1 ? numLevels - 1 : key.var1;

            words[idx].add(key.var2);
            for (int i = 0; i < getNumClasses(); i++) {
                Integer val = classCounts[i].get(key);
                if (val == null) val = 0;
                counts[idx][i].add(val);
            }
        }

        TimeSeriesInstance example = train.get(tde.getSubsampleIndices().get(0));
        BitWordInt word = new BitWordInt();
        double[] dft = tde.firstWordVis(example, word);

        OutFile of = new OutFile(visSavePath + "/vis" + seed + ".txt");
        of.writeLine(Double.toString(tde.getWeight()));
        of.writeLine(rank + " " + classifiers.size());
        of.writeLine(tde.getWordLength() + " " + wordLengthCounts.get(tde.getWordLength()));
        of.writeLine(tde.getNorm() + " " + normCounts.get(tde.getNorm()));
        of.writeLine(tde.getLevels() + " " + levelsCounts.get(tde.getLevels()));
        of.writeLine(tde.getIGB() + " " + IGBCounts.get(tde.getIGB()));
        of.writeLine(tde.getWindowSize() + " " + medianWindowLength);
        of.writeLine(Double.toString(weightSum));
        of.writeLine(Arrays.toString(classCount));
        of.writeLine(Arrays.toString(example.toValueArray()[0]));
        of.writeLine(Arrays.toString(dft));
        of.writeLine(word.toStringUnigram());
        double[][] breakpoints = tde.getBreakpoints();
        of.writeString(Arrays.toString(breakpoints[0]));
        for (int i = 1; i < breakpoints.length; i++) {
            of.writeString(";" + Arrays.toString(breakpoints[i]));
        }
        of.writeLine("");
        for (int i = 0; i < numLevels; i++) {
            of.writeLine(words[i].toString());
            for (int n = 0; n < getNumClasses(); n++) {
                of.writeLine(counts[i][n].toString());
            }
        }
        of.closeFile();

        Process p = Runtime.getRuntime().exec("py src/main/python/visTDE.py \"" +
                visSavePath.replace("\\", "/") + "\" " + seed + " " + getNumClasses());

        if (debug) {
            System.out.println("TDE vis python output:");
            BufferedReader out = new BufferedReader(new InputStreamReader(p.getInputStream()));
            BufferedReader err = new BufferedReader(new InputStreamReader(p.getErrorStream()));
            System.out.println("output : ");
            String outLine = out.readLine();
            while (outLine != null) {
                System.out.println(outLine);
                outLine = out.readLine();
            }
            System.out.println("error : ");
            String errLine = err.readLine();
            while (errLine != null) {
                System.out.println(errLine);
                errLine = err.readLine();
            }
        }

        return true;
    }

    /**
     * Development tests for the TDE classifier.
     *
     * @param args arguments, unused
     * @throws Exception if tests fail
     */
    public static void main(String[] args) throws Exception {
        int fold = 0;

        //Minimum working example
        String dataset = "ItalyPowerDemand";
        Instances[] data = DatasetLoading.sampleItalyPowerDemand(fold);
        Instances train = data[0];
        Instances test = data[1];

        String dataset2 = "ERing";
        Instances[] data2 = DatasetLoading.sampleERing(fold);
        Instances train2 = data2[0];
        Instances test2 = data2[1];

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

        //Output 15/03/21
        /*
            TDE accuracy on ItalyPowerDemand fold 0 = 0.9543245869776482
            Train accuracy on ItalyPowerDemand fold 0 = 0.9701492537313433
            TDE accuracy on ERing fold 0 = 0.9629629629629629
            Train accuracy on ERing fold 0 = 0.9333333333333333
            Contract 1 Min Checkpoint TDE accuracy on ItalyPowerDemand fold 0 = 0.9523809523809523
            Build time on ItalyPowerDemand fold 0 = 7 seconds
            Contract 1 Min Checkpoint TDE accuracy on ERing fold 0 = 0.9555555555555556
            Build time on ERing fold 0 = 60 seconds
        */
    }
}
