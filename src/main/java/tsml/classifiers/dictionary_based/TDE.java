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
import tsml.data_containers.TSCapabilities;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import utilities.ClassifierTools;
import weka.classifiers.functions.GaussianProcesses;
import weka.core.*;

import java.io.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static utilities.InstanceTools.resampleTrainAndTestInstances;
import static utilities.multivariate_tools.MultivariateInstanceTools.*;

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
 *
 * @author Matthew Middlehurst
 */
public class TDE extends EnhancedAbstractClassifier implements TrainTimeContractable,
        Checkpointable, TechnicalInformationHandler, MultiThreadable {

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
    public void setParametersConsidered(int size) { parametersConsidered = size; }

    /**
     * Max number of classifiers for the ensemble.
     *
     * @param size max ensemble size
     */
    public void setMaxEnsembleSize(int size) { maxEnsembleSize = size; }

    /**
     * Proportion of train set to be randomly subsampled for each classifier.
     *
     * @param d train subsample proportion
     */
    public void setTrainProportion(double d) { trainProportion = d; }

    /**
     * Dimension accuracy cutoff threshold for multivariate time series.
     *
     * @param d dimension cutoff threshold
     */
    public void setDimensionCutoffThreshold(double d) { dimensionCutoffThreshold = d; }

    /**
     * Maximum number of dimensions kept for multivariate time series.
     *
     * @param d Max number of dimensions
     */
    public void setMaxNoDimensions(int d) { maxNoDimensions = d; }

    /**
     * Whether to delete checkpoint files after building has finished.
     *
     * @param b clean up checkpoint files
     */
    public void setCleanupCheckpointFiles(boolean b) { cleanupCheckpointFiles = b; }

    /**
     * Whether to load checkpoint files and finish building from the loaded state.
     *
     * @param b load ser files and finish building
     */
    public void loadAndFinish(boolean b) { loadAndFinish = b; }

    /**
     * Max window length as proportion of the series length.
     *
     * @param d max window length proportion
     */
    public void setMaxWinLenProportion(double d) { maxWinLenProportion = d; }

    /**
     * Max number of window lengths to search through as proportion of the series length.
     *
     * @param d window length search proportion
     */
    public void setMaxWinSearchProportion(double d) { maxWinSearchProportion = d; }

    /**
     * Whether to use GP parameter selection for IndividualBOSS classifiers.
     *
     * @param b use GP parameter selection
     */
    public void setBayesianParameterSelection(boolean b) { bayesianParameterSelection = b; }

    /**
     * Wether to use bigrams in IndividualBOSS classifiers.
     *
     * @param b use bigrams
     */
    public void setUseBigrams(Boolean b) { useBigrams = b; }

    /**
     * Wether to use feature selection in IndividualBOSS classifiers.
     *
     * @param b use feature selection
     */
    public void setUseFeatureSelection(boolean b) { useFeatureSelection = b; }

    /**
     * Whether to remove ensemble members below a proportion of the highest accuracy.
     *
     * @param b use ensemble cutoff
     */
    public void setCutoff(boolean b) { cutoff = b; }

    /**
     * Ensemble accuracy cutoff as proportion of highest ensemble memeber accuracy.
     *
     * @param d cutoff proportion
     */
    public void setCutoffThreshold(double d) { cutoffThreshold = d; }

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
    public TSCapabilities getTSCapabilities(){
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

            if (checkpoint){
                checkpointIDs = new ArrayList<>();
                for (int i = 0; i < maxEnsembleSize; i++){
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
            if (series.isMultivariate()){
                indiv = new MultivariateIndividualTDE((int) parameters[0], (int) parameters[1], (int) parameters[2],
                        parameters[3] == 1, (int) parameters[4], parameters[5] == 1,
                        multiThread, numThreads, ex);
                ((MultivariateIndividualTDE)indiv).setDimensionCutoffThreshold(dimensionCutoffThreshold);
                ((MultivariateIndividualTDE)indiv).setMaxNoDimensions(maxNoDimensions);
            }
            else{
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
                    ? Double.MIN_VALUE : lowestAcc);
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

                        if (checkpoint){
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

                    if (checkpoint){
                        indiv.setEnsembleID(checkpointIDs.remove(0));
                        checkpointChange = true;
                    }
                } else if (accuracy > lowestAcc) {
                    double[] newLowestAcc = findMinEnsembleAcc();
                    lowestAccIdx = (int) newLowestAcc[0];
                    lowestAcc = newLowestAcc[1];

                    IndividualTDE rm = classifiers.remove(lowestAccIdx);
                    classifiers.add(lowestAccIdx, indiv);

                    if (checkpoint){
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
     * @param saveIndiv whether to save the new IndividualClassifier
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
     * @param winInc window size increment
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
     * @param indiv classifier being subsampled for
     * @return subsampled data
     */
    private TimeSeriesInstances subsampleData(TimeSeriesInstances series, IndividualTDE indiv) {
        int newSize = (int) (series.numInstances() * trainProportion);
        ArrayList<TimeSeriesInstance> data = new ArrayList<>();

        ArrayList<Integer> indices = new ArrayList<>(series.numInstances());
        for (int n = 0; n < series.numInstances(); n++){
            indices.add(n);
        }

        ArrayList<Integer> subsampleIndices = new ArrayList<>(series.numInstances());
        while (subsampleIndices.size() < newSize){
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
     * @param indiv classifier to evaluate
     * @param series subsampled data
     * @param lowestAcc lowest accuracy in the ensemble
     * @return estimated accuracy
     * @throws Exception unable to estimate accuracy
     */
    private double individualTrainAcc(IndividualTDE indiv, TimeSeriesInstances series, double lowestAcc)
            throws Exception {
        if (getEstimateOwnPerformance() && estimator == EstimatorMethod.NONE) {
            indiv.setTrainPreds(new ArrayList<>());
        }

        int correct = 0;
        int requiredCorrect = (int) (lowestAcc * series.numInstances());

        if (multiThread) {
            ArrayList<Future<Double>> futures = new ArrayList<>(series.numInstances());

            for (int i = 0; i < series.numInstances(); ++i) {
                if (series.isMultivariate())
                    futures.add(ex.submit(((MultivariateIndividualTDE)indiv).new TrainNearestNeighbourThread(i)));
                else
                    futures.add(ex.submit(indiv.new TrainNearestNeighbourThread(i)));
            }

            int idx = 0;
            for (Future<Double> f : futures) {
                if (f.get() == series.get(idx).getLabelIndex()) {
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
                if (c == series.get(i).getLabelIndex()) {
                    ++correct;
                }

                if (getEstimateOwnPerformance() && estimator == EstimatorMethod.NONE) {
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
        if (estimator == EstimatorMethod.OOB && trainProportion < 1){
            for (int i = 0; i < train.numInstances(); ++i) {
                double[] probs = new double[train.numClasses()];
                double sum = 0;

                for (int j = 0; j < classifiers.size(); j++) {
                    IndividualTDE classifier = classifiers.get(j);

                    if (!classifier.getSubsampleIndices().contains(i)){
                        probs[(int)classifier.classifyInstance(train.get(i))] += classifier.getWeight();
                        sum += classifier.getWeight();
                    }
                }

                if (sum != 0) {
                    for (int j = 0; j < probs.length; ++j)
                        probs[j] = (probs[j] / sum);
                }
                else{
                    Arrays.fill(probs, 1.0 / train.numClasses());
                }

                trainResults.addPrediction(train.get(i).getLabelIndex(), probs, findIndexOfMax(probs, rand),
                        -1, "");
            }

            trainResults.setClassifierName("TDEOOB");
            trainResults.setErrorEstimateMethod("OOB");
        }
        else {
            double[][] trainDistributions = new double[train.numInstances()][train.numClasses()];
            int[] idxSubsampleCount = new int[train.numInstances()];

            if (estimator == EstimatorMethod.NONE) {
                for (int i = 0; i < classifiers.size(); i++) {
                    ArrayList<Integer> trainIdx = classifiers.get(i).getSubsampleIndices();
                    ArrayList<Integer> trainPreds = classifiers.get(i).getTrainPreds();
                    double weight = classifiers.get(i).getWeight();
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
     * @param ins train instance index
     * @return array of doubles: probability of each class
     * @throws Exception failure to classify
     */
    private double[] distributionForInstance(int test) throws Exception {
        double[] probs = new double[train.numClasses()];

        //get sum of all channels, votes from each are weighted the same.
        double sum = 0;

        for (IndividualTDE classifier : classifiers) {
            double classification;

            if (classifier.getSubsampleIndices() == null){
                classification = classifier.classifyInstance(test);
            }
            else if (classifier.getSubsampleIndices().contains(test)){
                classification = classifier.classifyInstance(classifier.getSubsampleIndices().indexOf(test));
            }
            else if (estimator == EstimatorMethod.CV) {
                TimeSeriesInstance series = train.get(test);
                classification = classifier.classifyInstance(series);
            }
            else{
                continue;
            }

            probs[(int) classification] += classifier.getWeight();
            sum += classifier.getWeight();
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

    /**
     * Find class probabilities of an instance using the trained model.
     *
     * @param ins TimeSeriesInstance object
     * @return array of doubles: probability of each class
     * @throws Exception failure to classify
     */
    @Override //TSClassifier
    public double[] distributionForInstance(TimeSeriesInstance instance) throws Exception {
        double[] classHist = new double[numClasses];

        //get sum of all channels, votes from each are weighted the same.
        double sum = 0;

        if (multiThread){
            ArrayList<Future<Double>> futures = new ArrayList<>(classifiers.size());

            for (IndividualTDE classifier : classifiers) {
                if (train.isMultivariate())
                    futures.add(ex.submit(((MultivariateIndividualTDE)classifier)
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
        }
        else {
            for (IndividualTDE classifier : classifiers) {
                double classification = classifier.classifyInstance(instance);
                classHist[(int) classification] += classifier.getWeight();
                sum += classifier.getWeight();
            }
        }

        double[] distributions = new double[numClasses];

        if (sum != 0) {
            for (int i = 0; i < classHist.length; ++i)
                distributions[i] += classHist[i] / sum;
        }
        else{
            for (int i = 0; i < classHist.length; ++i)
                distributions[i] += 1.0 / numClasses;
        }

        return distributions;
    }

    /**
     * Find class probabilities of an instance using the trained model.
     *
     * @param ins weka Instance object
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
     * @param ins TimeSeriesInstance object
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
     * @param ins weka Instance object
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
        if(trainContractTimeNanos <= 0) return true; //Not contracted
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
        if(validPath){
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

        trainResults = saved.trainResults;
        if (!internalContractCheckpointHandling) trainResults.setBuildTime(System.nanoTime());
        seedClassifier = saved.seedClassifier;
        seed = saved.seed;
        rand = saved.rand;
        estimateOwnPerformance = saved.estimateOwnPerformance;
        estimator = saved.estimator;

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

    /**
     * Development tests for the TDE classifier.
     *
     * @param arg arguments, unused
     * @throws Exception if tests fail
     */
    public static void main(String[] args) throws Exception{
        int fold = 0;

        //Minimum working example
        String dataset = "ItalyPowerDemand";
        Instances train = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\"+dataset+
                "\\"+dataset+"_TRAIN.arff");
        Instances test = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Univariate_arff\\"+dataset+
                "\\"+dataset+"_TEST.arff");
        Instances[] data = resampleTrainAndTestInstances(train, test, fold);
        train = data[0];
        test = data[1];

        String dataset2 = "ERing";
        Instances train2 = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Multivariate_arff\\"+dataset2+
                "\\"+dataset2+"_TRAIN.arff");
        Instances test2 = DatasetLoading.loadDataNullable("Z:\\ArchiveData\\Multivariate_arff\\"+dataset2+
                "\\"+dataset2+"_TEST.arff");
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

        //Output 06/10/20
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
