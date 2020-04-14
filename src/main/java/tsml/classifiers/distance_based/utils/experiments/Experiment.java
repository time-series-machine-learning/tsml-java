package tsml.classifiers.distance_based.utils.experiments;

import com.beust.jcommander.IStringConverter;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.internal.Lists;
import evaluation.storage.ClassifierResults;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;
import org.junit.Assert;
import tsml.classifiers.Checkpointable;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.MemoryContractable;
import tsml.classifiers.TestTimeContractable;
import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.distance_based.proximity.RandomSource;
import tsml.classifiers.distance_based.utils.classifier_building.ClassifierBuilderFactory;
import tsml.classifiers.distance_based.utils.classifier_mixins.Copy;
import tsml.classifiers.distance_based.utils.classifier_mixins.TrainEstimateable;
import tsml.classifiers.distance_based.utils.logging.LogUtils;
import tsml.classifiers.distance_based.utils.logging.Loggable;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.stopwatch.TimeAmount;
import tsml.classifiers.distance_based.utils.system.memory.MemoryAmount;
import utilities.ArrayUtilities;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Randomizable;

/**
 * Purpose: class encapsulating a single experiment (i.e. single train / test of a classifier).
 * <p>
 * Contributors: goastler
 */
public class Experiment implements Copy, TrainTimeContractable, Checkpointable, Loggable, MemoryContractable,
    TestTimeContractable {

    static class TimeAmountConverter implements
        IStringConverter<TimeAmount> {

        @Override
        public TimeAmount convert(final String str) {
            return TimeAmount.parse(str);
        }
    }

    static class MemoryAmountConverter implements
        IStringConverter<MemoryAmount> {

        @Override
        public MemoryAmount convert(final String str) {
            return MemoryAmount.parse(str);
        }
    }

    public ParamSet getParamSet() {
        return paramSet;
    }

    public Experiment setParamSet(final ParamSet paramSet) {
        this.paramSet = paramSet;
        return this;
    }

    // abide by unix cmdline args convention! single char --> single hyphen, multiple chars --> double hyphen

    private Instances trainData;
    private Instances testData;
    private Classifier classifier;
    private ClassifierResults testResults;
    private ClassifierResults trainResults;

    // the seeds to run
    public static final String SEED_SHORT_FLAG = "-s";
    public static final String SEED_LONG_FLAG = "--seed";
    @Parameter(names = {SEED_SHORT_FLAG, SEED_LONG_FLAG}, description = "the seed to be used in sampling a dataset "
        + "and in the random source for the classifier", required = true)
    private Integer seed;

    // the classifier to use
    public static final String CLASSIFIER_SHORT_FLAG = "-c";
    public static final String CLASSIFIER_LONG_FLAG = "--classifier";
    @Parameter(names = {CLASSIFIER_SHORT_FLAG, CLASSIFIER_LONG_FLAG},
        description = "append the train memory contract to the classifier name")
    private String classifierName;

    // where to put the results when finished
    public static final String RESULTS_DIR_SHORT_FLAG = "-r";
    public static final String RESULTS_DIR_LONG_FLAG = "--resultsDir";
    @Parameter(names = {RESULTS_DIR_SHORT_FLAG, RESULTS_DIR_LONG_FLAG}, description = "path to a folder to place "
        + "results in",
        required = true)
    private String resultsDirPath;

    // paths to directory where problem data is stored
    public static final String DATASET_DIR_SHORT_FLAG = "--dd";
    public static final String DATASET_DIR_LONG_FLAG = "--datasetsDir";
    @Parameter(names = {DATASET_DIR_SHORT_FLAG, DATASET_DIR_LONG_FLAG}, description = "the path to the folder "
        + "containing the datasets",
        required = true)
    private String datasetDirPath;

    // names of the dataset that should be run
    public static final String DATASET_NAME_SHORT_FLAG = "-d";
    public static final String DATASET_NAME_LONG_FLAG = "--dataset";
    @Parameter(names = {DATASET_NAME_SHORT_FLAG, DATASET_NAME_LONG_FLAG}, description = "the name of the dataset",
        required = true)
    private String datasetName;

    // parameters to pass onto the classifiers
    public static final String PARAMETERS_SHORT_FLAG = "-p";
    public static final String PARAMETERS_LONG_FLAG = "--parameters";
    @Parameter(names = {PARAMETERS_SHORT_FLAG, PARAMETERS_LONG_FLAG}, description = "parameters for the classifiers. ", variableArity =
        true)
    private List<String> classifierParameterStrs = new ArrayList<>();
    private ParamSet classifierParameters = new ParamSet();

    // whether to append the classifier parameters to the classifier name
    public static final String APPEND_CLASSIFIER_PARAMETERS_SHORT_FLAG = "--acp";
    public static final String APPEND_CLASSIFIER_PARAMETERS_LONG_FLAG = "--appendClassifierParameters";
    @Parameter(names = {APPEND_CLASSIFIER_PARAMETERS_SHORT_FLAG, APPEND_CLASSIFIER_PARAMETERS_LONG_FLAG},
        description = "append the classifier parameters to the classifier name")
    private boolean appendClassifierParameters = false;

    private Logger logger = LogUtils.buildLogger(this);

    // the train time contract for the classifier
    public static final String TRAIN_TIME_CONTRACT_SHORT_FLAG = "--ttc";
    public static final String TRAIN_TIME_CONTRACT_LONG_FLAG = "--trainTimeContract";
    @Parameter(names = {TRAIN_TIME_CONTRACT_SHORT_FLAG, TRAIN_TIME_CONTRACT_LONG_FLAG}, converter =
        TimeAmountConverter.class, description =
        "specify a train time contract for the classifier in the form \"<amount> <units>\", e.g. \"4 hour\"")
    private List<TimeAmount> trainTimeContracts = Lists.newArrayList();

    // the train memory contract for the classifier
    public static final String TRAIN_MEMORY_CONTRACT_SHORT_FLAG = "--tmc";
    public static final String TRAIN_MEMORY_CONTRACT_LONG_FLAG = "--trainMemoryContract";
    @Parameter(names = {TRAIN_MEMORY_CONTRACT_SHORT_FLAG, TRAIN_MEMORY_CONTRACT_LONG_FLAG}, converter =
        MemoryAmountConverter.class, description =
        "specify a train memory contract for the classifier in the form \"<amount> <units>\", e.g. \"4 GIGABYTE\" - make"
            + " sure you've considered whether you need GIBIbyte or GIGAbyte though.")
    private List<MemoryAmount> trainMemoryContracts = Lists.newArrayList();

    // the test time contract
    public static final String TEST_TIME_CONTRACT_SHORT_FLAG = "--ptc";
    public static final String TEST_TIME_CONTRACT_LONG_FLAG = "--testTimeContract";
    @Parameter(names = {TEST_TIME_CONTRACT_SHORT_FLAG, TEST_TIME_CONTRACT_LONG_FLAG}, converter =
        TimeAmountConverter.class, description =
        "specify a test time contract for the classifier in the form \"<amount> <unit>\", e.g. \"1 minute\"")
    private List<TimeAmount> testTimeContracts = Lists.newArrayList();

    // checkpoint interval (if using checkpointing)
    private static final String CHECKPOINT_INTERVAL_SHORT_FLAG = "--cpi";
    private static final String CHECKPOINT_INTERVAL_LONG_FLAG = "--checkpointInterval";
    @Parameter(names = {CHECKPOINT_INTERVAL_SHORT_FLAG, CHECKPOINT_INTERVAL_LONG_FLAG}, converter =
        TimeAmountConverter.class, description =
        "how "
            + "often to "
            + "save the classifier to file in the form \"<amount> <unit>\", e.g. \"1 hour\"")
    // todo add checkpoint interval to classifier post tony's interface changes
    private TimeAmount checkpointInteval = new TimeAmount(1, TimeUnit.HOURS);

    // todo append times + mem to clsf name

    // whether to find a train estimate for the classifier
    private static final String ESTIMATE_TRAIN_ERROR_SHORT_FLAG = "-e";
    private static final String ESTIMATE_TRAIN_ERROR_LONG_FLAG = "--estimateTrainError";
    @Parameter(names = {ESTIMATE_TRAIN_ERROR_SHORT_FLAG, ESTIMATE_TRAIN_ERROR_LONG_FLAG}, description = "set the "
        + "classifier to find a train estimate")
    private boolean estimateTrainError = false;
    // todo another parameter for specifying a cv or something of a non-train-estimateable classifier to find a train
    //  estimate for it

    // whether to overwrite train files
    private static final String OVERWRITE_TRAIN_SHORT_FLAG = "--ot";
    private static final String OVERWRITE_TRAIN_LONG_FLAG = "--overwriteTrain";
    @Parameter(names = {OVERWRITE_TRAIN_SHORT_FLAG, OVERWRITE_TRAIN_LONG_FLAG}, description = "overwrite train results")
    private boolean overwriteTrain = false;

    // whether to overwrite test results
    private static final String OVERWRITE_TEST_SHORT_FLAG = "--op";
    private static final String OVERWRITE_TEST_LONG_FLAG = "--overwriteTest";
    @Parameter(names = {OVERWRITE_TEST_SHORT_FLAG, OVERWRITE_TEST_LONG_FLAG}, description = "overwrite test results")
    private boolean overwriteTest = false;

    // the factory to build classifiers using classifier name
    private ClassifierBuilderFactory<Classifier> classifierBuilderFactory =
        ClassifierBuilderFactory.getGlobalInstance();
    // todo get this by string, i.e. factory, and make into cmdline param


    public void resetTrain() {
        trained = false;
        trainResults = null;
    }

    public void resetTest() {
        tested = false;
        testResults = null;
    }

    @Override
    public boolean setSavePath(final String path) {
        if(Checkpointable.super.setSavePath(path)) {
            savePath = path;
            return true;
        } else {
            savePath = null;
            return false;
        }
    }

    @Override
    public String getSavePath() {
        return savePath;
    }

    @Override
    public String getLoadPath() {
        return loadPath;
    }

    @Override
    public boolean setLoadPath(final String path) {
        if(Checkpointable.super.setLoadPath(path)) {
            loadPath = path;
            return true;
        } else {
            loadPath = null;
            return false;
        }
    }

    public void train() throws Exception {
        if(trained) {
            throw new IllegalStateException("already trained");
        } else {
            trained = true;
        }
        getLogger().info("training...");
        // copy the data in case the classifier / other things adjust it (as it may be used elsewhere and we don't
        // want to cause a knock on effect)
        // copy data
        final Instances trainData = new Instances(getTrainData());
        // build the classifier
        final Classifier classifier = getClassifier();
        // set params
        if(!paramSet.isEmpty()) {
            if(classifier instanceof ParamHandler) {
                ((ParamHandler) classifier).setParams(paramSet);
            } else if(classifier instanceof OptionHandler) {
                ((OptionHandler) classifier).setOptions(paramSet.getOptions());
            } else {
                throw new IllegalStateException("{" + classifierName + "} cannot handle parameters");
            }
        }
        // enable train estimate if set
        if(isEstimateTrainError()) {
            if(classifier instanceof TrainEstimateable) {
                ((TrainEstimateable) classifier).setEstimateOwnPerformance(true);
            } else {
                throw new IllegalStateException("{" + classifierName + "} does not estimate train error");
            }
        }
        if(isCheckpointLoadingEnabled()) {
            if(classifier instanceof Checkpointable) {
                ((Checkpointable) classifier).setLoadPath(getLoadPath());
            } else {
                throw new IllegalStateException("{" + classifierName + "} is not checkpointable");
            }
        }
        if(isCheckpointSavingEnabled()) {
            if(classifier instanceof Checkpointable) {
                ((Checkpointable) classifier).setSavePath(getSavePath());
            } else {
                throw new IllegalStateException("{" + classifierName + "} is not checkpointable");
            }
        }
        if(trainTimeContractNanos != null) {
            if(classifier instanceof TrainTimeContractable) {
                ((TrainTimeContractable) classifier).setTrainTimeLimit(trainTimeContractNanos);
            } else {
                throw new IllegalStateException("{" + classifierName + "} not train contractable");
            }
        }
        if(classifier instanceof Randomizable) {
            ((Randomizable) classifier).setSeed(getSeed());
        } else {
            logger.warning("cannot set seed for {" + classifier.toString() + "}");
        }
        if(classifier instanceof Loggable) {
            // todo set to this experiment's logger instead
            ((Loggable) classifier).getLogger().setLevel(getLogger().getLevel());
        } else {
            logger.warning("cannot set logger for {" + classifierName + "}");
        }
        classifier.buildClassifier(trainData);
        if(isEstimateTrainError()) {
            if(classifier instanceof TrainEstimateable) {
                ClassifierResults trainResults = ((TrainEstimateable) classifier).getTrainResults();
                setTrainResults(trainResults);
                getLogger().info("train estimate: " + System.lineSeparator() + trainResults.writeSummaryResultsToString());
            } else {
                throw new IllegalArgumentException("classifier does not estimate train error");
            }
        }
    }

    public boolean isTested() {
        return tested;
    }

    public boolean isTrained() {
        return trained;
    }

    public void test() throws Exception {
        if(tested) {
            throw new IllegalStateException("already tested");
        } else {
            tested = true;
        }
        getLogger().info("testing...");
        final Classifier classifier = getClassifier();
        final Instances testData = new Instances(getTestData());
        // mark the test data as missing class
        InstanceTools.setClassMissing(testData);
        // test the classifier
        final ClassifierResults testResults = new ClassifierResults();
        for(final Instance testInstance : testData) {
            final long timestamp = System.nanoTime();
            final double[] distribution = classifier.distributionForInstance(testInstance);
            final int predictedClass = ArrayUtilities.argMax(distribution);
            final long timeTaken = System.nanoTime() - timestamp;
            testResults.addPrediction(testInstance.classValue(), distribution, predictedClass, timeTaken, "");
        }
        // swap back the random source *only if* we switched it before because of multiple contracts
        setTestResults(testResults);
        getLogger().info("test results: " + System.lineSeparator() + testResults.writeSummaryResultsToString());
    }


    private void setup(Instances trainData, Instances testData, Classifier classifier, int seed,
        String classifierName, String datasetName) {
        setClassifier(classifier);
        setTestData(testData);
        setTrainData(trainData);
        setSeed(seed);
        setClassifierName(classifierName);
        setDatasetName(datasetName);
        if(classifier instanceof Randomizable) {
            ((Randomizable) classifier).setSeed(seed);
        }
    }

    private Experiment() {

    }

    public Experiment(final Instances trainData, final Instances testData, final Classifier classifier, int seed,
        String classifierName, String datasetName) {
        setup(trainData, testData, classifier, seed, classifierName, datasetName);
    }

    public Instances getTrainData() {
        return trainData;
    }

    public Experiment setTrainData(final Instances trainData) {
        Assert.assertNotNull(trainData);
        this.trainData = trainData;
        return this;
    }

    public Instances getTestData() {
        return testData;
    }

    public Experiment setTestData(final Instances testData) {
        Assert.assertNotNull(testData);
        this.testData = testData;
        return this;
    }

    public Classifier getClassifier() {
        return classifier;
    }

    public Experiment setClassifier(final Classifier classifier) {
        this.classifier = classifier;
        return this;
    }

    public ClassifierResults getTestResults() {
        return testResults;
    }

    public Experiment setTestResults(final ClassifierResults testResults) {
        this.testResults = testResults;
        return this;
    }

    public int getSeed() {
        return seed;
    }

    public Experiment setSeed(final int seed) {
        this.seed = seed;
        return this;
    }

    public String getClassifierName() {
        return classifierName;
    }

    public Experiment setClassifierName(final String classifierName) {
        this.classifierName = classifierName;
        return this;
    }

    public String getDatasetName() {
        return datasetName;
    }

    public Experiment setDatasetName(final String datasetName) {
        this.datasetName = datasetName;
        return this;
    }

    public boolean isEstimateTrainError() {
        return estimateTrainError;
    }

    public Experiment setEstimateTrainError(final boolean estimateTrainError) {
        this.estimateTrainError = estimateTrainError;
        return this;
    }

    public ClassifierResults getTrainResults() {
        return trainResults;
    }

    public Experiment setTrainResults(final ClassifierResults trainResults) {
        this.trainResults = trainResults;
        return this;
    }

    @Override
    public void setTrainTimeLimit(final long time) {
        trainTimeContractNanos = time;
    }

    @Override
    public Logger getLogger() {
        return logger;
    }

    @Override
    public void setTestTimeLimit(final TimeUnit time, final long amount) {
        // todo change this to just nanos, adjust interface
        testTimeContractNanos = TimeUnit.NANOSECONDS.convert(amount, time);
    }

    @Override
    public void setMemoryLimit(final DataUnit unit, final long amount) {
        getLogger().warning("need to implement memory limiting");
    }

    // todo tostring
}
