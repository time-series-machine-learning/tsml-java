package tsml.classifiers.distance_based.utils.experiments;

import com.beust.jcommander.IStringConverter;
import com.beust.jcommander.IStringConverterFactory;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.logging.Level;
import tsml.classifiers.distance_based.utils.classifier_building.ClassifierBuilderFactory;
import tsml.classifiers.distance_based.utils.collections.CollectionUtils;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.stopwatch.TimeAmount;
import tsml.classifiers.distance_based.utils.system.memory.MemoryAmount;
import weka.classifiers.Classifier;

public class ExperimentArgs {

    private static class TimeAmountConverter implements
        IStringConverter<TimeAmount> {

        @Override
        public TimeAmount convert(final String str) {
            return TimeAmount.parse(str);
        }
    }

    private static class MemoryAmountConverter implements
        IStringConverter<MemoryAmount> {

        @Override
        public MemoryAmount convert(final String str) {
            return MemoryAmount.parse(str);
        }
    }

    // abide by unix cmdline args convention! single char --> single hyphen, multiple chars --> double hyphen

    // the classifier to use
    private static final String CLASSIFIER_SHORT_FLAG = "-c";
    private static final String CLASSIFIER_LONG_FLAG = "--classifier";
    @Parameter(names = {CLASSIFIER_SHORT_FLAG, CLASSIFIER_LONG_FLAG},
        description = "append the train memory contract to the classifier name")
    private String classifierName;

    // the seeds to run
    private static final String SEED_SHORT_FLAG = "-s";
    private static final String SEED_LONG_FLAG = "--seed";
    @Parameter(names = {SEED_SHORT_FLAG, SEED_LONG_FLAG}, description = "the seed to be used in sampling a dataset "
        + "and in the random source for the classifier", required = true)
    private Integer seed;

    // where to put the results when finished
    private static final String RESULTS_DIR_SHORT_FLAG = "-r";
    private static final String RESULTS_DIR_LONG_FLAG = "--resultsDir";
    @Parameter(names = {RESULTS_DIR_SHORT_FLAG, RESULTS_DIR_LONG_FLAG}, description = "path to a folder to place "
        + "results in",
        required = true)
    private String resultsDirPath;

    // paths to directory where problem data is stored
    private static final String DATASET_DIR_SHORT_FLAG = "--dd";
    private static final String DATASET_DIR_LONG_FLAG = "--datasetsDir";
    @Parameter(names = {DATASET_DIR_SHORT_FLAG, DATASET_DIR_LONG_FLAG}, description = "the path to the folder "
        + "containing the datasets",
        required = true)
    private String datasetDirPath;

    // names of the dataset that should be run
    private static final String DATASET_NAME_SHORT_FLAG = "-d";
    private static final String DATASET_NAME_LONG_FLAG = "--dataset";
    @Parameter(names = {DATASET_NAME_SHORT_FLAG, DATASET_NAME_LONG_FLAG}, description = "the name of the dataset",
        required = true)
    private String datasetName;


    // parameters to pass onto the classifiers
    private static final String PARAMETERS_SHORT_FLAG = "-p";
    private static final String PARAMETERS_LONG_FLAG = "--parameters";
    @Parameter(names = {PARAMETERS_SHORT_FLAG, PARAMETERS_LONG_FLAG}, description = "parameters for the classifiers. ", variableArity =
        true)
    private List<String> classifierParameterStrs = new ArrayList<>();
    private ParamSet classifierParameters = new ParamSet();


    // whether to append the classifier parameters to the classifier name
    private static final String APPEND_CLASSIFIER_PARAMETERS_SHORT_FLAG = "--acp";
    private static final String APPEND_CLASSIFIER_PARAMETERS_LONG_FLAG = "--appendClassifierParameters";
    @Parameter(names = {APPEND_CLASSIFIER_PARAMETERS_SHORT_FLAG, APPEND_CLASSIFIER_PARAMETERS_LONG_FLAG},
        description = "append the classifier parameters to the classifier name")
    private boolean appendClassifierParameters = false;

    // the train time contract for the classifier
    private static final String TRAIN_TIME_CONTRACT_SHORT_FLAG = "--ttc";
    private static final String TRAIN_TIME_CONTRACT_LONG_FLAG = "--trainTimeContract";
    @Parameter(names = {TRAIN_TIME_CONTRACT_SHORT_FLAG, TRAIN_TIME_CONTRACT_LONG_FLAG}, converter =
        TimeAmountConverter.class, description =
        "specify a train time contract for the classifier in the form \"<amount> <units>\", e.g. \"4 hour\"")
    private List<TimeAmount> trainTimeContracts = new ArrayList<>();

    // the train memory contract for the classifier
    private static final String TRAIN_MEMORY_CONTRACT_SHORT_FLAG = "--tmc";
    private static final String TRAIN_MEMORY_CONTRACT_LONG_FLAG = "--trainMemoryContract";
    @Parameter(names = {TRAIN_MEMORY_CONTRACT_SHORT_FLAG, TRAIN_MEMORY_CONTRACT_LONG_FLAG}, converter =
        MemoryAmountConverter.class, description =
        "specify a train memory contract for the classifier in the form \"<amount> <units>\", e.g. \"4 GIGABYTE\" - make"
            + " sure you've considered whether you need GIBIbyte or GIGAbyte though.")
    private List<MemoryAmount> trainMemoryContracts = new ArrayList<>();

    // the test time contract
    private static final String TEST_TIME_CONTRACT_SHORT_FLAG = "--ptc";
    private static final String TEST_TIME_CONTRACT_LONG_FLAG = "--testTimeContract";
    @Parameter(names = {TEST_TIME_CONTRACT_SHORT_FLAG, TEST_TIME_CONTRACT_LONG_FLAG}, converter =
        TimeAmountConverter.class, description =
        "specify a test time contract for the classifier in the form \"<amount> <unit>\", e.g. \"1 minute\"")
    private List<TimeAmount> testTimeContracts = new ArrayList<>();

    // whether to checkpoint or not. Paths for checkpointing will be auto generated.
    private static final String CHECKPOINT_SHORT_FLAG = "--cp";
    private static final String CHECKPOINT_LONG_FLAG = "--checkpoint";
    @Parameter(names = {CHECKPOINT_SHORT_FLAG, CHECKPOINT_LONG_FLAG}, description = "whether to save the classifier "
        + "to file")
    private boolean checkpoint = false;
    // todo swap train contract save / load path over for checkpointing
    // todo enable checkpointing on classifier

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

    // whether to append the train time to the classifier name
    private static final String APPEND_TRAIN_TIME_CONTRACT_SHORT_FLAG = "--attc";
    private static final String APPEND_TRAIN_TIME_CONTRACT_LONG_FLAG = "--appendTrainTimeContract";
    @Parameter(names = {APPEND_TRAIN_TIME_CONTRACT_SHORT_FLAG, APPEND_TRAIN_TIME_CONTRACT_LONG_FLAG}, description =
        "append the train time contract to the classifier name")
    private boolean appendTrainTimeContract = false;

    // whether to append the train memory contract to the classifier name
    private static final String APPEND_TRAIN_MEMORY_CONTRACT_SHORT_FLAG = "--atmc";
    private static final String APPEND_TRAIN_MEMORY_CONTRACT_LONG_FLAG = "--appendTrainMemoryContract";
    @Parameter(names = {APPEND_TRAIN_MEMORY_CONTRACT_SHORT_FLAG, APPEND_TRAIN_MEMORY_CONTRACT_LONG_FLAG},
        description = "append the train memory contract to the classifier name")
    private boolean appendTrainMemoryContract = false;

    // whether to append the test time contract to the classifier name
    private static final String APPEND_TEST_TIME_CONTRACT_SHORT_FLAG = "--aptc";
    private static final String APPEND_TEST_TIME_CONTRACT_LONG_FLAG = "--appendTestTimeContract";
    @Parameter(names = {APPEND_TEST_TIME_CONTRACT_SHORT_FLAG, APPEND_TEST_TIME_CONTRACT_LONG_FLAG}, description =
        "append the test time contract to the classifier name")
    private boolean appendTestTimeContract = false;

    // whether to find a train estimate for the classifier
    private static final String ESTIMATE_TRAIN_ERROR_SHORT_FLAG = "-e";
    private static final String ESTIMATE_TRAIN_ERROR_LONG_FLAG = "--estimateTrainError";
    @Parameter(names = {ESTIMATE_TRAIN_ERROR_SHORT_FLAG, ESTIMATE_TRAIN_ERROR_LONG_FLAG}, description = "set the "
        + "classifier to find a train estimate")
    private boolean estimateTrainError = false;
    // todo enable train estimate to be set on a per classifier basis, similar to universal / bespoke params
    // todo another parameter for specifying a cv or something of a non-train-estimateable classifier to find a train
    //  estimate for it

    // the log level to use on the classifier
    private static final String CLASSIFIER_VERBOSITY_SHORT_FLAG = "--cv";
    private static final String CLASSIFIER_VERBOSITY_LONG_FLAG = "--classifierVerbosity";
    @Parameter(names = {CLASSIFIER_VERBOSITY_SHORT_FLAG, CLASSIFIER_VERBOSITY_LONG_FLAG}, description = "classifier "
        + "verbosity")
    private String classifierVerbosityStr = Level.SEVERE.toString();
    private Level classifierVerbosityLevel;

    // the log level to use on the experiment
    private static final String EXPERIMENT_VERBOSITY_SHORT_FLAG = "--ev";
    private static final String EXPERIMENT_VERBOSITY_LONG_FLAG = "--experimentVerbosity";
    @Parameter(names = {EXPERIMENT_VERBOSITY_SHORT_FLAG, EXPERIMENT_VERBOSITY_LONG_FLAG}, description = "experiment "
        + "verbosity")
    private String experimentVerbosityStr = Level.ALL.toString();
    private Level experimentVerbosityLevel;

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


    @Override
    public String toString() {
        return "ExperimentArgs{" +
            "classifierName='" + classifierName + '\'' +
            ", seed=" + seed +
            ", resultsDirPath='" + resultsDirPath + '\'' +
            ", datasetDirPath='" + datasetDirPath + '\'' +
            ", datasetName='" + datasetName + '\'' +
            ", classifierParameterStrs=" + classifierParameterStrs +
            ", classifierParameters=" + classifierParameters +
            ", appendClassifierParameters=" + appendClassifierParameters +
            ", trainTimeContracts=" + trainTimeContracts +
            ", trainMemoryContracts=" + trainMemoryContracts +
            ", testTimeContracts=" + testTimeContracts +
            ", checkpoint=" + checkpoint +
            ", appendTrainTimeContract=" + appendTrainTimeContract +
            ", appendTrainMemoryContract=" + appendTrainMemoryContract +
            ", appendTestTimeContract=" + appendTestTimeContract +
            ", estimateTrainError=" + estimateTrainError +
            ", classifierVerbosityStr='" + classifierVerbosityStr + '\'' +
            ", classifierVerbosityLevel=" + classifierVerbosityLevel +
            ", experimentVerbosityStr='" + experimentVerbosityStr + '\'' +
            ", experimentVerbosityLevel=" + experimentVerbosityLevel +
            ", overwriteTrain=" + overwriteTrain +
            ", overwriteTest=" + overwriteTest +
            ", classifierBuilderFactory=" + classifierBuilderFactory +
            '}';
    }

    public ExperimentArgs(final String... args) throws Exception {
        parse(args);
    }

    public void parse(String... args) throws Exception {
        JCommander.newBuilder()
            .addObject(this)
            .build()
            .parse(args);
        // perform custom parsing here
        parseClassifierParameters();
        parseExperimentLogLevel();
        parseClassifierLogLevel();
        parseTrainMemoryContracts();
        parseTrainTimeContracts();
        parseTestTimeContracts();
    }

    private void parseExperimentLogLevel() {
        if(experimentVerbosityStr != null) {
            experimentVerbosityLevel = Level.parse(experimentVerbosityStr);
        }
    }

    private void parseClassifierLogLevel() {
        if(classifierVerbosityStr != null) {
            classifierVerbosityLevel = Level.parse(classifierVerbosityStr);
        }
    }

    private void parseTrainTimeContracts() {
        Collections.sort(trainTimeContracts);
        // if there's no train contract we'll put nulls in place. This causes the loop to fire and we'll handle nulls
        // as no contract inside the loop
        if(trainTimeContracts.isEmpty()) {
            trainTimeContracts.add(null);
        }
    }

    private void parseTestTimeContracts() {
        Collections.sort(testTimeContracts);
        if(testTimeContracts.isEmpty()) {
            testTimeContracts.add(null);
        }
    }

    private void parseTrainMemoryContracts() {
        Collections.sort(trainMemoryContracts);
        if(trainMemoryContracts.isEmpty()) {
            trainMemoryContracts.add(null);
        }
    }

    private void parseClassifierParameters() throws Exception {
        classifierParameters = new ParamSet();
        CollectionUtils.forEachPair(classifierParameterStrs, new BiConsumer<String, String>() {
            @Override
            public void accept(final String key, final String value) {
                try {
                    classifierParameters.setOptions(key, value);
                } catch(Exception e) {
                    throw new IllegalStateException(e);
                }
            }
        });
    }

    public static String getClassifierShortFlag() {
        return CLASSIFIER_SHORT_FLAG;
    }

    public static String getClassifierLongFlag() {
        return CLASSIFIER_LONG_FLAG;
    }

    public String getClassifierName() {
        return classifierName;
    }

    public void setClassifierName(final String classifierName) {
        this.classifierName = classifierName;
    }

    public static String getSeedShortFlag() {
        return SEED_SHORT_FLAG;
    }

    public static String getSeedLongFlag() {
        return SEED_LONG_FLAG;
    }

    public Integer getSeed() {
        return seed;
    }

    public void setSeed(final Integer seed) {
        this.seed = seed;
    }

    public static String getResultsDirShortFlag() {
        return RESULTS_DIR_SHORT_FLAG;
    }

    public static String getResultsDirLongFlag() {
        return RESULTS_DIR_LONG_FLAG;
    }

    public String getResultsDirPath() {
        return resultsDirPath;
    }

    public void setResultsDirPath(final String resultsDirPath) {
        this.resultsDirPath = resultsDirPath;
    }

    public static String getDatasetDirShortFlag() {
        return DATASET_DIR_SHORT_FLAG;
    }

    public static String getDatasetDirLongFlag() {
        return DATASET_DIR_LONG_FLAG;
    }

    public String getDatasetDirPath() {
        return datasetDirPath;
    }

    public void setDatasetDirPath(final String datasetDirPath) {
        this.datasetDirPath = datasetDirPath;
    }

    public static String getDatasetNameShortFlag() {
        return DATASET_NAME_SHORT_FLAG;
    }

    public static String getDatasetNameLongFlag() {
        return DATASET_NAME_LONG_FLAG;
    }

    public String getDatasetName() {
        return datasetName;
    }

    public void setDatasetName(final String datasetName) {
        this.datasetName = datasetName;
    }

    public static String getParametersShortFlag() {
        return PARAMETERS_SHORT_FLAG;
    }

    public static String getParametersLongFlag() {
        return PARAMETERS_LONG_FLAG;
    }

    public List<String> getClassifierParameterStrs() {
        return classifierParameterStrs;
    }

    public void setClassifierParameterStrs(final List<String> classifierParameterStrs) {
        this.classifierParameterStrs = classifierParameterStrs;
    }

    public ParamSet getClassifierParameters() {
        return classifierParameters;
    }

    public void setClassifierParameters(final ParamSet classifierParameters) {
        this.classifierParameters = classifierParameters;
    }

    public static String getAppendClassifierParametersShortFlag() {
        return APPEND_CLASSIFIER_PARAMETERS_SHORT_FLAG;
    }

    public static String getAppendClassifierParametersLongFlag() {
        return APPEND_CLASSIFIER_PARAMETERS_LONG_FLAG;
    }

    public boolean isAppendClassifierParameters() {
        return appendClassifierParameters;
    }

    public void setAppendClassifierParameters(final boolean appendClassifierParameters) {
        this.appendClassifierParameters = appendClassifierParameters;
    }

    public static String getTrainTimeContractShortFlag() {
        return TRAIN_TIME_CONTRACT_SHORT_FLAG;
    }

    public static String getTrainTimeContractLongFlag() {
        return TRAIN_TIME_CONTRACT_LONG_FLAG;
    }

    public List<TimeAmount> getTrainTimeContracts() {
        return trainTimeContracts;
    }

    public void setTrainTimeContracts(
        final List<TimeAmount> trainTimeContracts) {
        this.trainTimeContracts = trainTimeContracts;
    }

    public static String getTrainMemoryContractShortFlag() {
        return TRAIN_MEMORY_CONTRACT_SHORT_FLAG;
    }

    public static String getTrainMemoryContractLongFlag() {
        return TRAIN_MEMORY_CONTRACT_LONG_FLAG;
    }

    public List<MemoryAmount> getTrainMemoryContracts() {
        return trainMemoryContracts;
    }

    public void setTrainMemoryContracts(
        final List<MemoryAmount> trainMemoryContracts) {
        this.trainMemoryContracts = trainMemoryContracts;
    }

    public static String getTestTimeContractShortFlag() {
        return TEST_TIME_CONTRACT_SHORT_FLAG;
    }

    public static String getTestTimeContractLongFlag() {
        return TEST_TIME_CONTRACT_LONG_FLAG;
    }

    public List<TimeAmount> getTestTimeContracts() {
        return testTimeContracts;
    }

    public void setTestTimeContracts(
        final List<TimeAmount> testTimeContracts) {
        this.testTimeContracts = testTimeContracts;
    }

    public static String getCheckpointShortFlag() {
        return CHECKPOINT_SHORT_FLAG;
    }

    public static String getCheckpointLongFlag() {
        return CHECKPOINT_LONG_FLAG;
    }

    public boolean isCheckpoint() {
        return checkpoint;
    }

    public void setCheckpoint(final boolean checkpoint) {
        this.checkpoint = checkpoint;
    }

    public static String getCheckpointIntervalShortFlag() {
        return CHECKPOINT_INTERVAL_SHORT_FLAG;
    }

    public static String getCheckpointIntervalLongFlag() {
        return CHECKPOINT_INTERVAL_LONG_FLAG;
    }

    public static String getAppendTrainTimeContractShortFlag() {
        return APPEND_TRAIN_TIME_CONTRACT_SHORT_FLAG;
    }

    public static String getAppendTrainTimeContractLongFlag() {
        return APPEND_TRAIN_TIME_CONTRACT_LONG_FLAG;
    }

    public boolean isAppendTrainTimeContract() {
        return appendTrainTimeContract;
    }

    public void setAppendTrainTimeContract(final boolean appendTrainTimeContract) {
        this.appendTrainTimeContract = appendTrainTimeContract;
    }

    public static String getAppendTrainMemoryContractShortFlag() {
        return APPEND_TRAIN_MEMORY_CONTRACT_SHORT_FLAG;
    }

    public static String getAppendTrainMemoryContractLongFlag() {
        return APPEND_TRAIN_MEMORY_CONTRACT_LONG_FLAG;
    }

    public boolean isAppendTrainMemoryContract() {
        return appendTrainMemoryContract;
    }

    public void setAppendTrainMemoryContract(final boolean appendTrainMemoryContract) {
        this.appendTrainMemoryContract = appendTrainMemoryContract;
    }

    public static String getAppendTestTimeContractShortFlag() {
        return APPEND_TEST_TIME_CONTRACT_SHORT_FLAG;
    }

    public static String getAppendTestTimeContractLongFlag() {
        return APPEND_TEST_TIME_CONTRACT_LONG_FLAG;
    }

    public boolean isAppendTestTimeContract() {
        return appendTestTimeContract;
    }

    public void setAppendTestTimeContract(final boolean appendTestTimeContract) {
        this.appendTestTimeContract = appendTestTimeContract;
    }

    public static String getEstimateTrainErrorShortFlag() {
        return ESTIMATE_TRAIN_ERROR_SHORT_FLAG;
    }

    public static String getEstimateTrainErrorLongFlag() {
        return ESTIMATE_TRAIN_ERROR_LONG_FLAG;
    }

    public boolean isEstimateTrainError() {
        return estimateTrainError;
    }

    public void setEstimateTrainError(final boolean estimateTrainError) {
        this.estimateTrainError = estimateTrainError;
    }

    public static String getClassifierVerbosityShortFlag() {
        return CLASSIFIER_VERBOSITY_SHORT_FLAG;
    }

    public static String getClassifierVerbosityLongFlag() {
        return CLASSIFIER_VERBOSITY_LONG_FLAG;
    }

    public String getClassifierVerbosityStr() {
        return classifierVerbosityStr;
    }

    public void setClassifierVerbosityStr(final String classifierVerbosityStr) {
        this.classifierVerbosityStr = classifierVerbosityStr;
    }

    public Level getClassifierVerbosityLevel() {
        return classifierVerbosityLevel;
    }

    public void setClassifierVerbosityLevel(final Level classifierVerbosityLevel) {
        this.classifierVerbosityLevel = classifierVerbosityLevel;
    }

    public static String getExperimentVerbosityShortFlag() {
        return EXPERIMENT_VERBOSITY_SHORT_FLAG;
    }

    public static String getExperimentVerbosityLongFlag() {
        return EXPERIMENT_VERBOSITY_LONG_FLAG;
    }

    public String getExperimentVerbosityStr() {
        return experimentVerbosityStr;
    }

    public void setExperimentVerbosityStr(final String experimentVerbosityStr) {
        this.experimentVerbosityStr = experimentVerbosityStr;
    }

    public Level getExperimentVerbosityLevel() {
        return experimentVerbosityLevel;
    }

    public void setExperimentVerbosityLevel(final Level experimentVerbosityLevel) {
        this.experimentVerbosityLevel = experimentVerbosityLevel;
    }

    public static String getOverwriteTrainShortFlag() {
        return OVERWRITE_TRAIN_SHORT_FLAG;
    }

    public static String getOverwriteTrainLongFlag() {
        return OVERWRITE_TRAIN_LONG_FLAG;
    }

    public boolean isOverwriteTrain() {
        return overwriteTrain;
    }

    public void setOverwriteTrain(final boolean overwriteTrain) {
        this.overwriteTrain = overwriteTrain;
    }

    public static String getOverwriteTestShortFlag() {
        return OVERWRITE_TEST_SHORT_FLAG;
    }

    public static String getOverwriteTestLongFlag() {
        return OVERWRITE_TEST_LONG_FLAG;
    }

    public boolean isOverwriteTest() {
        return overwriteTest;
    }

    public void setOverwriteTest(final boolean overwriteTest) {
        this.overwriteTest = overwriteTest;
    }

    public ClassifierBuilderFactory<Classifier> getClassifierBuilderFactory() {
        return classifierBuilderFactory;
    }

    public void setClassifierBuilderFactory(
        final ClassifierBuilderFactory<Classifier> classifierBuilderFactory) {
        this.classifierBuilderFactory = classifierBuilderFactory;
    }
}
