package tsml.classifiers.distance_based.utils.experiments;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.internal.Lists;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.function.Function;
import java.util.logging.Level;
import java.util.logging.Logger;
import tsml.classifiers.distance_based.utils.classifier_building.ClassifierBuilderFactory;
import tsml.classifiers.distance_based.utils.classifier_building.ClassifierBuilderFactory.ClassifierBuilder;
import tsml.classifiers.distance_based.utils.collections.CollectionUtils;
import tsml.classifiers.distance_based.utils.collections.box.Box;
import tsml.classifiers.distance_based.utils.logging.LogUtils;
import tsml.classifiers.distance_based.utils.system.memory.MemoryAmount;
import tsml.classifiers.distance_based.utils.system.parallel.BlockingExecutor;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.stopwatch.TimeAmount;
import utilities.FileUtils;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Purpose: class to run a batch of experiments
 * <p>
 * Contributors: goastler
 */

public class ExperimentBatch {
    // todo use getters and setters internally

    // abide by unix cmdline args convention! single char --> single hyphen, multiple chars --> double hyphen
    // todo getter sna setters
    private final Logger logger = LogUtils.buildLogger(this);
    public static final String SHORT_CLASSIFIER_NAME_FLAG = "-c";
    public static final String LONG_CLASSIFIER_NAME_FLAG = "--classifier";
    @Parameter(names = {SHORT_CLASSIFIER_NAME_FLAG, LONG_CLASSIFIER_NAME_FLAG}, description = "todo",
        required = true)
    private List<String> classifierNames = new ArrayList<>();
    public static final String SHORT_DATASET_DIR_FLAG = "--dd";
    public static final String LONG_DATASET_DIR_FLAG = "--datasetsDir";
    @Parameter(names = {SHORT_DATASET_DIR_FLAG, LONG_DATASET_DIR_FLAG}, description = "todo", required = true)
    private List<String> datasetDirPaths = new ArrayList<>();
    public static final String SHORT_DATASET_NAME_FLAG = "-d";
    public static final String LONG_DATASET_NAME_FLAG = "--dataset";
    @Parameter(names = {SHORT_DATASET_NAME_FLAG, LONG_DATASET_NAME_FLAG}, description = "todo", required = true)
    private List<String> datasetNames = new ArrayList<>();
    public static final String SHORT_SEED_FLAG = "-s";
    public static final String LONG_SEED_FLAG = "--seed";
    @Parameter(names = {SHORT_SEED_FLAG, LONG_SEED_FLAG}, description = "todo", required = true)
    private List<Integer> seeds = Lists.newArrayList(0);
    public static final String SHORT_RESULTS_DIR_FLAG = "-r";
    public static final String LONG_RESULTS_DIR_FLAG = "--resultsDir";
    @Parameter(names = {SHORT_RESULTS_DIR_FLAG, LONG_RESULTS_DIR_FLAG}, description = "todo", required = true)
    private String resultsDirPath = null;
    public static final String SHORT_PARAMETERS_FLAG = "-p";
    public static final String LONG_PARAMETERS_FLAG = "--parameters";
    @Parameter(names = {SHORT_PARAMETERS_FLAG, LONG_PARAMETERS_FLAG}, description = "todo", variableArity = true)
    private List<String> classifierParameterStrs = new ArrayList<>();
    private Map<String, ParamSet> bespokeClassifierParametersMap = new HashMap<>();
    private ParamSet universalClassifierParameters = new ParamSet();
    public static final String SHORT_APPEND_CLASSIFIER_PARAMETERS_FLAG = "--acp";
    public static final String LONG_APPEND_CLASSIFIER_PARAMETERS_FLAG = "--appendClassifierParameters";
    @Parameter(names = {SHORT_APPEND_CLASSIFIER_PARAMETERS_FLAG, LONG_APPEND_CLASSIFIER_PARAMETERS_FLAG}, description = "todo")
    private boolean appendClassifierParameters = false;
    public static final String SHORT_TRAIN_TIME_CONTRACT_FLAG = "--ttc";
    public static final String LONG_TRAIN_TIME_CONTRACT_FLAG = "--trainTimeContract";
    @Parameter(names = {SHORT_TRAIN_TIME_CONTRACT_FLAG, LONG_TRAIN_TIME_CONTRACT_FLAG}, arity = 2, description = "todo")
    private List<String> trainTimeContractStrs = new ArrayList<>();
    private List<TimeAmount> trainTimeContracts = new ArrayList<>();
    public static final String SHORT_TRAIN_MEMORY_CONTRACT_FLAG = "--tmc";
    public static final String LONG_TRAIN_MEMORY_CONTRACT_FLAG = "--trainMemoryContract";
    @Parameter(names = {SHORT_TRAIN_MEMORY_CONTRACT_FLAG, LONG_TRAIN_MEMORY_CONTRACT_FLAG}, arity = 2, description = "todo")
    private List<String> trainMemoryContractStrs = new ArrayList<>();
    private List<MemoryAmount> trainMemoryContracts = new ArrayList<>();
    public static final String SHORT_TEST_TIME_CONTRACT_FLAG = "--ptc";
    public static final String LONG_TEST_TIME_CONTRACT_FLAG = "--testTimeContract";
    @Parameter(names = {SHORT_TEST_TIME_CONTRACT_FLAG, LONG_TEST_TIME_CONTRACT_FLAG}, arity = 2, description = "todo")
    private List<String> testTimeContractStrs = new ArrayList<>();
    private List<TimeAmount> testTimeContracts = new ArrayList<>();
    public static final String SHORT_CHECKPOINT_FLAG = "--cp";
    public static final String LONG_CHECKPOINT_FLAG = "--checkpoint";
    @Parameter(names = {SHORT_CHECKPOINT_FLAG, LONG_CHECKPOINT_FLAG}, description = "todo")
    private boolean checkpoint = false; // todo swap train contract save / load path over for checkpointing
    public static final String SHORT_THREADS_FLAG = "-t";
    public static final String LONG_THREADS_FLAG = "--threads";
    @Parameter(names = {SHORT_THREADS_FLAG, LONG_THREADS_FLAG}, description = "todo")
    private int numThreads = 1; // <=0 for all available threads
    public static final String SHORT_APPEND_TRAIN_TIME_CONTRACT_FLAG = "--attc";
    public static final String LONG_APPEND_TRAIN_TIME_CONTRACT_FLAG = "--appendTrainTimeContract";
    @Parameter(names = {SHORT_APPEND_TRAIN_TIME_CONTRACT_FLAG, LONG_APPEND_TRAIN_TIME_CONTRACT_FLAG}, description = "todo")
    private boolean appendTrainTimeContract = false;
    public static final String SHORT_APPEND_TRAIN_MEMORY_CONTRACT_FLAG = "--atmc";
    public static final String LONG_APPEND_TRAIN_MEMORY_CONTRACT_FLAG = "--appendTrainMemoryContract";
    @Parameter(names = {SHORT_APPEND_TRAIN_MEMORY_CONTRACT_FLAG, LONG_APPEND_TRAIN_MEMORY_CONTRACT_FLAG}, description = "todo")
    private boolean appendTrainMemoryContract = false;
    public static final String SHORT_APPEND_TEST_TIME_CONTRACT_FLAG = "--aptc";
    public static final String LONG_APPEND_TEST_TIME_CONTRACT_FLAG = "--appendTestTimeContract";
    @Parameter(names = {SHORT_APPEND_TEST_TIME_CONTRACT_FLAG, LONG_APPEND_TEST_TIME_CONTRACT_FLAG}, description =
        "todo")
    private boolean appendTestTimeContract = false;
    public static final String SHORT_ESTIMATE_TRAIN_ERROR_FLAG = "-e";
    public static final String LONG_ESTIMATE_TRAIN_ERROR_FLAG = "--estimateTrainError";
    @Parameter(names = {SHORT_ESTIMATE_TRAIN_ERROR_FLAG, LONG_ESTIMATE_TRAIN_ERROR_FLAG}, description = "todo")
    private boolean estimateTrainError = false;
    public static final String SHORT_LOG_LEVEL_CLASSIFIER_FLAG = "-l";
    public static final String LONG_LOG_LEVEL_CLASSIFIER_FLAG = "--logLevel";
    @Parameter(names = {SHORT_LOG_LEVEL_CLASSIFIER_FLAG, LONG_LOG_LEVEL_CLASSIFIER_FLAG}, description = "todo")
    private String logLevelClassifier = Level.SEVERE.toString(); // todo better name for these two
    public static final String SHORT_LOG_LEVEL_EXPERIMENT_FLAG = "-el";
    public static final String LONG_LOG_LEVEL_EXPERIMENT_FLAG = "--experimentLogLevel";
    @Parameter(names = {SHORT_LOG_LEVEL_EXPERIMENT_FLAG, LONG_LOG_LEVEL_EXPERIMENT_FLAG}, description = "todo")
    private String logLevelExperiment = Level.ALL.toString();
    public static final String SHORT_OVERWRITE_TRAIN_FLAG = "--ot";
    public static final String LONG_OVERWRITE_TRAIN_FLAG = "--overwriteTrain";
    @Parameter(names = {SHORT_OVERWRITE_TRAIN_FLAG, LONG_OVERWRITE_TRAIN_FLAG}, description = "todo")
    private boolean overwriteTrain = false;
    public static final String SHORT_OVERWRITE_TEST_FLAG = "--op";
    public static final String LONG_OVERWRITE_TEST_FLAG = "--overwriteTest";
    @Parameter(names = {SHORT_OVERWRITE_TEST_FLAG, LONG_OVERWRITE_TEST_FLAG}, description = "todo")
    private boolean overwriteTest = false;
    private ClassifierBuilderFactory<Classifier> classifierBuilderFactory =
        ClassifierBuilderFactory.getGlobalInstance(); // todo get this by string, i.e. factory

    public ExperimentBatch(String... args) throws Exception {
        parse(args);
    }

    public static void main(String... args) throws Exception {
        new ExperimentBatch(args).runExperiments();
    }

    @Override
    public String toString() {
        return "ExperimentBatch{" +
            "classifierNames=" + classifierNames +
            ", datasetDirPaths=" + datasetDirPaths +
            ", datasetNames=" + datasetNames +
            ", seeds=" + seeds +
            ", resultsDirPath='" + resultsDirPath + '\'' +
            ", classifierParameterStrs=" + classifierParameterStrs +
            ", bespokeClassifierParametersMap=" + bespokeClassifierParametersMap +
            ", universalClassifierParameters=" + universalClassifierParameters +
            ", appendClassifierParameters=" + appendClassifierParameters +
            ", trainTimeContractStrs=" + trainTimeContractStrs +
            ", trainTimeContracts=" + trainTimeContracts +
            ", trainMemoryContractStrs=" + trainMemoryContractStrs +
            ", trainMemoryContracts=" + trainMemoryContracts +
            ", testTimeContractStrs=" + testTimeContractStrs +
            ", testTimeContracts=" + testTimeContracts +
            ", checkpoint=" + checkpoint +
            ", numThreads=" + numThreads +
            ", appendTrainTimeContract=" + appendTrainTimeContract +
            ", appendTrainMemoryContract=" + appendTrainMemoryContract +
            ", appendTestTimeContract=" + appendTestTimeContract +
            ", estimateTrainError=" + estimateTrainError +
            ", logLevelClassifier='" + logLevelClassifier + '\'' +
            ", logLevelExperiment='" + logLevelExperiment + '\'' +
            ", overwriteTrain=" + overwriteTrain +
            ", overwriteTest=" + overwriteTest +
            ", classifierBuilderFactory=" + classifierBuilderFactory +
            '}';
    }

    public static List<TimeAmount> convertStringPairsToTimeAmounts(List<String> strs) {
        List<TimeAmount> times = CollectionUtils.convertPairs(strs, TimeAmount::parse);
        Collections.sort(times);
        return times;
    }

    public static List<MemoryAmount> convertStringPairsToMemoryAmounts(List<String> strs) {
        List<MemoryAmount> times = CollectionUtils.convertPairs(strs, MemoryAmount::parse);
        Collections.sort(times);
        return times;
    }

    public void parse(String... args) throws Exception {
        JCommander.newBuilder()
            .addObject(this)
            .build()
            .parse(args);
        // perform custom parsing here
        parseClassifierParameters();
        parseLogLevel();
        parseTrainMemoryContracts();
        parseTrainTimeContracts();
        parseTestTimeContracts();
    }

    private void parseLogLevel() {
        logger.finest("parsing {" + logLevelExperiment + "} as logLevel");
        if(logLevelExperiment != null) {
            logger.setLevel(Level.parse(logLevelExperiment));
        }
    }

    private void parseTrainTimeContracts() {
        if(trainTimeContractStrs.size() % 2 != 0) {
            throw new IllegalStateException("train time contracts must be a list of pairs, i.e. \"5\" \"minutes\"");
        }
        trainTimeContracts = convertStringPairsToTimeAmounts(trainTimeContractStrs);
        // if there's no train contract we'll put nulls in place. This causes the loop to fire and we'll handle nulls
        // as no contract inside the loop
        if(trainTimeContracts.isEmpty()) {
            trainTimeContracts.add(null);
        }
        logger.finest("parsed trainTimeContracts as {" + trainTimeContracts + "}");
    }

    private void parseTestTimeContracts() {
        if(testTimeContractStrs.size() % 2 != 0) {
            throw new IllegalStateException("test time contracts must be a list of pairs, i.e. \"5\" \"minutes\"");
        }
        testTimeContracts = convertStringPairsToTimeAmounts(testTimeContractStrs);
        if(testTimeContracts.isEmpty()) {
            testTimeContracts.add(null);
        }
        logger.finest("parsed testTimeContracts as {" + testTimeContracts + "}");
    }

    private void parseTrainMemoryContracts() {
        if(trainMemoryContractStrs.size() % 2 != 0) {
            throw new IllegalStateException("train memory contracts must be a list of pairs, i.e. \"5\" \"minutes\"");
        }
        trainMemoryContracts = convertStringPairsToMemoryAmounts(trainMemoryContractStrs);
        if(trainMemoryContracts.isEmpty()) {
            trainMemoryContracts.add(null);
        }
        logger.finest("parsed trainMemoryContracts as {" + trainMemoryContracts + "}");
    }

    private void parseClassifierParameters() throws Exception {
        bespokeClassifierParametersMap = new HashMap<>();
        universalClassifierParameters = new ParamSet();
        Set<String> classifierNamesLookup = new HashSet<>(classifierNames);
        for(int i = 0; i < classifierParameterStrs.size(); ) {
            String str = classifierParameterStrs.get(i++);
            String parameterName;
            String parameterValue;
            ParamSet paramSet;
            if(classifierNamesLookup.contains(str)) {
                // we're in a 3 arity situation, i.e. <classifier name> <parameter name> <parameter value>
                parameterName = classifierParameterStrs.get(i++);
                parameterValue = classifierParameterStrs.get(i++);
                paramSet = bespokeClassifierParametersMap.computeIfAbsent(str, k -> new ParamSet());
            } else {
                // we're in a 2 arity situation, i.e. <parameter name> <parameter value>
                parameterName = str;
                parameterValue = classifierParameterStrs.get(i++);
                paramSet = universalClassifierParameters;
            }
            paramSet.setOptions(parameterName, parameterValue);
        }
        logger.finest("parsed bespokeClassifierParametersMap as {" + bespokeClassifierParametersMap + "}");
        logger.finest("parsed universalClassifierParametersMap as {" + universalClassifierParameters + "}");
    }

    private Instances[] loadData(String name, int seed) {
        List<Exception> exceptions = new ArrayList<>();
        for(final String path : datasetDirPaths) {
            try {
                Instances[] data = DatasetLoading.sampleDataset(path, name, seed);
                if(data == null) {
                    throw new Exception();
                }
                logger.finest("loaded {" + name + "} from {" + path + "}");
                return data;
            } catch(Exception ignored) {
                exceptions.add(ignored);
            }
        }
        IllegalArgumentException overallException = new IllegalArgumentException("couldn't load data");
        for(Exception exception : exceptions) {
            overallException.addSuppressed(exception);
        }
        throw overallException;
    }

    private ExecutorService buildExecutor() {
        int numThreads = this.numThreads;
        if(numThreads < 1) {
            numThreads = Runtime.getRuntime().availableProcessors();
        }
        ThreadPoolExecutor threadPool = (ThreadPoolExecutor) Executors.newFixedThreadPool(numThreads);
        logger.finest("built BlockingExecutor containing {" + numThreads + "} threads");
        return new BlockingExecutor(threadPool);
    }

    private ParamSet buildParamSetForClassifier(String classifierName) throws Exception { // todo apply
        ParamSet bespokeClassifierParameters = bespokeClassifierParametersMap
            .getOrDefault(classifierName, new ParamSet());
        ParamSet paramSet = new ParamSet();
        paramSet.addAll(bespokeClassifierParameters);
        paramSet.addAll(universalClassifierParameters);
        if(!paramSet.isEmpty()) {
            logger.finest("built ParamSet {" + paramSet + "}");
        }
        return paramSet;
    }

    private boolean runExperimentTest(Experiment experiment) {
        try {
            testExperiment(experiment);
        } catch(Exception e) {
            e.printStackTrace();
        }
        return true;
    }

    private void runExperimentTrain(Experiment experiment) {
        forEachExperimentTrain(experiment, experimentToTrain -> {
            if(shouldTrain(experimentToTrain)) {
                try {
                    trainExperiment(experiment);
                    if(shouldTest(experiment)) {
                        forEachExperimentTest(experiment, this::runExperimentTest);
                    }
                } catch(Exception e) {
                    e.printStackTrace();
                }
            }
            // shallow copy experiment so we can reuse the configuration under the next train contract
            //                experiment = (Experiment) experiment.shallowCopy(); // todo is this needed?
            return true;
        });
    }

    private Runnable buildExperimentTask(Experiment experiment) {
        return () -> {
            try {
                runExperimentTrain(experiment);
            } catch(Exception e) {
                e.printStackTrace();
            }
        };
    }

    public void runExperiments() {
        logger.info("experiments config: " + this);
        ExecutorService executor = buildExecutor();
        for(final int seed : seeds) {
            for(final String datasetName : datasetNames) {
                Instances[] data = null;
                try {
                    data = loadData(datasetName, seed);
                } catch(Exception e) {
                    logger.severe(e.toString());
                    continue;
                }
                final Instances trainData = data[0];
                final Instances testData = data[1];
                for(final String classifierName : classifierNames) {
                    try {
                        final ClassifierBuilder<? extends Classifier> classifierBuilder = classifierBuilderFactory
                            .getClassifierBuilderByName(classifierName);
                        if(classifierBuilder == null) {
                            logger.severe("no classifier by the name of {" + classifierName + "}, skipping experiment");
                            continue;
                        }
                        final Classifier classifier = classifierBuilder.build();
                        final Experiment experiment = new Experiment(trainData, testData, classifier, seed, classifierName, datasetName);
                        experiment.getLogger().setLevel(Level.parse(logLevelClassifier));
                        experiment.setEstimateTrainError(estimateTrainError);
                        executor.submit(buildExperimentTask(experiment));
                    } catch(Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        }
        executor.shutdown();
    }

    private void applyTrainTimeContract(Experiment experiment, TimeAmount trainTimeContract) {
        // setup the next train contract
        String classifierName = experiment.getClassifierName();
        if(trainTimeContract != null) {
            logger.info(
                "train time contract of {" + trainTimeContract + "} for {" + experiment.getClassifierName() + "} on {"
                    + experiment.getDatasetName() + "}");
            experiment.setTrainTimeLimit(trainTimeContract.getAmount(), trainTimeContract.getUnit()); // todo add this to
            // the interface, overload
            if(appendTrainTimeContract) {
                experiment
                    .setClassifierName(classifierName + "_" + trainTimeContract.toString().replaceAll(" ", "_"));
            }
        } else {
            // no train contract
            // todo set train contract disabled somehow? Tony set some boolean in the api somewhere, see if
            //  that'll do
        }
    }

    private void applyTrainMemoryContract(Experiment experiment, MemoryAmount trainMemoryContract) {
        // setup the next train contract
        String classifierName = experiment.getClassifierName();
        if(trainMemoryContract != null) {
            logger.info(
                "train memory contract of {" + trainMemoryContract + "} for {" + experiment.getClassifierName() + "} "
                    + "on {"
                    + experiment.getDatasetName() + "}");
//            experiment.setMemoryLimit(trainMemoryContract.getAmount(), trainMemoryContract.getUnit()); // todo add this to
            // the interface, overload
            if(appendTrainMemoryContract) {
                experiment
                    .setClassifierName(classifierName + "_" + trainMemoryContract.toString().replaceAll(" ", "_"));
            }
        } else {
            // no train contract
        }
    }

    private void applyTestTimeContract(Experiment experiment, TimeAmount testTimeContract) {
        // setup the next train contract
        String classifierName = experiment.getClassifierName();
        if(testTimeContract != null) {
            logger.info(
                "test time contract of {" + testTimeContract + "} for {" + experiment.getClassifierName() + "} "
                    + "on {"
                    + experiment.getDatasetName() + "}");
            experiment.setTestTimeLimit(testTimeContract.getUnit(), testTimeContract.getAmount());
            // todo add this to
            // the interface, overload
            if(appendTestTimeContract) {
                experiment
                    .setClassifierName(classifierName + "_" + testTimeContract.toString().replaceAll(" ", "_"));
            }
        } else {
            // no train contract
        }
    }

    private void appendParametersToClassifierName(Experiment experiment) {
        if(appendClassifierParameters) {
            String origClassifierName = experiment.getClassifierName();
            String workingClassifierName = origClassifierName;
            ParamSet paramSet = experiment.getParamSet();
            String paramSetStr = "_" + paramSet.toString().replaceAll(" ", "_").replaceAll("\"", "#");
            workingClassifierName += paramSetStr;
            experiment.setClassifierName(workingClassifierName);
            logger.info("changing {" + origClassifierName + "} to {" + workingClassifierName + "}");
        }
    }

    private String buildTrainResultsFilePath(Experiment experiment) {
        return buildClassifierResultsDirPath(experiment) + "trainFold" + experiment.getSeed() +
            ".csv";
    }

    private String buildTestResultsFilePath(Experiment experiment) {
        return buildClassifierResultsDirPath(experiment) + "testFold" + experiment.getSeed() +
            ".csv";
    }

    private boolean shouldTest(Experiment experiment) {
        return shouldTest(experiment, true);
    }

    private boolean shouldTest(Experiment experiment, boolean log) {
        if(isOverwriteTest()) {
            if(log) logger.finest("overwriting test results for {" + experiment.getClassifierName() + "} on {" + experiment.getDatasetName() + "}");
            return true;
        }
        String path = buildTestResultsFilePath(experiment);
        boolean result = !new File(path).exists();
        if(result) {
            if(log) logger.finest("non-existent test results for {" + experiment.getClassifierName() + "} on {" + experiment.getDatasetName() + "}");
        } else {
            if(log) logger.info("existing test results for {" + experiment.getClassifierName() + "} on {" + experiment.getDatasetName() + "}");
        }
        return result;
    }

    private boolean shouldTrain(Experiment experiment) {
        Box<Boolean> box = new Box<>(false);
        forEachExperimentTest(experiment, experiment1 -> {
            boolean shouldTest = shouldTest(experiment, false);
            box.set(shouldTest);
            return !shouldTest; // if we've found a test which needs to be performed then stop
        });
        // if there are tests to be performed then we must train
        if(box.get()) {
            logger.finest("testing required so overwriting train results for {" + experiment.getClassifierName() + "} "
                + "on {" + experiment.getDatasetName() + "}");
            return true;
        }
        if(!isEstimateTrainError()) {
            logger.finest("not estimating train error for {" + experiment.getClassifierName() + "} on {" + experiment.getDatasetName() + "}");
            return false;
        }
        if(isOverwriteTrain()) {
            logger.finest("overwriting train results for {" + experiment.getClassifierName() + "} on {" + experiment.getDatasetName() + "}");
            return true;
        }
        String path = buildTrainResultsFilePath(experiment);
        boolean exists = new File(path).exists();
        if(exists) {
            logger.finest("existing train results for {" + experiment.getClassifierName() + "} on {" + experiment.getDatasetName() + "}");
        } else {
            logger.finest("non-existent train results for {" + experiment.getClassifierName() + "} on {" + experiment.getDatasetName() + "}");
        }
        return !exists;
    }

    private void trainExperiment(Experiment experiment) throws Exception {
        // train classifier
        logger.info("training {" + experiment.getClassifierName() + "} on {" + experiment.getDatasetName() + "}");
        experiment.train();
        // write train results if enabled
        if(experiment.isEstimateTrainError()) {
            logger.info("finding train results for {" + experiment.getClassifierName() + "} on {" + experiment.getDatasetName() + "}");
            final String trainResultsFilePath = buildTrainResultsFilePath(experiment);
            final ClassifierResults trainResults = experiment.getTrainResults();
            logger.info("writing train results for {" + experiment.getClassifierName() + "} on {" + experiment.getDatasetName() + "} to {" + trainResultsFilePath + "}");
            FileUtils.writeToFile(trainResults.writeFullResultsToString(), trainResultsFilePath);
        }
    }

    private void forEachExperimentTrain(Experiment experiment, Function<Experiment, Boolean> function) {
        appendParametersToClassifierName(experiment);
        // for each train contract (pair of strs, one for the amount, one for the unit)
        for(MemoryAmount trainMemoryContract : trainMemoryContracts) {
            applyTrainMemoryContract(experiment, trainMemoryContract);
            for(TimeAmount trainTimeContract : trainTimeContracts) {
                applyTrainTimeContract(experiment, trainTimeContract);
                if(!function.apply(experiment)) {
                    return;
                }
            }
        }
    }

    private void forEachExperimentTest(Experiment experiment, Function<Experiment, Boolean> function) {
        for(TimeAmount testTimeContract : testTimeContracts) {
            applyTestTimeContract(experiment, testTimeContract);
            if(shouldTest(experiment)) {
                if(!function.apply(experiment)) {
                    return;
                }
            }
        }
    }

    private void testExperiment(Experiment experiment) throws Exception {
        // test classifier
        logger.info("testing {" + experiment.getClassifierName() + "} on {" + experiment.getDatasetName() + "}");
        experiment.test();
        // write test results
        final String testResultsFilePath = buildTestResultsFilePath(experiment);
        final ClassifierResults testResults = experiment.getTestResults();
        logger.info("writing test results for {" + experiment.getClassifierName() + "} on {" + experiment.getDatasetName() + "} to {" + testResultsFilePath + "}");
        FileUtils.writeToFile(testResults.writeFullResultsToString(), testResultsFilePath);
    }

    private String buildClassifierResultsDirPath(Experiment experiment) {
        return resultsDirPath + "/" + experiment.getClassifierName() + "/Predictions/" + experiment.getDatasetName() +
            "/";
    }

    public List<String> getClassifierNames() {
        return classifierNames;
    }

    public void setClassifierNames(final List<String> classifierNames) {
        this.classifierNames = classifierNames;
    }

    public List<String> getDatasetDirPaths() {
        return datasetDirPaths;
    }

    public void setDatasetDirPaths(final List<String> datasetDirPaths) {
        this.datasetDirPaths = datasetDirPaths;
    }

    public List<String> getDatasetNames() {
        return datasetNames;
    }

    public void setDatasetNames(final List<String> datasetNames) {
        this.datasetNames = datasetNames;
    }

    public List<Integer> getSeeds() {
        return seeds;
    }

    public void setSeeds(final List<Integer> seeds) {
        this.seeds = seeds;
    }

    public String getResultsDirPath() {
        return resultsDirPath;
    }

    public void setResultsDirPath(final String resultsDirPath) {
        this.resultsDirPath = resultsDirPath;
    }

    public List<String> getTrainTimeContractStrs() {
        return trainTimeContractStrs;
    }

    public void setTrainTimeContractStrs(final List<String> trainTimeContractStrs) {
        this.trainTimeContractStrs = trainTimeContractStrs;
    }

    public List<TimeAmount> getTrainTimeContracts() {
        return trainTimeContracts;
    }

    public void setTrainTimeContracts(final List<TimeAmount> trainTimeContracts) {
        this.trainTimeContracts = trainTimeContracts;
    }

    public boolean isCheckpoint() {
        return checkpoint;
    }

    public void setCheckpoint(final boolean checkpoint) {
        this.checkpoint = checkpoint;
    }

    public int getNumThreads() {
        return numThreads;
    }

    public void setNumThreads(final int numThreads) {
        this.numThreads = numThreads;
    }

    public boolean isAppendTrainTimeContract() {
        return appendTrainTimeContract;
    }

    public void setAppendTrainTimeContract(final boolean appendTrainTimeContract) {
        this.appendTrainTimeContract = appendTrainTimeContract;
    }

    public boolean isEstimateTrainError() {
        return estimateTrainError;
    }

    public void setEstimateTrainError(final boolean estimateTrainError) {
        this.estimateTrainError = estimateTrainError;
    }

    public String getLogLevelClassifier() {
        return logLevelClassifier;
    }

    public void setLogLevelClassifier(final String logLevelClassifier) {
        this.logLevelClassifier = logLevelClassifier;
    }

    public Logger getLogger() {
        return logger;
    }

    public ClassifierBuilderFactory<Classifier> getClassifierBuilderFactory() {
        return classifierBuilderFactory;
    }

    public void setClassifierBuilderFactory(
        final ClassifierBuilderFactory<Classifier> classifierBuilderFactory) {
        this.classifierBuilderFactory = classifierBuilderFactory;
    }

    public boolean isOverwriteTrain() {
        return overwriteTrain;
    }

    public void setOverwriteTrain(final boolean overwriteTrain) {
        this.overwriteTrain = overwriteTrain;
    }

    public boolean isOverwriteTest() {
        return overwriteTest;
    }

    public void setOverwriteTest(final boolean overwriteTest) {
        this.overwriteTest = overwriteTest;
    }

    public List<String> getTrainMemoryContractStrs() {
        return trainMemoryContractStrs;
    }

    public void setTrainMemoryContractStrs(final List<String> trainMemoryContractStrs) {
        this.trainMemoryContractStrs = trainMemoryContractStrs;
    }

    public List<MemoryAmount> getTrainMemoryContracts() {
        return trainMemoryContracts;
    }

    public void setTrainMemoryContracts(
        final List<MemoryAmount> trainMemoryContracts) {
        this.trainMemoryContracts = trainMemoryContracts;
    }

    public List<String> getTestTimeContractStrs() {
        return testTimeContractStrs;
    }

    public void setTestTimeContractStrs(final List<String> testTimeContractStrs) {
        this.testTimeContractStrs = testTimeContractStrs;
    }

    public List<TimeAmount> getTestTimeContracts() {
        return testTimeContracts;
    }

    public void setTestTimeContracts(
        final List<TimeAmount> testTimeContracts) {
        this.testTimeContracts = testTimeContracts;
    }

    public boolean isAppendTrainMemoryContract() {
        return appendTrainMemoryContract;
    }

    public void setAppendTrainMemoryContract(final boolean appendTrainMemoryContract) {
        this.appendTrainMemoryContract = appendTrainMemoryContract;
    }

    public boolean isAppendTestTimeContract() {
        return appendTestTimeContract;
    }

    public void setAppendTestTimeContract(final boolean appendTestTimeContract) {
        this.appendTestTimeContract = appendTestTimeContract;
    }

    public List<String> getClassifierParameterStrs() {
        return classifierParameterStrs;
    }

    public ExperimentBatch setClassifierParameterStrs(final List<String> classifierParameterStrs) {
        this.classifierParameterStrs = classifierParameterStrs;
        return this;
    }

    public Map<String, ParamSet> getBespokeClassifierParametersMap() {
        return bespokeClassifierParametersMap;
    }

    public ExperimentBatch setBespokeClassifierParametersMap(
        final Map<String, ParamSet> bespokeClassifierParametersMap) {
        this.bespokeClassifierParametersMap = bespokeClassifierParametersMap;
        return this;
    }

    public ParamSet getUniversalClassifierParameters() {
        return universalClassifierParameters;
    }

    public ExperimentBatch setUniversalClassifierParameters(
        final ParamSet universalClassifierParameters) {
        this.universalClassifierParameters = universalClassifierParameters;
        return this;
    }

    public boolean isAppendClassifierParameters() {
        return appendClassifierParameters;
    }

    public ExperimentBatch setAppendClassifierParameters(final boolean appendClassifierParameters) {
        this.appendClassifierParameters = appendClassifierParameters;
        return this;
    }

    public String getLogLevelExperiment() {
        return logLevelExperiment;
    }

    public ExperimentBatch setLogLevelExperiment(final String logLevelExperiment) {
        this.logLevelExperiment = logLevelExperiment;
        return this;
    }

    public static class Runner {

        public static void main(String[] args) throws Exception { // todo abide by unix cmdline convertion
            ExperimentBatch.main(
                "--threads", "1"
                , "-r", "results"
                , "-s", "0"
                , "-c", "DTW_1NN_V1"
                , "-e"
                , "-d", "GunPoint"
                , "--dd", "/bench/datasets"
                , "-p", "DTW_1NN_V1", "-k", "3"
                //                , "-p", "DTW_1NN_V1","-e","false"
            );
        }
    }
}
