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

    // logger for printing messages
    private final Logger logger = LogUtils.buildLogger(this);

    // names of the classifiers that should be run
    public static final String CLASSIFIER_NAME_SHORT_FLAG = "-c";
    public static final String CLASSIFIER_NAME_LONG_FLAG = "--classifier";
    @Parameter(names = {CLASSIFIER_NAME_SHORT_FLAG, CLASSIFIER_NAME_LONG_FLAG}, description = "the name of the "
        + "classifier to be run",
        required = true)
    private List<String> classifierNames = new ArrayList<>();

    // paths to directories where problem data is stored
    public static final String DATASET_DIR_SHORT_FLAG = "--dd";
    public static final String DATASET_DIR_LONG_FLAG = "--datasetsDir";
    @Parameter(names = {DATASET_DIR_SHORT_FLAG, DATASET_DIR_LONG_FLAG}, description = "the path to the folder "
        + "containing the datasets",
        required = true)
    private List<String> datasetDirPaths = new ArrayList<>();

    // names of the datasets that should be run
    public static final String DATASET_NAME_SHORT_FLAG = "-d";
    public static final String DATASET_NAME_LONG_FLAG = "--dataset";
    @Parameter(names = {DATASET_NAME_SHORT_FLAG, DATASET_NAME_LONG_FLAG}, description = "the name of the dataset",
        required = true)
    private List<String> datasetNames = new ArrayList<>();

    // the seeds to run
    public static final String SEED_SHORT_FLAG = "-s";
    public static final String SEED_LONG_FLAG = "--seed";
    @Parameter(names = {SEED_SHORT_FLAG, SEED_LONG_FLAG}, description = "the seed to be used in sampling a dataset "
        + "and in the random source for the classifier", required = true)
    private List<Integer> seeds = Lists.newArrayList(0);

    // where to put the results when finished
    public static final String RESULTS_DIR_SHORT_FLAG = "-r";
    public static final String RESULTS_DIR_LONG_FLAG = "--resultsDir";
    @Parameter(names = {RESULTS_DIR_SHORT_FLAG, RESULTS_DIR_LONG_FLAG}, description = "path to a folder to place "
        + "results in",
        required = true)
    private String resultsDirPath = null;

    // parameters to pass onto the classifiers. Note, there are two types: universal parameters and bespoke
    // parameters. Universal parameters are applied to all classifiers whereas bespoke parameters are prepended with
    // the name of the classifier to apply those parameters to.
    public static final String PARAMETERS_SHORT_FLAG = "-p";
    public static final String PARAMETERS_LONG_FLAG = "--parameters";
    @Parameter(names = {PARAMETERS_SHORT_FLAG, PARAMETERS_LONG_FLAG}, description = "parameters for the classifiers. "
        + "These can be specified as \"-<parameter_name> <parameter_value>\" (two terms) for all classifiers or "
        + "\"<classifier_name> "
        + "<parameter_name> <parameter_value>\" to apply "
        + "the parameter to only <classifier_name>", variableArity =
        true)
    private List<String> classifierParameterStrs = new ArrayList<>();
    private Map<String, ParamSet> bespokeClassifierParametersMap = new HashMap<>();
    private ParamSet universalClassifierParameters = new ParamSet();

    // whether to append the classifier parameters to the classifier name
    public static final String APPEND_CLASSIFIER_PARAMETERS_SHORT_FLAG = "--acp";
    public static final String APPEND_CLASSIFIER_PARAMETERS_LONG_FLAG = "--appendClassifierParameters";
    @Parameter(names = {APPEND_CLASSIFIER_PARAMETERS_SHORT_FLAG, APPEND_CLASSIFIER_PARAMETERS_LONG_FLAG},
        description = "append the classifier parameters to the classifier name")
    private boolean appendClassifierParameters = false;

    // the train time contract for the classifier
    public static final String TRAIN_TIME_CONTRACT_SHORT_FLAG = "--ttc";
    public static final String TRAIN_TIME_CONTRACT_LONG_FLAG = "--trainTimeContract";
    @Parameter(names = {TRAIN_TIME_CONTRACT_SHORT_FLAG, TRAIN_TIME_CONTRACT_LONG_FLAG}, arity = 2, description =
        "specify a train time contract for the classifier in the form \"<amount> <units>\", e.g. \"4 hour\"")
    private List<String> trainTimeContractStrs = new ArrayList<>();
    private List<TimeAmount> trainTimeContracts = new ArrayList<>();

    // the train memory contract for the classifier
    public static final String TRAIN_MEMORY_CONTRACT_SHORT_FLAG = "--tmc";
    public static final String TRAIN_MEMORY_CONTRACT_LONG_FLAG = "--trainMemoryContract";
    @Parameter(names = {TRAIN_MEMORY_CONTRACT_SHORT_FLAG, TRAIN_MEMORY_CONTRACT_LONG_FLAG}, arity = 2, description =
        "specify a train memory contract for the classifier in the form \"<amount> <units>\", e.g. \"4 GIGABYTE\" - make"
            + " sure you've considered whether you need GIBIbyte or GIGAbyte though.")
    private List<String> trainMemoryContractStrs = new ArrayList<>();
    private List<MemoryAmount> trainMemoryContracts = new ArrayList<>();

    // the test time contract
    public static final String TEST_TIME_CONTRACT_SHORT_FLAG = "--ptc";
    public static final String TEST_TIME_CONTRACT_LONG_FLAG = "--testTimeContract";
    @Parameter(names = {TEST_TIME_CONTRACT_SHORT_FLAG, TEST_TIME_CONTRACT_LONG_FLAG}, arity = 2, description =
        "specify a test time contract for the classifier in the form \"<amount> <unit>\", e.g. \"1 minute\"")
    private List<String> testTimeContractStrs = new ArrayList<>();
    private List<TimeAmount> testTimeContracts = new ArrayList<>();

    // whether to checkpoint or not. Paths for checkpointing will be auto generated.
    public static final String CHECKPOINT_SHORT_FLAG = "--cp";
    public static final String CHECKPOINT_LONG_FLAG = "--checkpoint";
    @Parameter(names = {CHECKPOINT_SHORT_FLAG, CHECKPOINT_LONG_FLAG}, description = "whether to save the classifier "
        + "to file")
    private boolean checkpoint = false;
    // todo swap train contract save / load path over for checkpointing
    // todo enable checkpointing on classifier

    // checkpoint interval (if using checkpointing)
    public static final String CHECKPOINT_INTERVAL_SHORT_FLAG = "--cpi";
    public static final String CHECKPOINT_INTERVAL_LONG_FLAG = "--checkpointInterval";
    @Parameter(names = {CHECKPOINT_INTERVAL_SHORT_FLAG, CHECKPOINT_INTERVAL_LONG_FLAG}, description = "how often to "
        + "save the classifier to file in the form \"<amount> <unit>\", e.g. \"1 hour\"")
    // todo add checkpoint interval to classifier post tony's interface changes

    // the number of threads to run individual experiments on
    public static final String THREADS_SHORT_FLAG = "-t";
    public static final String THREADS_LONG_FLAG = "--threads";
    @Parameter(names = {THREADS_SHORT_FLAG, THREADS_LONG_FLAG}, description = "how many threads to run experiments on"
        + ". Set this to <=0 to use all processor cores.")
    private int numThreads = 1;

    // whether to append the train time to the classifier name
    public static final String APPEND_TRAIN_TIME_CONTRACT_SHORT_FLAG = "--attc";
    public static final String APPEND_TRAIN_TIME_CONTRACT_LONG_FLAG = "--appendTrainTimeContract";
    @Parameter(names = {APPEND_TRAIN_TIME_CONTRACT_SHORT_FLAG, APPEND_TRAIN_TIME_CONTRACT_LONG_FLAG}, description =
        "append the train time contract to the classifier name")
    private boolean appendTrainTimeContract = false;

    // whether to append the train memory contract to the classifier name
    public static final String APPEND_TRAIN_MEMORY_CONTRACT_SHORT_FLAG = "--atmc";
    public static final String APPEND_TRAIN_MEMORY_CONTRACT_LONG_FLAG = "--appendTrainMemoryContract";
    @Parameter(names = {APPEND_TRAIN_MEMORY_CONTRACT_SHORT_FLAG, APPEND_TRAIN_MEMORY_CONTRACT_LONG_FLAG},
        description = "append the train memory contract to the classifier name")
    private boolean appendTrainMemoryContract = false;

    // whether to append the test time contract to the classifier name
    public static final String APPEND_TEST_TIME_CONTRACT_SHORT_FLAG = "--aptc";
    public static final String APPEND_TEST_TIME_CONTRACT_LONG_FLAG = "--appendTestTimeContract";
    @Parameter(names = {APPEND_TEST_TIME_CONTRACT_SHORT_FLAG, APPEND_TEST_TIME_CONTRACT_LONG_FLAG}, description =
        "append the test time contract to the classifier name")
    private boolean appendTestTimeContract = false;

    // whether to find a train estimate for the classifier
    public static final String ESTIMATE_TRAIN_ERROR_SHORT_FLAG = "-e";
    public static final String ESTIMATE_TRAIN_ERROR_LONG_FLAG = "--estimateTrainError";
    @Parameter(names = {ESTIMATE_TRAIN_ERROR_SHORT_FLAG, ESTIMATE_TRAIN_ERROR_LONG_FLAG}, description = "set the "
        + "classifier to find a train estimate")
    private boolean estimateTrainError = false;
    // todo enable train estimate to be set on a per classifier basis, similar to universal / bespoke params
    // todo another parameter for specifying a cv or something of a non-train-estimateable classifier to find a train
    //  estimate for it

    // the log level to use on the classifier
    public static final String CLASSIFIER_VERBOSITY_SHORT_FLAG = "--cv";
    public static final String CLASSIFIER_VERBOSITY_LONG_FLAG = "--classifierVerbosity";
    @Parameter(names = {CLASSIFIER_VERBOSITY_SHORT_FLAG, CLASSIFIER_VERBOSITY_LONG_FLAG}, description = "classifier "
        + "verbosity")
    private String classifierVerbosity = Level.SEVERE.toString();

    // the log level to use on the experiment
    public static final String EXPERIMENT_VERBOSITY_SHORT_FLAG = "--ev";
    public static final String EXPERIMENT_VERBOSITY_LONG_FLAG = "--experimentVerbosity";
    @Parameter(names = {EXPERIMENT_VERBOSITY_SHORT_FLAG, EXPERIMENT_VERBOSITY_LONG_FLAG}, description = "experiment "
        + "verbosity")
    private String experimentVerbosity = Level.ALL.toString();

    // whether to overwrite train files
    public static final String OVERWRITE_TRAIN_SHORT_FLAG = "--ot";
    public static final String OVERWRITE_TRAIN_LONG_FLAG = "--overwriteTrain";
    @Parameter(names = {OVERWRITE_TRAIN_SHORT_FLAG, OVERWRITE_TRAIN_LONG_FLAG}, description = "overwrite train results")
    private boolean overwriteTrain = false;

    // whether to overwrite test results
    public static final String OVERWRITE_TEST_SHORT_FLAG = "--op";
    public static final String OVERWRITE_TEST_LONG_FLAG = "--overwriteTest";
    @Parameter(names = {OVERWRITE_TEST_SHORT_FLAG, OVERWRITE_TEST_LONG_FLAG}, description = "overwrite test results")
    private boolean overwriteTest = false;

    // the factory to build classifiers using classifier name
    private ClassifierBuilderFactory<Classifier> classifierBuilderFactory =
        ClassifierBuilderFactory.getGlobalInstance();
    // todo get this by string, i.e. factory, and make into cmdline param

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
            ", logLevelClassifier='" + classifierVerbosity + '\'' +
            ", logLevelExperiment='" + experimentVerbosity + '\'' +
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
        logger.finest("parsing {" + experimentVerbosity + "} as logLevel");
        if(experimentVerbosity != null) {
            logger.setLevel(Level.parse(experimentVerbosity));
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
                        experiment.getLogger().setLevel(Level.parse(classifierVerbosity));
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

    public String getClassifierVerbosity() {
        return classifierVerbosity;
    }

    public void setClassifierVerbosity(final String classifierVerbosity) {
        this.classifierVerbosity = classifierVerbosity;
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

    public String getExperimentVerbosity() {
        return experimentVerbosity;
    }

    public ExperimentBatch setExperimentVerbosity(final String experimentVerbosity) {
        this.experimentVerbosity = experimentVerbosity;
        return this;
    }

    public static String getClassifierNameShortFlag() {
        return CLASSIFIER_NAME_SHORT_FLAG;
    }

    public static String getClassifierNameLongFlag() {
        return CLASSIFIER_NAME_LONG_FLAG;
    }

    public static String getDatasetDirShortFlag() {
        return DATASET_DIR_SHORT_FLAG;
    }

    public static String getDatasetDirLongFlag() {
        return DATASET_DIR_LONG_FLAG;
    }

    public static String getDatasetNameShortFlag() {
        return DATASET_NAME_SHORT_FLAG;
    }

    public static String getDatasetNameLongFlag() {
        return DATASET_NAME_LONG_FLAG;
    }

    public static String getSeedShortFlag() {
        return SEED_SHORT_FLAG;
    }

    public static String getSeedLongFlag() {
        return SEED_LONG_FLAG;
    }

    public static String getResultsDirShortFlag() {
        return RESULTS_DIR_SHORT_FLAG;
    }

    public static String getResultsDirLongFlag() {
        return RESULTS_DIR_LONG_FLAG;
    }

    public static String getParametersShortFlag() {
        return PARAMETERS_SHORT_FLAG;
    }

    public static String getParametersLongFlag() {
        return PARAMETERS_LONG_FLAG;
    }

    public static String getAppendClassifierParametersShortFlag() {
        return APPEND_CLASSIFIER_PARAMETERS_SHORT_FLAG;
    }

    public static String getAppendClassifierParametersLongFlag() {
        return APPEND_CLASSIFIER_PARAMETERS_LONG_FLAG;
    }

    public static String getTrainTimeContractShortFlag() {
        return TRAIN_TIME_CONTRACT_SHORT_FLAG;
    }

    public static String getTrainTimeContractLongFlag() {
        return TRAIN_TIME_CONTRACT_LONG_FLAG;
    }

    public static String getTrainMemoryContractShortFlag() {
        return TRAIN_MEMORY_CONTRACT_SHORT_FLAG;
    }

    public static String getTrainMemoryContractLongFlag() {
        return TRAIN_MEMORY_CONTRACT_LONG_FLAG;
    }

    public static String getTestTimeContractShortFlag() {
        return TEST_TIME_CONTRACT_SHORT_FLAG;
    }

    public static String getTestTimeContractLongFlag() {
        return TEST_TIME_CONTRACT_LONG_FLAG;
    }

    public static String getCheckpointShortFlag() {
        return CHECKPOINT_SHORT_FLAG;
    }

    public static String getCheckpointLongFlag() {
        return CHECKPOINT_LONG_FLAG;
    }

    public static String getThreadsShortFlag() {
        return THREADS_SHORT_FLAG;
    }

    public static String getThreadsLongFlag() {
        return THREADS_LONG_FLAG;
    }

    public static String getAppendTrainTimeContractShortFlag() {
        return APPEND_TRAIN_TIME_CONTRACT_SHORT_FLAG;
    }

    public static String getAppendTrainTimeContractLongFlag() {
        return APPEND_TRAIN_TIME_CONTRACT_LONG_FLAG;
    }

    public static String getAppendTrainMemoryContractShortFlag() {
        return APPEND_TRAIN_MEMORY_CONTRACT_SHORT_FLAG;
    }

    public static String getAppendTrainMemoryContractLongFlag() {
        return APPEND_TRAIN_MEMORY_CONTRACT_LONG_FLAG;
    }

    public static String getAppendTestTimeContractShortFlag() {
        return APPEND_TEST_TIME_CONTRACT_SHORT_FLAG;
    }

    public static String getAppendTestTimeContractLongFlag() {
        return APPEND_TEST_TIME_CONTRACT_LONG_FLAG;
    }

    public static String getEstimateTrainErrorShortFlag() {
        return ESTIMATE_TRAIN_ERROR_SHORT_FLAG;
    }

    public static String getEstimateTrainErrorLongFlag() {
        return ESTIMATE_TRAIN_ERROR_LONG_FLAG;
    }

    public static String getClassifierVerbosityShortFlag() {
        return CLASSIFIER_VERBOSITY_SHORT_FLAG;
    }

    public static String getClassifierVerbosityLongFlag() {
        return CLASSIFIER_VERBOSITY_LONG_FLAG;
    }

    public static String getExperimentVerbosityShortFlag() {
        return EXPERIMENT_VERBOSITY_SHORT_FLAG;
    }

    public static String getExperimentVerbosityLongFlag() {
        return EXPERIMENT_VERBOSITY_LONG_FLAG;
    }

    public static String getOverwriteTrainShortFlag() {
        return OVERWRITE_TRAIN_SHORT_FLAG;
    }

    public static String getOverwriteTrainLongFlag() {
        return OVERWRITE_TRAIN_LONG_FLAG;
    }

    public static String getOverwriteTestShortFlag() {
        return OVERWRITE_TEST_SHORT_FLAG;
    }

    public static String getOverwriteTestLongFlag() {
        return OVERWRITE_TEST_LONG_FLAG;
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
