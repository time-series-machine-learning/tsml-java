package tsml.classifiers.distance_based.utils;

import com.beust.jcommander.DynamicParameter;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.SubParameter;
import com.beust.jcommander.internal.Lists;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.function.BiFunction;
import java.util.logging.Level;
import java.util.logging.Logger;
import tsml.classifiers.distance_based.utils.classifier_building.ClassifierBuilderFactory;
import tsml.classifiers.distance_based.utils.classifier_building.ClassifierBuilderFactory.ClassifierBuilder;
import tsml.classifiers.distance_based.utils.logging.LogUtils;
import tsml.classifiers.distance_based.utils.memory.MemoryAmount;
import tsml.classifiers.distance_based.utils.parallel.BlockingExecutor;
import tsml.classifiers.distance_based.utils.stopwatch.TimeAmount;
import utilities.FileUtils;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */

public class Main {
    // todo use getters and setters internally

    @Parameter(names = {"-c", "--classifier"}, description = "todo", required = true)
    private List<String> classifierNames = new ArrayList<>();

    @Parameter(names = {"--datasetsDir", "-dd"}, description = "todo", required = true)
    private List<String> datasetDirPaths = new ArrayList<>();

    @Parameter(names = {"--dataset", "-d"}, description = "todo", required = true)
    private List<String> datasetNames = new ArrayList<>();

    @Parameter(names = {"--seed", "-s"}, description = "todo", required = true)
    private List<Integer> seeds = Lists.newArrayList(0);

    @Parameter(names = {"-rd", "--resultsDir"}, description = "todo", required = true)
    private String resultsDirPath = null;

    private static class ClassifierParameters {
        @SubParameter(order = 0)
        private String classifierName;

        @SubParameter(order = 1)
        private String parameters;

        public String getClassifierName() {
            return classifierName;
        }

        public void setClassifierName(final String classifierName) {
            this.classifierName = classifierName;
        }

        public String getParameters() {
            return parameters;
        }

        public void setParameters(final String parameters) {
            this.parameters = parameters;
        }
    }

    @DynamicParameter(assignment = " ", names = {"-p", "--parameters"}, description = "todo")
    private List<ClassifierParameters> classifierParameters = new ArrayList<>(); // todo apply these to the classifier
    private Map<String, String> classifierNameToParametersMap;

    @Parameter(names = {"-ttc", "--trainTimeContract"}, arity = 2, description = "todo")
    private List<String> trainTimeContractStrs = new ArrayList<>();
    private List<TimeAmount> trainTimeContracts = new ArrayList<>();

    @Parameter(names = {"-tmc", "--trainMemoryContract"}, arity = 2, description = "todo")
    private List<String> trainMemoryContractStrs = new ArrayList<>();
    private List<MemoryAmount> trainMemoryContracts = new ArrayList<>();

    @Parameter(names = {"-ptc", "--predictTimeContract"}, arity = 2, description = "todo")
    private List<String> predictTimeContractStrs = new ArrayList<>();
    private List<TimeAmount> predictTimeContracts = new ArrayList<>();

    @Parameter(names = "--checkpoint", description = "todo")
    private boolean checkpoint = false; // todo swap train contract save / load path over

    @Parameter(names = {"-t", "--threads"}, description = "todo")
    private int numThreads = 1; // <=0 for all available threads

    @Parameter(names = {"-attc", "--appendTrainTimeContract"}, description = "todo")
    private boolean appendTrainTimeContract = false;

    @Parameter(names = {"-atmc", "--appendTrainMemoryContract"}, description = "todo")
    private boolean appendTrainMemoryContract = false;

    @Parameter(names = {"-aptc", "--appendPredictTimeContract"}, description = "todo")
    private boolean appendPredictTimeContract = false;

    @Parameter(names = {"-ete", "--estimateTrainError"}, description = "todo")
    private boolean estimateTrainError = false;

    @Parameter(names = {"-l", "--logLevel"}, description = "todo")
    private String logLevel = Level.ALL.toString(); // todo set log level in classifier / experiment

    @Parameter(names = {"-of", "--overwriteTrain"}, description = "todo")
    private boolean overwriteTrain = false;

    @Parameter(names = {"-op", "--overwriteTest"}, description = "todo")
    private boolean overwritePredict = false;

    private final Logger logger = LogUtils.buildLogger(this);

    private ClassifierBuilderFactory<Classifier> classifierBuilderFactory =
        ClassifierBuilderFactory.getGlobalInstance(); // todo get this by string, i.e. factory

    public static void main(String ... args) {
        new Main(args).runExperiments();
    }

    public static class Runner {

        public static void main(String[] args) {
            Main.main(
                "--threads", "1",
                "-r", "results",
                "-s", "0",
                "-c", "DTW_1NN_V1",
                "--estimateTrainError",
                "-d", "GunPoint",
                "-p", "/bench/datasets"
            );
        }
    }



    public static <A> List<A> convertStringPairs(List<String> strs, BiFunction<String, String, A> func) {
        List<A> objs = new ArrayList<>();
        for(int i = 0; i < strs.size(); i += 2) {
            final String trainContractAmountStr = strs.get(i);
            final String trainContractUnitStr = strs.get(i + 1);
            final A obj = func.apply(trainContractAmountStr, trainContractUnitStr);
            objs.add(obj);
        }
        return objs;
    }

    public static List<TimeAmount> convertStringPairsToTimeAmounts(List<String> strs) {
        List<TimeAmount> times = convertStringPairs(strs, TimeAmount::parse);
        Collections.sort(times);
        return times;
    }

    public static List<MemoryAmount> convertStringPairsToMemoryAmounts(List<String> strs) {
        List<MemoryAmount> times = convertStringPairs(strs, MemoryAmount::parse);
        Collections.sort(times);
        return times;
    }

    public void parse(String... args) {
        JCommander.newBuilder()
            .addObject(this)
            .build()
            .parse(args);
        // perform custom parsing here
        if(logLevel != null) {
            logger.setLevel(Level.parse(logLevel));
        }
        if(trainTimeContractStrs.size() % 2 != 0) {
            throw new IllegalStateException("train contracts must be a list of pairs, i.e. \"5\" \"minutes\"");
        }
        if(predictTimeContractStrs.size() % 2 != 0) {
            throw new IllegalStateException("test contracts must be a list of pairs, i.e. \"5\" \"minutes\"");
        }
        if(trainMemoryContractStrs.size() % 2 != 0) {
            throw new IllegalStateException("test contracts must be a list of pairs, i.e. \"5\" \"minutes\"");
        }
        trainTimeContracts = convertStringPairsToTimeAmounts(trainTimeContractStrs);
        predictTimeContracts = convertStringPairsToTimeAmounts(predictTimeContractStrs);
        trainMemoryContracts = convertStringPairsToMemoryAmounts(trainMemoryContractStrs);
        classifierNameToParametersMap = new HashMap<>();
        for(ClassifierParameters classifierParameter : classifierParameters) {
            classifierNameToParametersMap.put(classifierParameter.getClassifierName(),
                classifierParameter.getParameters());
        }
    }

    public Main(String... args) {
        parse(args);
    }

    private Instances[] loadData(String name, int seed) {
        for(final String path : datasetDirPaths) {
            try {
                Instances[] data = DatasetLoading.sampleDataset(path, name, seed);
                if(data == null) {
                    throw new Exception();
                }
                return data;
            } catch(Exception ignored) {

            }
        }
        throw new IllegalArgumentException("couldn't load data");
    }

    @Override
    public String toString() {
        return
            "classifierNames=" + classifierNames +
            ", datasetDirPaths=" + datasetDirPaths +
            ", datasetNames=" + datasetNames +
            ", seeds=" + seeds +
            ", resultsDirPath='" + resultsDirPath + '\'' +
            ", trainContracts=" + trainTimeContractStrs +
            ", checkpoint=" + checkpoint +
            ", classifierBuilderFactory=" + classifierBuilderFactory
            ;
    }

    private ExecutorService buildExecutor() {
        int numThreads = this.numThreads;
        if(numThreads < 1) {
            numThreads = Runtime.getRuntime().availableProcessors();
        }
        ThreadPoolExecutor threadPool = (ThreadPoolExecutor) Executors.newFixedThreadPool(numThreads);
        return new BlockingExecutor(threadPool);
    }

    /**
     *
     // todo break this into a method of splitting, i.e. resamples / cv - what about using an evaluator?
     //  Just set it from factory methods in params above. This currently has a problem as seed 0 doesn't
     //  resample to the same as the offline file split
     */
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
                    final ClassifierBuilder<? extends Classifier> classifierBuilder = classifierBuilderFactory
                        .getClassifierBuilderByName(classifierName);
                    if(classifierBuilder == null) {
                        logger.severe("no classifier by the name of {" + classifierName + "}, skipping experiment");
                        continue;
                    }
                    final Classifier classifier = classifierBuilder.build();
                    final Experiment experiment = new Experiment(trainData, testData, classifier, seed,
                        classifierName, datasetName);
                    executor.submit(() -> {
                        try {
                            runExperimentBatch(experiment);
                        } catch(Exception e) {
                            e.printStackTrace();
                        }
                    });
                }
            }
        }
        executor.shutdown();
    }

    // switch to control whether we need to switch out the random source for testing. For example, if we train a
    // classifier for 5 mins, then test, then train for another 5 mins (to 10 mins), then test, the results are
    // different to training for 10 minutes alone then testing. This is because the classifier sources random
    // numbers during testing and training, therefore the extra testing in the first version causes different
    // random numbers. Obviously this only matters if the classifier uses the random source during testing, but
    // for safety it is best to assume all classifiers do and switch the source to an alternate source for each
    // test batch.
    private void runExperimentBatch(Experiment experiment) throws Exception {
        // if there's no train contract we'll put nulls in place. This causes the loop to fire and we'll handle nulls
        // as no contract inside the loop
        if(trainTimeContracts.isEmpty()) {
            trainTimeContracts.add(null);
        }
        final String origClassifierName = experiment.getClassifierName();
        // for each train contract (pair of strs, one for the amount, one for the unit)
        for(TimeAmount trainContract : trainTimeContracts) {
            // setup the next train contract
            if(trainContract == null) {
                // no train contract
                logger.info("no train contract for {" + experiment.getClassifierName() + "} on {" + experiment.getDatasetName() + "}");
                // todo set train contract disabled somehow? Tony set some boolean in the api somewhere, see if
                //  that'll do
            } else {
                logger.info("train contract of {" + trainContract + "} for {" + experiment.getClassifierName() + "} on {" + experiment.getDatasetName() + "}");
                experiment.setTrainTimeLimit(trainContract.getAmount(), trainContract.getUnit()); // todo add this to
                // the interface, overload
                if(appendTrainTimeContract) {
                    experiment.setClassifierName(origClassifierName + "_" + trainContract.toString().replaceAll(" ", "_"));
                }
            }
            if(isOverwriteTrain() && isOverwritePredict()) {
                getLogger().info("results exist");
                continue;
            }
            // train classifier
            experiment.train();
            // write train results if enabled
            final String classifierResultsDirPath =
                resultsDirPath + "/" + experiment.getClassifierName() + "/" + experiment.getDatasetName() + "/";
            if(experiment.isEstimateTrain() && !isOverwriteTrain()) {
                final String trainResultsFilePath = classifierResultsDirPath + "trainFold" + experiment.getSeed() +
                    ".csv";
                final ClassifierResults trainResults = experiment.getTrainResults();
                FileUtils.writeToFile(trainResults.writeFullResultsToString(), trainResultsFilePath);
            }
            // test classifier
            experiment.test();
            // write test results
            if(!isOverwritePredict()) {
                final String testResultsFilePath = classifierResultsDirPath + "trainFold" + experiment.getSeed() + ".csv";
                final ClassifierResults testResults = experiment.getTestResults();
                FileUtils.writeToFile(testResults.writeFullResultsToString(), testResultsFilePath);
            }
            // shallow copy experiment so we can reuse the configuration under the next train contract
            experiment = (Experiment) experiment.shallowCopy();
        }
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

    public String getLogLevel() {
        return logLevel;
    }

    public void setLogLevel(final String logLevel) {
        this.logLevel = logLevel;
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

    public boolean isOverwritePredict() {
        return overwritePredict;
    }

    public void setOverwritePredict(final boolean overwritePredict) {
        this.overwritePredict = overwritePredict;
    }

    public List<ClassifierParameters> getClassifierParameters() {
        return classifierParameters;
    }

    public void setClassifierParameters(
        final List<ClassifierParameters> classifierParameters) {
        this.classifierParameters = classifierParameters;
    }

    public Map<String, String> getClassifierNameToParametersMap() {
        return classifierNameToParametersMap;
    }

    public void setClassifierNameToParametersMap(final Map<String, String> classifierNameToParametersMap) {
        this.classifierNameToParametersMap = classifierNameToParametersMap;
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

    public List<String> getPredictTimeContractStrs() {
        return predictTimeContractStrs;
    }

    public void setPredictTimeContractStrs(final List<String> predictTimeContractStrs) {
        this.predictTimeContractStrs = predictTimeContractStrs;
    }

    public List<TimeAmount> getPredictTimeContracts() {
        return predictTimeContracts;
    }

    public void setPredictTimeContracts(
        final List<TimeAmount> predictTimeContracts) {
        this.predictTimeContracts = predictTimeContracts;
    }

    public boolean isAppendTrainMemoryContract() {
        return appendTrainMemoryContract;
    }

    public void setAppendTrainMemoryContract(final boolean appendTrainMemoryContract) {
        this.appendTrainMemoryContract = appendTrainMemoryContract;
    }

    public boolean isAppendPredictTimeContract() {
        return appendPredictTimeContract;
    }

    public void setAppendPredictTimeContract(final boolean appendPredictTimeContract) {
        this.appendPredictTimeContract = appendPredictTimeContract;
    }
}
