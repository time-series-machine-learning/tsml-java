//package tsml.classifiers.distance_based.utils.experiments;
//
//import com.beust.jcommander.IStringConverter;
//import com.beust.jcommander.JCommander;
//import com.beust.jcommander.Parameter;
//import experiments.data.DatasetLoading;
//import java.io.File;
//import java.util.ArrayList;
//import java.util.Collections;
//import java.util.List;
//import java.util.logging.Logger;
//import tsml.classifiers.Checkpointable;
//import tsml.classifiers.distance_based.utils.classifier_building.ClassifierBuilderFactory;
//import tsml.classifiers.distance_based.utils.classifier_building.ClassifierBuilderFactory.ClassifierBuilder;
//import tsml.classifiers.distance_based.utils.logging.LogUtils;
//import tsml.classifiers.distance_based.utils.params.ParamSet;
//import tsml.classifiers.distance_based.utils.stopwatch.TimeAmount;
//import tsml.classifiers.distance_based.utils.system.memory.MemoryAmount;
//import utilities.FileUtils.FileLock;
//import utilities.FileUtils.FileLock.LockException;
//import weka.classifiers.Classifier;
//import weka.core.Instances;
//
//public class Exp {
//
//    public Exp(final Args args) {
//        this.args = args;
//    }
//
//    public static class Args {
//
//        private static class TimeAmountConverter implements
//            IStringConverter<TimeAmount> {
//
//            @Override
//            public TimeAmount convert(final String str) {
//                return TimeAmount.parse(str);
//            }
//        }
//
//        private static class MemoryAmountConverter implements
//            IStringConverter<MemoryAmount> {
//
//            @Override
//            public MemoryAmount convert(final String str) {
//                return MemoryAmount.parse(str);
//            }
//        }
//
//        // the seeds to run
//        public static final String SEED_SHORT_FLAG = "-s";
//        public static final String SEED_LONG_FLAG = "--seed";
//        @Parameter(names = {SEED_SHORT_FLAG, SEED_LONG_FLAG}, description = "the seed to be used in sampling a dataset "
//            + "and in the random source for the classifier", required = true)
//        private Integer seed;
//
//        // the classifier to use
//        public static final String CLASSIFIER_SHORT_FLAG = "-c";
//        public static final String CLASSIFIER_LONG_FLAG = "--classifier";
//        @Parameter(names = {CLASSIFIER_SHORT_FLAG, CLASSIFIER_LONG_FLAG},
//            description = "append the train memory contract to the classifier name")
//        private String classifierName;
//
//        // where to put the results when finished
//        public static final String RESULTS_DIR_SHORT_FLAG = "-r";
//        public static final String RESULTS_DIR_LONG_FLAG = "--resultsDir";
//        @Parameter(names = {RESULTS_DIR_SHORT_FLAG, RESULTS_DIR_LONG_FLAG}, description = "path to a folder to place "
//            + "results in",
//            required = true)
//        private String resultsDirPath;
//
//        // paths to directory where problem data is stored
//        public static final String DATASET_DIR_SHORT_FLAG = "--dd";
//        public static final String DATASET_DIR_LONG_FLAG = "--datasetsDir";
//        @Parameter(names = {DATASET_DIR_SHORT_FLAG, DATASET_DIR_LONG_FLAG}, description = "the path to the folder "
//            + "containing the datasets",
//            required = true)
//        private String datasetDirPath;
//
//        // names of the dataset that should be run
//        public static final String DATASET_NAME_SHORT_FLAG = "-d";
//        public static final String DATASET_NAME_LONG_FLAG = "--dataset";
//        @Parameter(names = {DATASET_NAME_SHORT_FLAG, DATASET_NAME_LONG_FLAG}, description = "the name of the dataset",
//            required = true)
//        private String datasetName;
//
//        // whether to overwrite train files
//        public static final String OVERWRITE_TRAIN_SHORT_FLAG = "--ot";
//        public static final String OVERWRITE_TRAIN_LONG_FLAG = "--overwriteTrain";
//        @Parameter(names = {OVERWRITE_TRAIN_SHORT_FLAG, OVERWRITE_TRAIN_LONG_FLAG}, description = "overwrite train results")
//        private boolean overwriteTrain = false;
//
//        // whether to overwrite test results
//        public static final String OVERWRITE_TEST_SHORT_FLAG = "--op";
//        public static final String OVERWRITE_TEST_LONG_FLAG = "--overwriteTest";
//        @Parameter(names = {OVERWRITE_TEST_SHORT_FLAG, OVERWRITE_TEST_LONG_FLAG}, description = "overwrite test results")
//        private boolean overwriteTest = false;
//
//        // the train time contract for the classifier
//        public static final String TRAIN_TIME_CONTRACT_SHORT_FLAG = "--ttc";
//        public static final String TRAIN_TIME_CONTRACT_LONG_FLAG = "--trainTimeContract";
//        @Parameter(names = {TRAIN_TIME_CONTRACT_SHORT_FLAG, TRAIN_TIME_CONTRACT_LONG_FLAG}, converter =
//            TimeAmountConverter.class, description =
//            "specify a train time contract for the classifier in the form \"<amount> <units>\", e.g. \"4 hour\"")
//        private List<TimeAmount> trainTimeContracts = new ArrayList<>();
//
//        // the train memory contract for the classifier
//        public static final String TRAIN_MEMORY_CONTRACT_SHORT_FLAG = "--tmc";
//        public static final String TRAIN_MEMORY_CONTRACT_LONG_FLAG = "--trainMemoryContract";
//        @Parameter(names = {TRAIN_MEMORY_CONTRACT_SHORT_FLAG, TRAIN_MEMORY_CONTRACT_LONG_FLAG}, converter =
//            MemoryAmountConverter.class, description =
//            "specify a train memory contract for the classifier in the form \"<amount> <units>\", e.g. \"4 GIGABYTE\" - make"
//                + " sure you've considered whether you need GIBIbyte or GIGAbyte though.")
//        private List<MemoryAmount> trainMemoryContracts = new ArrayList<>();
//
//        // the test time contract
//        public static final String TEST_TIME_CONTRACT_SHORT_FLAG = "--ptc";
//        public static final String TEST_TIME_CONTRACT_LONG_FLAG = "--testTimeContract";
//        @Parameter(names = {TEST_TIME_CONTRACT_SHORT_FLAG, TEST_TIME_CONTRACT_LONG_FLAG}, converter =
//            TimeAmountConverter.class, description =
//            "specify a test time contract for the classifier in the form \"<amount> <unit>\", e.g. \"1 minute\"")
//        private List<TimeAmount> testTimeContracts = new ArrayList<>();
//
//        private ClassifierBuilderFactory<Classifier> classifierBuilderFactory =
//            ClassifierBuilderFactory.getGlobalInstance();
//
//        public ClassifierBuilderFactory<Classifier> getClassifierBuilderFactory() {
//            return classifierBuilderFactory;
//        }
//
//        public void parse(String... args) {
//            // parse args
//            JCommander.newBuilder().addObject(this).build().parse(args);
//        }
//
//        public Args(String... args) {
//            parse(args);
//        }
//
//        public Integer getSeed() {
//            return seed;
//        }
//
//        public String getClassifierName() {
//            return classifierName;
//        }
//
//        public String getResultsDirPath() {
//            return resultsDirPath;
//        }
//
//        public String getDatasetDirPath() {
//            return datasetDirPath;
//        }
//
//        public String getDatasetName() {
//            return datasetName;
//        }
//
//        public boolean isOverwriteTrain() {
//            return overwriteTrain;
//        }
//
//        public boolean isOverwriteTest() {
//            return overwriteTest;
//        }
//
//        public List<TimeAmount> getTrainTimeContracts() {
//            return trainTimeContracts;
//        }
//
//        public List<MemoryAmount> getTrainMemoryContracts() {
//            return trainMemoryContracts;
//        }
//
//        public List<TimeAmount> getTestTimeContracts() {
//            return testTimeContracts;
//        }
//    }
//
//    private final Args args;
//    private Experiment experiment;
//    private final Logger logger = LogUtils.buildLogger(this);
//
//    public Logger getLogger() {
//        return logger;
//    }
//
//    public Args getArgs() {
//        return args;
//    }
//
//    private Instances[] loadData() throws Exception {
//        Args args = getArgs();
//        Instances[] data = DatasetLoading.sampleDataset(args.getDatasetDirPath(), args.getDatasetName(),
//            args.getSeed());
//        if(data == null) {
//            throw new Exception("data not found");
//        }
//        getLogger().finest("loaded {" + args.getDatasetName() + "} from {" + args.getDatasetDirPath() + "}");
//        return data;
//    }
//
//    private String buildClassifierWorkspaceDirPath() {
//        return args.getResultsDirPath() + "/" + experiment.getClassifierName() + "/Workspace/" + experiment.getDatasetName() + "/fold" + experiment.getSeed() + "/";
//    }
//
//    private String buildExperimentResultsDirPath() {
//        return args.getResultsDirPath() + "/" + experiment.getClassifierName() + "/Predictions/" + experiment.getDatasetName() +
//            "/";
//    }
//
//    private String buildTrainResultsFilePath() {
//        return buildExperimentResultsDirPath() + "trainFold" + experiment.getSeed() +
//            ".csv";
//    }
//
//    private String buildTestResultsFilePath() {
//        return buildExperimentResultsDirPath() + "testFold" + experiment.getSeed() +
//            ".csv";
//    }
//
//    private Experiment buildExperiment() throws Exception {
//        // build classifier
//        final Classifier classifier = buildClassifier();
//        // load data
//        final Instances[] data = loadData();
//        final Instances trainData = data[0];
//        final Instances testData = data[1];
//        final Args args = getArgs();
//        setupCollection(args.getTrainMemoryContracts());
//        setupCollection(args.getTrainTimeContracts());
//        setupCollection(args.getTestTimeContracts());
//        final Experiment experiment = new Experiment(trainData, testData, classifier, args.getSeed(),
//            args.getClassifierName(), args.getDatasetName());
//        return experiment;
//    }
//
//    private boolean shouldTest() {
//        if(args.isOverwriteTest()) {
//            return true;
//        }
//        return new File(buildTestResultsFilePath()).exists();
//    }
//
//    private boolean shouldTrain() {
//        if(args.isOverwriteTrain()) {
//            return true;
//        }
//        if(shouldTest()) {
//            return true;
//        }
//        return new File(buildTrainResultsFilePath()).exists();
//    }
//
//    private Classifier buildClassifier() {
//        Args args = getArgs();
//        final String classifierName = args.getClassifierName();
//        final ClassifierBuilder<? extends Classifier> classifierBuilder = args.getClassifierBuilderFactory()
//            .getClassifierBuilderByName(classifierName);
//        if(classifierBuilder == null) {
//            throw new IllegalArgumentException("no classifier by the name of {" + classifierName + "}, skipping experiment");
//        }
//        final Classifier classifier = classifierBuilder.build();
//        return classifier;
//    }
//
//    private static <A extends Comparable<A>> void setupCollection(List<A> list) {
//        if(list.isEmpty()) {
//            list.add(null);
//        } else {
//            Collections.sort(list);
//        }
//    }
//
//    public void runExperiments() throws Exception {
//        experiment = buildExperiment();
//        Args args = getArgs();
//        final String origClassifierName = experiment.getClassifierName();
//        String prevClassifierName = null;
//        for(MemoryAmount trainMemoryContract : args.getTrainMemoryContracts()) {
//            for(TimeAmount trainTimeContract : args.getTrainTimeContracts()) {
//                for(TimeAmount testTimeContract : args.getTestTimeContracts()) {
//                    experiment.setClassifierName(origClassifierName);
//                    applyTrainMemoryContract(trainMemoryContract);
//                    applyTrainTimeContract(trainTimeContract);
//                    applyTestTimeContract(testTimeContract);
//                    appendParametersToClassifierName();
//                    applyCheckpointPaths(prevClassifierName);
//                    runExperiment();
//                    prevClassifierName = experiment.getClassifierName();
//                }
//            }
//        }
//    }
//
//    private void applyCheckpointPaths(final String prevClassifierName) {
//        Experiment experiment = getExperiment();
//        Classifier classifier = experiment.getClassifier();
//        if(classifier instanceof Checkpointable) {
//            String savePath = buildClassifierWorkspaceDirPath();
//            String loadPath;
//            if(prevClassifierName == null) {
//                loadPath = savePath;
//            } else {
//                final String origClassifierName = getExperiment().getClassifierName();
//                experiment.setClassifierName(prevClassifierName);
//                loadPath = buildClassifierWorkspaceDirPath();
//                experiment.setClassifierName(origClassifierName);
//            }
//            getLogger().info("set load path to {" + loadPath + "}");
//            getLogger().info("set save path to {" + savePath + "}");
//            ((Checkpointable) classifier).setLoadPath(loadPath);
//            ((Checkpointable) classifier).setSavePath(savePath);
//        }
//    }
//
//    private void runExperiment() {
//        try (FileLock lock = new FileLock(buildTestResultsFilePath())) {
//            appendParametersToClassifierName();
//            if(shouldTrain()) {
//                // train classifier
//                getLogger().info("training");
//                experiment.train();
//                if(experiment.isEstimateTrainError()) {
//                    experiment.getTrainResults().writeFullResultsToFile(buildTrainResultsFilePath());
//                }
//                // test classifier
//                if(shouldTest()) {
//                    getLogger().info("testing");
//                    experiment.test();
//                    experiment.getTestResults().writeFullResultsToFile(buildTestResultsFilePath());
//                } else {
//                    getLogger().info("skipping testing");
//                }
//            }
//        } catch(LockException e) {
//            logger.severe("failed log lock experiment");
//        } catch(Exception e) {
//            logger.severe(e.toString());
//        }
//    }
//
//
//    private void applyTrainTimeContract(TimeAmount trainTimeContract) {
//        // setup the next train contract
//        if(trainTimeContract != null) {
//            String classifierName = experiment.getClassifierName();
//            getLogger().info(
//                "train time contract of {" + trainTimeContract + "}");
//            experiment.setTrainTimeLimit(trainTimeContract.getAmount(), trainTimeContract.getUnit()); // todo add this to
//            // the interface, overload
//            classifierName = classifierName + "_" + trainTimeContract.toString().replaceAll(" ", "_");
//            experiment.setClassifierName(classifierName);
//            getLogger().info("changed classifier name to {" + classifierName + "}");
//        } else {
//            // no train contract
//            // todo set train contract disabled somehow? Tony set some boolean in the api somewhere, see if
//            //  that'll do
//        }
//    }
//
//    private void applyTrainMemoryContract(MemoryAmount trainMemoryContract) {
//        // setup the next train contract
//        if(trainMemoryContract != null) {
//            String classifierName = experiment.getClassifierName();
//            getLogger().info(
//                "train memory contract of {" + trainMemoryContract + "}");
//            //            experiment.setMemoryLimit(trainMemoryContract.getAmount(), trainMemoryContract.getUnit()); // todo add this to
//            // the interface, overload
//            // todo setup mem limit in experiment
//            classifierName = classifierName + "_" + trainMemoryContract.toString().replaceAll(" ", "_");
//            experiment.setClassifierName(classifierName);
//            getLogger().info("changed classifier name to {" + classifierName + "}");
//        } // else no train contract
//
//    }
//
//    private void applyTestTimeContract(TimeAmount testTimeContract) {
//        // setup the next train contract
//        if(testTimeContract != null) {
//            String classifierName = experiment.getClassifierName();
//            getLogger().info(
//                "test time contract of {" + testTimeContract + "}");
//            experiment.setTestTimeLimit(testTimeContract.getUnit(), testTimeContract.getAmount());
//            // todo add this to
//            // the interface, overload
//            classifierName = classifierName + "_" + testTimeContract.toString().replaceAll(" ", "_");
//            experiment.setClassifierName(classifierName);
//            getLogger().info("changed classifier name to {" + classifierName + "}");
//        } // else no train contract
//
//    }
//
//    public Experiment getExperiment() {
//        return experiment;
//    }
//
//    private void appendParametersToClassifierName() {
//        Experiment experiment = getExperiment();
//        ParamSet paramSet = experiment.getParamSet();
//        if(!paramSet.isEmpty()) {
//            String classifierName = experiment.getClassifierName();
//            String paramSetStr = "_" + paramSet.toString().replaceAll(" ", "_").replaceAll("\"", "#");
//            classifierName += paramSetStr;
//            experiment.setClassifierName(classifierName);
//            getLogger().info("changed classifier name to {" + classifierName + "}");
//        }
//    }
//}
