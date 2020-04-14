package tsml.classifiers.distance_based.utils.experiments;

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import java.io.File;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.logging.Logger;
import tsml.classifiers.distance_based.utils.classifier_building.ClassifierBuilderFactory.ClassifierBuilder;
import tsml.classifiers.distance_based.utils.classifier_mixins.Copy;
import tsml.classifiers.distance_based.utils.collections.box.Box;
import tsml.classifiers.distance_based.utils.logging.LogUtils;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import tsml.classifiers.distance_based.utils.stopwatch.TimeAmount;
import tsml.classifiers.distance_based.utils.system.memory.MemoryAmount;
import utilities.FileUtils;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Purpose: run a classifier.
 *
 * Contributors: goastler
 */
public class ExperimentBatch {
    // logger for printing messages
    private final Logger logger = LogUtils.buildLogger(this);

    private final ExperimentArgs args;

    public ExperimentBatch(final ExperimentArgs args) {
        this.args = args;
    }

    public static void main(String... args) throws Exception {
        ExperimentArgs experimentArgs = new ExperimentArgs(args);
        ExperimentBatch experimentBatch = new ExperimentBatch(experimentArgs);
        experimentBatch.run();
    }


    private Instances[] loadData() throws Exception {
        Instances[] data = DatasetLoading.sampleDataset(args.getDatasetDirPath(), args.getDatasetName(),
            args.getSeed());
        if(data == null) {
            throw new Exception("data not found");
        }
        logger.finest("loaded {" + args.getDatasetName() + "} from {" + args.getDatasetDirPath() + "}");
        return data;
    }

    private Classifier buildClassifier() {
        final String classifierName = args.getClassifierName();
        final ClassifierBuilder<? extends Classifier> classifierBuilder = args.getClassifierBuilderFactory()
            .getClassifierBuilderByName(classifierName);
        if(classifierBuilder == null) {
            throw new IllegalArgumentException("no classifier by the name of {" + classifierName + "}, skipping experiment");
        }
        final Classifier classifier = classifierBuilder.build();
        return classifier;
    }

    private boolean conductingMultipleExperiments() {
        return !args.getTrainMemoryContracts().isEmpty() || !args.getTestTimeContracts().isEmpty() || !args.getTrainTimeContracts().isEmpty();
    }

    private Experiment buildExperiment() throws Exception {
        final Classifier classifier = buildClassifier();
        final Instances[] data = loadData();
        final Instances trainData = data[0];
        final Instances testData = data[1];
        final Experiment experiment = new Experiment(trainData, testData, classifier, args.getSeed(),
            args.getClassifierName()
            , args.getDatasetName());
        return experiment;
    }

    public void run() throws Exception {
        logger.setLevel(args.getExperimentVerbosityLevel());
        logger.info("experiments config: " + this);
        final Experiment experiment = buildExperiment();
        experiment.getLogger().setLevel(args.getClassifierVerbosityLevel());
        experiment.setEstimateTrainError(args.isEstimateTrainError());
        runExperimentTrain(experiment);
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
            return true;
        });
    }

    private boolean runExperimentTest(Experiment experiment) {
        try {
            testExperiment(experiment);
        } catch(Exception e) {
            e.printStackTrace();
        }
        return true;
    }

    private void trainExperiment(Experiment experiment) throws Exception {
        // train classifier
        logger.info("training {" + experiment.getClassifierName() + "} on {" + experiment.getDatasetName() + "}");
        if(experiment.isTrained() && conductingMultipleExperiments()) {
            logger.info("cloning classifier for further experiments");
            Classifier classifier = experiment.getClassifier();
            classifier = Copy.deepCopy(classifier);
            experiment.setClassifier(classifier);
            experiment.resetTrain();
        }
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

    private void forEachExperimentTrain(Experiment experiment, Function<Experiment, Boolean> function) {
        appendParametersToClassifierName(experiment);
        // for each train contract (pair of strs, one for the amount, one for the unit)
        for(MemoryAmount trainMemoryContract : args.getTrainMemoryContracts()) {
            applyTrainMemoryContract(experiment, trainMemoryContract);
            for(TimeAmount trainTimeContract : args.getTrainTimeContracts()) {
                applyTrainTimeContract(experiment, trainTimeContract);
                if(!function.apply(experiment)) {
                    return;
                }
            }
        }
    }

    private void forEachExperimentTest(Experiment experiment, Function<Experiment, Boolean> function) {
        for(TimeAmount testTimeContract : args.getTestTimeContracts()) {
            applyTestTimeContract(experiment, testTimeContract);
            if(shouldTest(experiment)) {
                if(!function.apply(experiment)) {
                    return;
                }
            }
        }
    }

    private boolean shouldTest(Experiment experiment) {
        return shouldTest(experiment, true);
    }

    private boolean shouldTest(Experiment experiment, boolean log) {
        if(args.isOverwriteTest()) {
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
        if(!args.isEstimateTrainError()) {
            logger.finest("not estimating train error for {" + experiment.getClassifierName() + "} on {" + experiment.getDatasetName() + "}");
            return false;
        }
        if(args.isOverwriteTrain()) {
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

    private String buildClassifierResultsDirPath(Experiment experiment) {
        return args.getResultsDirPath() + "/" + experiment.getClassifierName() + "/Predictions/" + experiment.getDatasetName() +
            "/";
    }

    private String buildTrainResultsFilePath(Experiment experiment) {
        return buildClassifierResultsDirPath(experiment) + "trainFold" + experiment.getSeed() +
            ".csv";
    }

    private String buildTestResultsFilePath(Experiment experiment) {
        return buildClassifierResultsDirPath(experiment) + "testFold" + experiment.getSeed() +
            ".csv";
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
            if(args.isAppendTrainTimeContract()) {
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
            if(args.isAppendTrainMemoryContract()) {
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
            if(args.isAppendTestTimeContract()) {
                experiment
                    .setClassifierName(classifierName + "_" + testTimeContract.toString().replaceAll(" ", "_"));
            }
        } else {
            // no train contract
        }
    }

    private void appendParametersToClassifierName(Experiment experiment) {
        if(args.isAppendClassifierParameters()) {
            String origClassifierName = experiment.getClassifierName();
            String workingClassifierName = origClassifierName;
            ParamSet paramSet = experiment.getParamSet();
            String paramSetStr = "_" + paramSet.toString().replaceAll(" ", "_").replaceAll("\"", "#");
            workingClassifierName += paramSetStr;
            experiment.setClassifierName(workingClassifierName);
            logger.info("changing {" + origClassifierName + "} to {" + workingClassifierName + "}");
        }
    }
}
