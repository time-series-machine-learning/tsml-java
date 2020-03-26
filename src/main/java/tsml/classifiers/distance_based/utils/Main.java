package tsml.classifiers.distance_based.utils;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.internal.Lists;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import tsml.classifiers.distance_based.utils.classifier_building.ClassifierBuilderFactory;
import tsml.classifiers.distance_based.utils.classifier_building.ClassifierBuilderFactory.ClassifierBuilder;
import tsml.classifiers.distance_based.utils.logging.LogUtils;
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
    @Parameter(names = {"-c", "--classifier"}, description = "todo", required = true)
    private List<String> classifierNames = new ArrayList<>();

    @Parameter(names = {"--datasetsDir", "-p"}, description = "todo", required = true)
    private List<String> datasetDirPaths = new ArrayList<>();

    @Parameter(names = {"--datasetName", "-d"}, description = "todo", required = true)
    private List<String> datasetNames = new ArrayList<>();

    @Parameter(names = {"--seed", "-s"}, description = "todo", required = true)
    private List<Integer> seeds = Lists.newArrayList(0);

    @Parameter(names = {"-r", "--resultsDir"}, description = "todo", required = true)
    private String resultsDirPath = null;

    // todo classifier params

    @Parameter(names = "--trainContract", arity = 2, description = "todo")
    private List<String> trainContractStrs = new ArrayList<>();
    private List<TimeAmount> trainContracts = new ArrayList<>();

    @Parameter(names = "--checkpoint", description = "todo")
    private boolean checkpoint = false;

    @Parameter(names = "--threads", description = "todo")
    private int numThreads = 1; // <=0 for all available threads

    @Parameter(names = "--trainContractInName", description = "todo")
    private boolean appendTrainContractToClassifierName = false;

    @Parameter(names = {"--estimateTrainError"}, description = "todo")
    private boolean estimateTrainError = false;

    @Parameter(names = {"-l", "--logLevel"}, description = "todo")
    private String logLevel = null;

    private final Logger logger = LogUtils.buildLogger(this);

    private ClassifierBuilderFactory<Classifier> classifierBuilderFactory =
        ClassifierBuilderFactory.getGlobalInstance(); // todo get this by string, i.e. factory

    public static void main(String ... args) {
//        new Main(args).runExperiments();
        System.out.println(
            ClassifierBuilderFactory.getGlobalInstance().getClassifierNames());
    }

    public static class Runner {

        public static void main(String[] args) {
            Main.main(
                "--threads", "1",
                "-r", "results",
                "-s", "0",
                "-c", "ED_1NN",
                "--estimateTrainError",
                "-d", "GunPoint",
                "-p", "/bench/datasets"
            );
        }
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
        if(trainContractStrs.size() % 2 != 0) {
            throw new IllegalStateException("train contracts must be a list of pairs, i.e. \"5\" \"minutes\"");
        }
        trainContracts = new ArrayList<>();
        for(int i = 0; i < trainContractStrs.size(); i += 2) {
            final String trainContractAmountStr = trainContractStrs.get(i);
            final String trainContractUnitStr = trainContractStrs.get(i + 1);
            final TimeAmount trainContract = new TimeAmount(Long.parseLong(trainContractAmountStr), TimeUnit.valueOf(trainContractUnitStr));
            trainContracts.add(trainContract);
        }
        Collections.sort(trainContracts);
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
            ", trainContracts=" + trainContractStrs +
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
        if(trainContracts.isEmpty()) {
            trainContracts.add(null);
        }
        final String origClassifierName = experiment.getClassifierName();
        // for each train contract (pair of strs, one for the amount, one for the unit)
        for(TimeAmount trainContract : trainContracts) {
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
                if(appendTrainContractToClassifierName) {
                    experiment.setClassifierName(origClassifierName + "_" + trainContract.toString().replaceAll(" ", "_"));
                }
            }
            // train classifier
            experiment.train();
            // write train results if enabled
            final String classifierResultsDirPath =
                resultsDirPath + "/" + experiment.getClassifierName() + "/" + experiment.getDatasetName() + "/";
            if(experiment.isEstimateTrain()) {
                final String trainResultsFilePath = classifierResultsDirPath + "trainFold" + experiment.getSeed() +
                    ".csv";
                final ClassifierResults trainResults = experiment.getTrainResults();
                FileUtils.writeToFile(trainResults.writeFullResultsToString(), trainResultsFilePath);
            }
            // test classifier
            experiment.test();
            // write test results
            final String testResultsFilePath = classifierResultsDirPath + "trainFold" + experiment.getSeed() +
                ".csv";
            final ClassifierResults testResults = experiment.getTestResults();
            FileUtils.writeToFile(testResults.writeFullResultsToString(), testResultsFilePath);
            // shallow copy experiment so we can reuse the configuration under the next train contract
            experiment = (Experiment) experiment.shallowCopy();
        }
    }



}
