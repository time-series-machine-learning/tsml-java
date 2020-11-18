package tsml.classifiers.distance_based.utils.experiment;

import com.beust.jcommander.IStringConverter;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import org.junit.Assert;
import tsml.classifiers.*;
import tsml.classifiers.distance_based.proximity.ProximityForest;
import tsml.classifiers.distance_based.proximity.ProximityForestWrapper;
import tsml.classifiers.distance_based.proximity.ProximityTree;
import tsml.classifiers.distance_based.utils.classifiers.Builder;
import tsml.classifiers.distance_based.utils.classifiers.Configurer;
import tsml.classifiers.distance_based.utils.classifiers.TrainEstimateable;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.collections.CollectionUtils;
import tsml.classifiers.distance_based.utils.strings.StrUtils;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.system.logging.Loggable;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.classifiers.distance_based.utils.system.timing.TimeAmount;
import utilities.ClassifierTools;
import utilities.FileUtils;
import weka.classifiers.Classifier;
import weka.core.Randomizable;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;

import static org.apache.commons.lang3.SystemUtils.getHostName;
import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;
import static weka.core.Debug.OFF;

public class Experiment {

    @Parameter(names = {"-c", "--classifier"}, description = "The classifier to use.")
    private String classifierName;

    @Parameter(names = {"-s", "--seed"}, description = "The seed used in resampling the data and producing random numbers in the classifier.")
    private Integer seed;

    @Parameter(names = {"-r", "--results"}, description = "The results directory.")
    private String resultsDirPath;

    @Parameter(names = {"-d", "--data"}, description = "The data directory.")
    private String dataDirPath;

    @Parameter(names = {"-p", "--problem"}, description = "The problem name. E.g. GunPoint.")
    private String problemName;

    @Parameter(names = {"--checkpoint"}, description = "Periodically save the classifier to disk. Default: off")
    private boolean checkpoint = false;

    @Parameter(names = {"--trainTimeContract"}, arity = 2, description = "Contract the classifier to build in a set time period. Give this option two arguments in the form of '--contractTrain <amount> <units>', e.g. '--contractTrain 5 minutes'")
    private List<String> trainTimeContractStrs;
    private List<TimeAmount> trainTimeContracts;

    @Parameter(names = {"-e", "--estimateTrain"}, description = "Estimate the train error. Default: false")
    private boolean estimateTrain = false;

    @Parameter(names = {"-t", "--threads"}, description = "The number of threads to use. Set to 0 or less for all available processors at runtime. Default: 1")
    private int numThreads = 1;

    @Parameter(names = {"--trainOnly"}, description = "Only train the classifier, do not test.")
    private boolean trainOnly = false;

    @Parameter(names = {"-o", "--overwrite"}, description = "Overwrite previous results.")
    private boolean overwriteResults = false;

    private static class LogLevelConverter implements IStringConverter<Level> {

        @Override public Level convert(final String s) {
            return Level.parse(s.toUpperCase());
        }
    }

    @Parameter(names = {"-l", "--logLevel"}, description = "The amount of logging. This should be set to a Java log level. Default: OFF", converter = LogLevelConverter.class)
    private Level logLevel = Level.OFF;

    private final Logger log = LogUtils.buildLogger(this);
    
    public Experiment() {
        addConfigs(ProximityForest.Config.values());
        addConfigs(ProximityTree.Config.values());
        addConfig("PF_WRAPPER", ProximityForestWrapper::new);
    }
    
    private void addConfig(String key, Builder<? extends Classifier> value) {
        classifierLookup.put(key, value);
    }
    
    private <A> void addConfigs(Iterable<A> entries) {
        for(A entry : entries) {
            if(!(entry instanceof Enum)) {
                throw new IllegalArgumentException("not an enum entry");
            }
            String key = ((Enum) entry).name();
            Builder<? extends Classifier> value;
            try {
                value = (Builder<? extends Classifier>) entry;
            } catch(ClassCastException e) {
                throw new IllegalArgumentException("not an instance of Configurer which accepts a Classifier instance");
            }
            addConfig(key, value);
        }
    }
    
    private <A> void addConfigs(A... entries) {
        addConfigs(Arrays.asList(entries));
    }
    
    private static <A, B> Map<String, Configurer<? extends Classifier>> enumToMap(Class clazz) {
        final EnumMap enumMap = new EnumMap<>(clazz);
        final Map<String, Configurer<? extends Classifier>> map = new HashMap<>();
        for(Object obj : enumMap.entrySet()) {
            Enum entry = (Enum) obj;
            map.put(entry.name(), (Configurer<? extends Classifier>) entry);
        }
        return map;
    }
    
    private void benchmark() {
        long sum = 0;
        int repeats = 30;
        long startTime = System.nanoTime();
        for(int i = 0; i < repeats; i++) {
            Random random = new Random(i);
            final double[] array = new double[1000000];
            for(int j = 0; j < array.length; j++) {
                array[j] = random.nextDouble();
            }
            Arrays.sort(array);
        }
        benchmarkScore = (System.nanoTime() - startTime) / repeats;
    }

    private void setup(String[] args) {
        JCommander.newBuilder()
                .addObject(this)
                .build()
                .parse(args);
        // check the state of this experiment is set up
        if(seed == null) throw new IllegalStateException("seed not set");
        if(classifierName == null) throw new IllegalStateException("classifier name not set");
        if(problemName == null) throw new IllegalStateException("problem name not set");
        if(dataDirPath == null) throw new IllegalStateException("data dir path not set");
        if(resultsDirPath == null) throw new IllegalStateException("results dir path not set");
        // default to no train time contracts
        if(trainTimeContractStrs == null) {
            trainTimeContracts = newArrayList(null);
        } else {
            trainTimeContracts = CollectionUtils.convertPairs(trainTimeContractStrs, TimeAmount::parse);
            // sort the train contracts in asc order
            trainTimeContracts.sort(Comparator.naturalOrder());
        }
        // default to all cpus
        if(numThreads < 1) {
            numThreads = Runtime.getRuntime().availableProcessors();
        }
    }

    public static void main(String... args) throws Exception {
        final Experiment experiment = new Experiment();
        experiment.setup(args);
        experiment.run();
    }

    private Classifier newClassifier() throws InstantiationException, IllegalAccessException {
        return classifierLookup.get(classifierName).build();
    }
    
    private final Map<String, Builder<? extends Classifier>> classifierLookup = new HashMap<>();
    private DatasetSplit split;
    private Classifier classifier;
    private String classifierNameInResults;
    private String experimentResultsDirPath;
    private String checkpointDirPath;
    private final StopWatch timer = new StopWatch();
    private final StopWatch experimentTimer = new StopWatch();
    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    private final MemoryWatcher experimentMemoryWatcher = new MemoryWatcher();
    private TimeAmount trainTimeContract;
    private long benchmarkScore;
    
    private String getExperimentResultsDirPath() {
        return StrUtils.joinPath( resultsDirPath, classifierNameInResults, "Predictions", problemName);
    }
    
    private String getClassifierNameWithTrainTimeContract() {
        return classifierName + "_" + trainTimeContract.toString().replaceAll(" ", "_");
    }
    
    private String getLockFilePath() {
        return StrUtils.joinPath(experimentResultsDirPath, "fold" + seed + ".lock");
    }
    
    private String getCheckpointDirPath() {
        return StrUtils.joinPath( resultsDirPath, classifierNameInResults, "Predictions", problemName, "fold" + seed);
    }
    
    /**
     * Runs the experiment. Make sure all fields have been set prior to this call otherwise Exceptions will be thrown accordingly.
     */
    public void run() throws Exception {
        // build the classifier
        classifier = newClassifier();
        // configure the classifier with experiment settings as necessary
        configureClassifier();
        // load the data
        log.info("loading " + problemName);
        split = new DatasetSplit(DatasetLoading.sampleDataset(dataDirPath, problemName, seed));
        log.info("benchmarking hardware");
        benchmark();
        // make a lock to deny multiple processes from writing to the same results
        FileUtils.FileLock lock = null;
        // run the experiment
        // for each train time contract
        Assert.assertFalse(trainTimeContracts.isEmpty());
        // reset the memory watchers
        experimentMemoryWatcher.resetAndStart();
        for(TimeAmount trainTimeContract : trainTimeContracts) {
            this.trainTimeContract = trainTimeContract;
            copyOverMostRecentCheckpoint();
            if(lock != null) {
                // unlock the previous lock
                lock.unlock();
            }
            // setup the contract
            setTrainTimeContract();
            // setup the results paths
            // if running multiple contracts OR no train contract set
            if(trainTimeContract != null) {
                // add the train time contract to the output dir
                classifierNameInResults = getClassifierNameWithTrainTimeContract();
            } else {
                classifierNameInResults = classifierName;
            }
            experimentResultsDirPath = getExperimentResultsDirPath();
            lock = new FileUtils.FileLock(getLockFilePath());
            if(checkpoint) {
                checkpointDirPath = getCheckpointDirPath();
                ((Checkpointable) classifier).setCheckpointPath(checkpointDirPath);
            }
            // build the classifier
            log.info("training classifier");
            timer.resetAndStart();
            memoryWatcher.start();
            if(classifier instanceof TSClassifier) {
                ((TSClassifier) classifier).buildClassifier(split.getTrainDataTS());
            } else {
                classifier.buildClassifier(split.getTrainDataArff());
            }
            timer.stop();
            memoryWatcher.stop();
            log.info("train time: " + timer.elapsedTime());
            log.info("train mem: " + memoryWatcher.getMaxMemoryUsage());
            // if estimating the train error then write out train results
            if(estimateTrain) {
                final ClassifierResults trainResults = ((TrainEstimateable) classifier).getTrainResults();
                ResultUtils.setInfo(trainResults, classifier, split.getTrainDataArff()); // todo change to ts
                setResultInfo(trainResults);
                setTrainTime(trainResults, timer);
                setMemory(trainResults, memoryWatcher);
                trainResults.findAllStatsOnce();
                log.info("train results: ");
                log.info(trainResults.writeSummaryResultsToString());
                writeResults("train", trainResults, StrUtils.joinPath(experimentResultsDirPath, "trainFold" + seed + ".csv"));
            }
            // if only training then skip the test phase
            if(trainOnly) {
                log.info("skipping testing classifier");
                continue;
            }
            // test the classifier
            log.info("testing classifier");
            timer.resetAndStart();
            memoryWatcher.resetAndStart();
            final ClassifierResults testResults = new ClassifierResults();
            // todo change to ts
//            if(classifier instanceof TSClassifier) {
//                ClassifierTools.addPredictions((TSClassifier) classifier, split.getTestDataTS(), testResults, new Random(seed));
//            } else {
                ClassifierTools.addPredictions(classifier, split.getTestDataArff(), testResults, new Random(seed));
//            }
            timer.stop();
            memoryWatcher.stop();
            log.info("test time: " + timer.elapsedTime());
            log.info("test mem: " + memoryWatcher.getMaxMemoryUsage());
            ResultUtils.setInfo(testResults, classifier, split.getTrainDataArff()); // todo ts version
            setResultInfo(testResults);
            setTrainTime(testResults, timer);
            setMemory(testResults, memoryWatcher);
            log.info("test results: ");
            log.info(testResults.writeSummaryResultsToString());
            writeResults("test", testResults, StrUtils.joinPath(experimentResultsDirPath, "testFold" + seed + ".csv"));
            if(trainTimeContract == null) {
                log.info("experiment complete");
            } else {
                log.info("train time contract " + trainTimeContract + " experiment complete");
            }
            experimentTimer.stop();
            experimentMemoryWatcher.stop();
            log.info("experiment time: " + experimentTimer.elapsedTime());
            log.info("experiment mem: " + experimentMemoryWatcher.getMaxMemoryUsage());
        }
        // unlock the lock file
        lock.unlock();
    }
    
    private void copyOverMostRecentCheckpoint() throws FileUtils.FileLock.LockException, IOException {
        // check the state of all train time contracts so far to copy over old checkpoints
        if(checkpoint) {
            TimeAmount target = trainTimeContract;
            TimeAmount mostRecentTrainTimeContract = null;
            for(TimeAmount trainTimeContract : trainTimeContracts) {
                // if there's no train time contracts then there's no prior work to begin from
                if(trainTimeContract == null) {
                    break;
                }
                // if the contract is larger than the target then can't use the progress (as spend more time than the contract)
                if(trainTimeContract.compareTo(target) > 0) {
                    break;
                }
                // check the state of the current contract
                classifierNameInResults = classifierName + trainTimeContract.toString().replaceAll(" ", "_");
                // lock
                experimentResultsDirPath = getExperimentResultsDirPath();
                final FileUtils.FileLock lock = new FileUtils.FileLock(getLockFilePath());
                checkpointDirPath = getCheckpointDirPath();
                if(!FileUtils.isEmptyDir(checkpointDirPath)) {
                    mostRecentTrainTimeContract = trainTimeContract;
                }
                lock.unlock();
            }
            // if no most recent train time contract exists then bail
            if(mostRecentTrainTimeContract == null) {
                return;
            }
            // if the most recent checkpoint is for the current contract then no copying is necessary
            if(mostRecentTrainTimeContract.equals(target)) {
                return;
            }
            // copy over the checkpoint from the most recent train time contract
            String srcCheckppintDirPath = checkpointDirPath;
            trainTimeContract = target;
            classifierNameInResults = getClassifierNameWithTrainTimeContract();
            checkpointDirPath = getCheckpointDirPath();
            Files.copy(new File(srcCheckppintDirPath).toPath(), new File(checkpointDirPath).toPath());
            log.info("copied checkpoint from previous contract: " + mostRecentTrainTimeContract);
        }
    }
    
    private void setResultInfo(ClassifierResults results) throws IOException {
        results.setFoldID(seed);
        results.setParas(results.getParas() + ",hostName," + getHostName());
        results.setBenchmarkTime(benchmarkScore);
    }
    
    private void setTrainTime(ClassifierResults results, StopWatch timer) {
        // if the build time has not been set
        if(results.getBuildTime() < 0) {
            // then set it to the build time witnessed during this experiment
            results.setBuildTime(timer.elapsedTime());
            results.setBuildPlusEstimateTime(timer.elapsedTime());
            results.setTimeUnit(TimeUnit.NANOSECONDS);
        }
    }

    private void setMemory(ClassifierResults results, MemoryWatcher memoryWatcher) {
        // if the memory usage has not been set
        if(results.getMemory() < 0) {
            // then set it to the max mem witnessed during this experiment
            results.setMemory(memoryWatcher.getMaxMemoryUsage());
        }
    }

    private void writeResults(String label, ClassifierResults results, String path) throws Exception {
        // write the train results to file, overwriting if necessary
        results.setSplit(label);
        final boolean trainResultsFileExists = new File(path).exists();
        if(trainResultsFileExists && !overwriteResults) {
            log.info(label + " results already exist");
            throw new IllegalStateException(label + " results exist");
        } else {
            log.info((trainResultsFileExists ? "overwriting" : "writing") + " " + label + " results");
            results.writeFullResultsToFile(path);
        }
    }

    private void setTrainTimeContract() {
        if(trainTimeContract != null) {
            // there is a contract
            if(classifier instanceof TrainTimeContractable) {
                log.info("setting classifier train contract to " + trainTimeContract);
                ((TrainTimeContractable) classifier).setTrainTimeLimit(trainTimeContract.getAmount(), trainTimeContract.getUnit());
            } else {
                throw new IllegalStateException("classifier cannot handle train time contract");
            }
        }  // else there is no contract, proceed as is
    }

    private void configureClassifier() {
        // set estimate train error
        if(estimateTrain) {
            if(classifier instanceof TrainEstimateable) {
                log.info("setting classifier to estimate train error");
                ((TrainEstimateable) classifier).setEstimateOwnPerformance(true);
            } else {
                throw new IllegalStateException("classifier cannot estimate the train error");
            }
        }
        // set checkpointing
        if(checkpointDirPath != null) {
            if(classifier instanceof Checkpointable) {
                log.info("setting classifier to checkpoint to " + checkpointDirPath);
                ((Checkpointable) classifier).setCheckpointPath(checkpointDirPath);
            } else {
                throw new IllegalStateException("classifier cannot checkpoint");
            }
        }
        // set log level
        if(classifier instanceof Loggable) {
            log.info("setting classifier log level to " + logLevel);
            ((Loggable) classifier).setLogLevel(logLevel);
        } else if(classifier instanceof EnhancedAbstractClassifier) {
            boolean debug = !logLevel.equals(OFF);
            log.info("setting classifier debug to " + debug);
            ((EnhancedAbstractClassifier) classifier).setDebug(debug);
        } else {
            if(!logLevel.equals(Level.OFF)) {
                log.info("classifier does not support logging");
            }
        }
        // set seed
        if(classifier instanceof Randomizable) {
            log.info("setting classifier seed to " + seed);
            ((Randomizable) classifier).setSeed(seed);
        } else {
            log.info("classifier does not accept a seed");
        }
        // set threads
        if(classifier instanceof MultiThreadable) {
            ((MultiThreadable) classifier).enableMultiThreading(numThreads);
        }
    }
}
