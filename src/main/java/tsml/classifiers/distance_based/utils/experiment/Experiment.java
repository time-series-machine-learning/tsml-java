package tsml.classifiers.distance_based.utils.experiment;

import evaluation.storage.ClassifierResults;
import experiments.ExperimentalArguments;
import experiments.ClassifierExperiments;
import experiments.data.DatasetLoading;
import tsml.classifiers.*;
import tsml.classifiers.distance_based.proximity.ProximityForest;
import tsml.classifiers.distance_based.proximity.ProximityForestWrapper;
import tsml.classifiers.distance_based.utils.classifiers.*;
import tsml.classifiers.distance_based.utils.classifiers.checkpointing.Checkpointed;
import tsml.classifiers.distance_based.utils.classifiers.configs.Builder;
import tsml.classifiers.distance_based.utils.classifiers.configs.Config;
import tsml.classifiers.distance_based.utils.classifiers.configs.Configs;
import tsml.classifiers.distance_based.utils.classifiers.contracting.ContractedTrain;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.system.copy.Copier;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.system.logging.Loggable;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import utilities.ClassifierTools;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Randomizable;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;

import static tsml.classifiers.distance_based.utils.system.SysUtils.hostName;
import static utilities.FileUtils.*;
import static weka.core.Debug.OFF;

public class Experiment implements Copier {
    
    public Experiment(String[] args) {
        this(new ExperimentConfig(args));
    }
    
    public Experiment(ExperimentConfig config) {
        this.config = Objects.requireNonNull(config);
        setExperimentLogLevel(Level.ALL);
        
        addClassifierConfigs(ProximityForest.CONFIGS);
        addClassifierConfigs(ProximityForestWrapper.CONFIGS);
    }
    
    private void addClassifierConfig(Config<? extends Classifier> config) {
        if(classifierLookup.containsKey(config.name())) {
            throw new IllegalStateException("already contains classifier config for " + config.name());
        }
        classifierLookup.put(config.name(), config);
    }
    
    private void addClassifierConfigs(Configs<? extends Classifier> configs) {
        configs.forEach((Consumer<Config<? extends Classifier>>) this::addClassifierConfig);
    }
    
    private void benchmarkHardware() {
        log.info("benchmarking hardware");
        // delegate to the benchmarking system from main experiments code. This maintains consistency across benchmarks, but it substantially quicker and therefore less reliable of a benchmark. todo talks to james about merging these
        ExperimentalArguments args = new ExperimentalArguments();
        args.performTimingBenchmark = true;
        benchmarkScore = ClassifierExperiments.findBenchmarkTime(args);
//        long sum = 0;
//        int repeats = 30;
//        long startTime = System.nanoTime();
//        for(int i = 0; i < repeats; i++) {
//            Random random = new Random(i);
//            final double[] array = new double[1000000];
//            for(int j = 0; j < array.length; j++) {
//                array[j] = random.nextDouble();
//            }
//            Arrays.sort(array);
//        }
//        benchmarkScore = (System.nanoTime() - startTime) / repeats;
    }

    public static void main(String... args) throws Exception {
        new Experiment(args).run();
    }

    private void buildClassifier() {
        log.info("creating new instance of " + config.getClassifierName());
        final Builder<? extends Classifier> builder = classifierLookup.get(config.getClassifierName());
        if(builder == null) {
            throw new NoSuchElementException(config.getClassifierName() + " not found");
        }
        classifier = builder.build();
        if(classifier instanceof TSClassifier) {
            tsClassifier = (TSClassifier) classifier;
        } else {
            tsClassifier = TSClassifier.wrapClassifier(classifier);
        }
    }
    
    private final Map<String, Builder<? extends Classifier>> classifierLookup = new TreeMap<>();
    private TimeSeriesInstances trainData;
    private TimeSeriesInstances testData;
    private Classifier classifier;
    private TSClassifier tsClassifier;
    private final StopWatch timer = new StopWatch();
    private final StopWatch experimentTimer = new StopWatch();
    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    private final MemoryWatcher experimentMemoryWatcher = new MemoryWatcher();
    private long benchmarkScore;
    private final Logger log = LogUtils.getLogger(this);
    private ExperimentConfig config;
    private ExperimentConfig previousConfig;
    private Map<String, FileLock> locks;
    
    private void loadData() throws Exception {
        final Instances[] instances =
                DatasetLoading.sampleDataset(config.getDataDirPath(), config.getProblemName(), config.getSeed());
        trainData = Converter.fromArff(instances[0]);
        testData = Converter.fromArff(instances[1]);
        // load the data
        log.info("loading " + config.getProblemName() + " dataset ");
    }
    
    private FileLock getLock(String path) throws FileLock.LockException {
        FileLock lock = locks.get(path);
        if(lock == null) {
            lock = new FileLock(path, false);
            locks.put(path, lock);
        }
        return lock;
    }
    
    /**
     * Runs the experiment. Make sure all fields have been set prior to this call otherwise Exceptions will be thrown accordingly.
     */
    public void run() throws Exception {
        locks = new HashMap<>();
        // build the classifier
        buildClassifier();
        // configure the classifier with experiment settings as necessary
        configureClassifier();
        loadData();
        benchmarkHardware();
        // reset the memory watchers
        experimentTimer.resetAndStart();
        experimentMemoryWatcher.resetAndStart();
        // iterate over every train time limit
        final List<TimeSpan> trainTimeLimits = config.getTrainTimeLimits();
        for(int trainTimeLimitIndex = 0; trainTimeLimitIndex < trainTimeLimits.size(); trainTimeLimitIndex++) {
            final TimeSpan trainTimeLimit = trainTimeLimits.get(trainTimeLimitIndex);
            config.setTrainTimeLimit(trainTimeLimit);
            // attempt to lock the current config and maintain lock on previous
            final FileLock lock = getLock(config);
            lock.lock();
            // run checks to see if files are in order (e.g. results don't exist / can be overwritten)
            checkTestResultsExistence();
            checkTrainResultsExistence();
            // setup the contract
            setupTrainTimeContract();
            // setup checkpointing config
            setupCheckpointing();
            // copy over any previous checkpoints from previous contracts (and optionally remove previous checkpoint post copy)
            copyMostRecentCheckpoint(trainTimeLimitIndex);
            // train the classifier
            train();
            // test the classifier
            test();
            // stop the classifier from rebuilding on next buildClassifier call
            if(classifier instanceof Rebuildable) {
                ((Rebuildable) classifier).setRebuild(false);
            } else if(config.getTrainTimeLimits().size() > 1) {
                log.warning("cannot disable rebuild on " + config.getClassifierNameInResults() +
                                    ", therefore it will be rebuilt entirely for every train time contract");
            }
            if(trainTimeLimitIndex >= trainTimeLimits.size() - 1) {
                // optionally remove the checkpoint as it's the last one
                optionallyRemoveCheckpoint(config);
            }
            if(config.getTrainTimeLimit() == null) {
                log.info(config.getClassifierNameInResults() + " experiment complete");
            } else {
                log.info(config.getClassifierNameInResults() + " experiment complete under train time contract " + config.getTrainTimeLimit().label());
            }
            // release lock
            lock.unlock();
            previousConfig = config;
        }
        // clear any previously held locks / check they've been released
        locks.forEach((k, v) -> {
            if(v.isLocked()) {
                throw new IllegalStateException("all locks should have been released: " + k);
            }
        });
        locks = null;
        // experiment complete
        experimentTimer.stop();
        experimentMemoryWatcher.stop();
        log.info("experiment time: " + experimentTimer.toTimeSpan());
        log.info("experiment mem: " + experimentMemoryWatcher.getMaxMemoryUsage());
    }

    /**
     * I.e. if we're currently preparing to run a 3h contract and previously a 1h and 2h have been run, we should check the 1h and 2h workspace for checkpoint files. If there are no checkpoint files for 2h but there are for 1h, copy them into the checkpoint dir folder to resume from the 1h contract end point.
     * @throws IOException
     */
    private void copyMostRecentCheckpoint(int trainTimeLimitIndex) throws IOException {
        if(isModelFullyBuiltFromPreviousRun()) {
            log.info("skipping setup recent checkpoints for " + config.getClassifierNameInResults() + " as model fully built");
            return;
        }
        // check the state of all train time contracts so far to copy over old checkpoints
        if(config.isCheckpoint()) {
            // if there's no train contract then bail
            if(config.getTrainTimeLimit() == null) {
                return;
            }
            // check whether the checkpoint dir is empty. If not, then we already have a checkpoint to work from, i.e. no need to copy a checkpoint from a lesser contract.
            if(!isEmptyDir(config.getCheckpointDirPath())) {
                log.info("checkpoint already exists in " + config.getCheckpointDirPath() + " , not copying from previous train time limit runs");
                return;
            }
            // if the checkpoint dir is empty then there's no usable checkpoints
            final FileLock lock = getLock(previousConfig);
            lock.lock();
            if(!isEmptyDir(previousConfig.getCheckpointDirPath())) {
                // if a previous checkpoint has been located, copy the contents into the checkpoint dir for this contract time run
                log.info("checkpoint found in " + previousConfig.getClassifierNameInResults() + " workspace");
                final String src = previousConfig.getCheckpointDirPath();
                final String dest = config.getCheckpointDirPath();
                log.info("copying checkpoint contents from " + src + " to " + dest);
                makeDir(config.getCheckpointDirPath());
                copy(src, dest);
                // optionally remove the checkpoint now it's copied to a new location
                optionallyRemoveCheckpoint(previousConfig);
            } else {
                log.info("no checkpoints found from previous contracts");
            }
            lock.unlock();
        }
    }
    
    private void setResultInfo(ClassifierResults results) throws IOException {
        results.setFoldID(config.getSeed());
        String paras = results.getParas();
        if(!paras.isEmpty()) {
            paras += ",";
        }
        paras += "hostName," + hostName();
        results.setParas(paras);
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
    
    private void checkTrainResultsExistence() {
        if(config.isEvaluateClassifier()) {
            checkResultsExistence("train", config.getTrainFilePath());
        }
    }
    
    private void checkTestResultsExistence() {
        if(!config.isTrainOnly()) {
            checkResultsExistence("test", config.getTestFilePath());
        }
    }

    private void writeResults(String label, ClassifierResults results, String path) throws Exception {
        // write the train results to file, overwriting if necessary
        final boolean exists = checkResultsExistence(label, path);
        results.setSplit(label);
        log.info((exists ? "overwriting" : "writing") + " " + label + " results");
        results.writeFullResultsToFile(path);
    }
    
    private boolean checkResultsExistence(String label, String path) {
        final boolean exists = new File(path).exists();
        if(exists && !config.isOverwriteResults()) {
            throw new IllegalStateException(label + " results exist at " + path);
        }
        return exists;
    }

    private void setupTrainTimeContract() {
        if(isModelFullyBuiltFromPreviousRun()) {
            log.info("skipping setup contract for " + config.getClassifierNameInResults() + " as model fully built");
            return;
        }
        if(config.getTrainTimeLimit() != null) {
            // there is a contract
            if(classifier instanceof TrainTimeContractable) {
                log.info("setting " + config.getClassifierName() + " train contract to " + config.getTrainTimeLimit() + " : " + config.getClassifierNameInResults());
                ((TrainTimeContractable) classifier).setTrainTimeLimit(config.getTrainTimeLimit().inNanos());
            } else {
                throw new IllegalStateException("classifier cannot handle train time contract");
            }
        }  // else there is no contract, proceed as is
    }

    private void configureClassifier() {
        // set estimate train error
        if(config.isEvaluateClassifier()) {
            if(classifier instanceof TrainEstimateable) {
                log.info("setting " + config.getClassifierNameInResults() + " to estimate train error");
                ((TrainEstimateable) classifier).setEstimateOwnPerformance(true);
                if(classifier instanceof EnhancedAbstractClassifier) {
                    EnhancedAbstractClassifier eac = (EnhancedAbstractClassifier) classifier;
                    // default to a cv if not set when building the classifier
                    if(eac.getEstimatorMethod().equalsIgnoreCase("none")) {
                        eac.setTrainEstimateMethod("cv");
                    }
                }
            } else {
                throw new IllegalStateException("classifier cannot evaluate the train error");
            }
        }
        // set log level
        if(classifier instanceof Loggable) {
            log.info("setting " + config.getClassifierNameInResults() + " log level to " + config.getLogLevel());
            ((Loggable) classifier).setLogLevel(config.getLogLevel());
        } else if(classifier instanceof EnhancedAbstractClassifier) {
            boolean debug = !config.getLogLevel().equals(OFF);
            log.info("setting " + config.getClassifierNameInResults() + " debug to " + debug);
            ((EnhancedAbstractClassifier) classifier).setDebug(debug);
        } else {
            if(!config.getLogLevel().equals(Level.OFF)) {
                log.info("classifier does not support logging");
            }
        }
        // set seed
        if(classifier instanceof Randomizable) {
            log.info("setting " + config.getClassifierNameInResults() + " seed to " + config.getSeed());
            ((Randomizable) classifier).setSeed(config.getSeed());
        } else {
            log.info("classifier does not accept a seed");
        }
        // set threads
        if(classifier instanceof MultiThreadable) {
            log.info("setting " + config.getClassifierNameInResults() + " to use " + config.getNumThreads() + " threads");
            ((MultiThreadable) classifier).enableMultiThreading(config.getNumThreads());
        } else if(config.getNumThreads() != 1) {
            log.info("classifier cannot use multiple threads");
        }
        // todo mem
//        // set memory
//        if(classifier instanceof MemoryContractable) {
//            ((MemoryContractable) classifier).setMemoryLimit();
//        }
    }
    
    private boolean isModelFullyBuiltFromPreviousRun() {
        // skip if the model is fully built from a previous contract
        return previousConfig != null && classifier instanceof ContractedTrain && ((ContractedTrain) classifier).isFullyBuilt();
    }
    
    private void train() throws Exception {
        if(isModelFullyBuiltFromPreviousRun()) {
            log.info("skipping training " + config.getClassifierNameInResults() + " as model fully built");
            String src = previousConfig.getTrainFilePath();
            String dest = config.getTrainFilePath();
            if(Files.exists(Paths.get(src))) {
                log.info("copying " + src + " to " + dest);
                copy(src, dest);
            }
            return;
        }
        // build the classifier
        log.info("training " + config.getClassifierNameInResults());
        timer.start();
        memoryWatcher.start();
        tsClassifier.buildClassifier(trainData);
        memoryWatcher.stop();
        timer.stop();
        log.info("train time: " + timer.toTimeSpan());
        log.info("train mem: " + memoryWatcher.getMaxMemoryUsage());
        // if estimating the train error then write out train results
        if(config.isEvaluateClassifier()) {
            final ClassifierResults trainResults = ((TrainEstimateable) classifier).getTrainResults();
            ResultUtils.setInfo(trainResults, tsClassifier, trainData);
            setResultInfo(trainResults);
            setTrainTime(trainResults, timer);
            setMemory(trainResults, memoryWatcher);
            trainResults.findAllStatsOnce();
            log.info("train results: ");
            log.info(trainResults.writeSummaryResultsToString());
            writeResults("train", trainResults, config.getTrainFilePath());
        }
        log.info(config.getClassifierNameInResults() + " training complete");
    }
    
    private void test() throws Exception {
        // if only training then skip the test phase
        if(config.isTrainOnly()) {
            log.info("skipping testing classifier");
            return;
        }
        if(isModelFullyBuiltFromPreviousRun()) {
            log.info("skipping testing " + config.getClassifierNameInResults() + " as model fully built");
            String src = previousConfig.getTestFilePath();
            String dest = config.getTestFilePath();
            log.info("copying " + src + " to " + dest);
            copy(src, dest);
            return;
        }
        // test the classifier
        log.info("testing " + config.getClassifierNameInResults());
        timer.resetAndStart();
        memoryWatcher.resetAndStart();
        final ClassifierResults testResults = new ClassifierResults();
        ClassifierTools.addPredictions(tsClassifier, testData, testResults, new Random(config.getSeed()));
        timer.stop();
        memoryWatcher.stop();
        log.info("test time: " + timer.toTimeSpan());
        log.info("test mem: " + memoryWatcher.getMaxMemoryUsage());
        ResultUtils.setInfo(testResults, tsClassifier, trainData);
        setResultInfo(testResults);
        setTrainTime(testResults, timer);
        setMemory(testResults, memoryWatcher);
        log.info("test results: ");
        log.info(testResults.writeSummaryResultsToString());
        writeResults("test", testResults, config.getTestFilePath());
        log.info(config.getClassifierNameInResults() + " testing complete");
    }
    
    private void setupCheckpointing() throws IOException {
        if(config.isCheckpoint()) {
            if(classifier instanceof Checkpointable) {
                // the copy over the most suitable checkpoint from another run if exists
                log.info("setting checkpoint path for " + config.getClassifierNameInResults() + " to " + config.getCheckpointDirPath());
                ((Checkpointable) classifier).setCheckpointPath(config.getCheckpointDirPath());
            } else {
                log.info(config.getClassifierNameInResults() + " cannot produce checkpoints");
            }
            if(classifier instanceof Checkpointed) {
                if(config.getCheckpointInterval() != null) {
                    log.info("setting checkpoint interval for " + config.getClassifierNameInResults() + " to " + config.getCheckpointInterval());
                    ((Checkpointed) classifier).setCheckpointInterval(config.getCheckpointInterval());
                }
                if(config.isKeepCheckpoints()) {
                    log.info("setting keep all checkpoints for " + config.getClassifierNameInResults());
                }
                ((Checkpointed) classifier).setKeepCheckpoints(config.isKeepCheckpoints());
            }
        }
    }
    
    public Level getExperimentLogLevel() {
        return log.getLevel();
    }

    public void setExperimentLogLevel(final Level level) {
        log.setLevel(level);
    }
    
    public ExperimentConfig getConfig() {
        return config;
    }
    
    private void optionallyRemoveCheckpoint(ExperimentConfig config) throws IOException {
        if(config == null) {
            return;
        }
        if(config.isCheckpoint()) {
            // optionally remove the checkpoint
            if(config.isRemoveCheckpoint()) {
                log.info("deleting checkpoint at " + config.getCheckpointDirPath());
                final FileLock lock = getLock(config);
                lock.lock();
                delete(config.getCheckpointDirPath());
                lock.unlock();
            }
        }
    }
    
    private FileLock getLock(ExperimentConfig config) {
        return getLock(config.getLockFilePath());
    }
}
