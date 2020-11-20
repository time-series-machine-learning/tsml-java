package tsml.classifiers.distance_based.utils.experiment;

import com.beust.jcommander.JCommander;
import evaluation.storage.ClassifierResults;
import experiments.Experiments;
import experiments.data.DatasetLoading;
import org.junit.Assert;
import tsml.classifiers.*;
import tsml.classifiers.distance_based.proximity.ProximityForest;
import tsml.classifiers.distance_based.proximity.ProximityForestWrapper;
import tsml.classifiers.distance_based.proximity.ProximityTree;
import tsml.classifiers.distance_based.utils.classifiers.*;
import tsml.classifiers.distance_based.utils.classifiers.results.ResultUtils;
import tsml.classifiers.distance_based.utils.strings.StrUtils;
import tsml.classifiers.distance_based.utils.system.logging.LogUtils;
import tsml.classifiers.distance_based.utils.system.logging.Loggable;
import tsml.classifiers.distance_based.utils.system.memory.MemoryWatcher;
import tsml.classifiers.distance_based.utils.system.timing.StopWatch;
import utilities.ClassifierTools;
import utilities.FileUtils;
import weka.classifiers.Classifier;
import weka.core.PropertyPath;
import weka.core.Randomizable;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import static tsml.classifiers.distance_based.utils.system.SysUtils.hostName;
import static weka.core.Debug.OFF;

public class Experiment implements Copier {
    
    public Experiment(String[] args) {
        this(new ExperimentConfig(args));
    }
    
    public Experiment(ExperimentConfig config) {
        this.config = Objects.requireNonNull(config);
        setExperimentLogLevel(Level.ALL);
        addClassifierConfigs(ProximityForest.Config.values());
        addClassifierConfigs(ProximityTree.Config.values());
        Builder<ProximityForestWrapper> pfwR5 = () -> {
            ProximityForestWrapper pfw = new ProximityForestWrapper();
            pfw.setR(5);
            return pfw;
        };
        addClassifierConfig("PF_WRAPPER", pfwR5);
        addClassifierConfig("PF_WRAPPER_R5", pfwR5);
        addClassifierConfig("PF_WRAPPER_R1",  () -> {
            ProximityForestWrapper pfw = new ProximityForestWrapper();
            pfw.setR(1);
            return pfw;
        });
        addClassifierConfig("PF_WRAPPER_R10",  () -> {
            ProximityForestWrapper pfw = new ProximityForestWrapper();
            pfw.setR(10);
            return pfw;
        });
    }
    
    private void addClassifierConfig(String key, Builder<? extends Classifier> value) {
        classifierLookup.put(key, value);
    }
    
    private <A> void addClassifierConfigs(Iterable<A> entries) {
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
            addClassifierConfig(key, value);
        }
    }
    
    private <A> void addClassifierConfigs(A... entries) {
        addClassifierConfigs(Arrays.asList(entries));
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
        // delegate to the benchmarking system from main experiments code. This maintains consistency across benchmarks, but it substantially quicker and therefore less reliable of a benchmark. todo talks to james about merging these
        Experiments.ExperimentalArguments args = new Experiments.ExperimentalArguments();
        args.performTimingBenchmark = true;
        benchmarkScore = Experiments.findBenchmarkTime(args);
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

    private Classifier newClassifier() throws InstantiationException, IllegalAccessException {
        log.info("creating new instance of " + config.getClassifierName());
        return classifierLookup.get(config.getClassifierName()).build();
    }
    
    private final Map<String, Builder<? extends Classifier>> classifierLookup = new HashMap<>();
    private DatasetSplit split;
    private Classifier classifier;
    private final StopWatch timer = new StopWatch();
    private final StopWatch experimentTimer = new StopWatch();
    private final MemoryWatcher memoryWatcher = new MemoryWatcher();
    private final MemoryWatcher experimentMemoryWatcher = new MemoryWatcher();
    private TimeSpan trainTimeLimit;
    private long benchmarkScore;
    private final Logger log = LogUtils.buildLogger(this);
    private ExperimentConfig config;
    
    /**
     * Runs the experiment. Make sure all fields have been set prior to this call otherwise Exceptions will be thrown accordingly.
     */
    public void run() throws Exception {
        // build the classifier
        classifier = newClassifier();
        // configure the classifier with experiment settings as necessary
        configureClassifier();
        // load the data
        log.info("loading " + config.getProblemName() + " dataset ");
        split = new DatasetSplit(DatasetLoading.sampleDataset(config.getDataDirPath(), config.getProblemName(), config.getSeed()));
        log.info("benchmarking hardware");
        benchmark();
        // make a lock to deny multiple processes from writing to the same results
        FileUtils.FileLock lock = null;
        // reset the memory watchers
        experimentTimer.resetAndStart();
        experimentMemoryWatcher.resetAndStart();
        Experiment previousExperiment;
        for(TimeSpan trainTimeLimit : config.getTrainTimeLimits()) {
            this.trainTimeLimit = trainTimeLimit;
            if(lock != null) {
                // unlock the previous lock
                log.info("unlocking " + config.getLockFilePath());
                lock.unlock();
            }
            // lock the output file to ensure only this experiment is writing results
            log.info("locking " + config.getLockFilePath());
            lock = new FileUtils.FileLock(config.getLockFilePath());
            // setup the contract
            setupTrainTimeContract();
            // setup checkpointing config
            setupCheckpointing();
            // train the classifier
            train();
            // test the classifier
            test();
            // stop the classifier from rebuilding on next buildClassifier call
            if(classifier instanceof Rebuildable) {
                ((Rebuildable) classifier).setRebuild(false);
            } else if(config.getTrainTimeLimits().size() > 1) {
                log.warning("cannot disable rebuild on " + config.getClassifierNameInResults() + ", therefore it will be rebuilt entirely for every train time contract");
            }
        }
        experimentTimer.stop();
        experimentMemoryWatcher.stop();
        log.info("experiment time: " + experimentTimer.toDuration());
        log.info("experiment mem: " + experimentMemoryWatcher.getMaxMemoryUsage());
        // unlock the lock file
        if(lock != null) {
            log.info("unlocking " + config.getLockFilePath());
            lock.unlock();
        }
    }

    /**
     * I.e. if we're currently preparing to run a 3h contract and previously a 1h and 2h have been run, we should check the 1h and 2h workspace for checkpoint files. If there are no checkpoint files for 2h but there are for 1h, copy them into the checkpoint dir folder to resume from the 1h contract end point.
     * @throws FileUtils.FileLock.LockException
     * @throws IOException
     */
    private void copyMostRecentCheckpoint() throws FileUtils.FileLock.LockException, IOException {
        // check the state of all train time contracts so far to copy over old checkpoints
        if(config.isCheckpoint()) {
            // check whether the checkpoint dir is empty. If not, then we already have a checkpoint to work from, i.e. no need to copy a checkpoint from a lesser contract.
            if(!FileUtils.isEmptyDir(config.getCheckpointDirPath())) {
                log.info("checkpoint found in workspace");
                return;
            }
            // the target train time limit we'll be running next
            TimeSpan target = trainTimeLimit;
            // the nearest traim time limit WITH a checkpoint
            TimeSpan mostRecentTrainTimeContract = null;
            // copy this experiment to reconfigure for other contracts
            final ExperimentConfig otherConfig = config.shallowCopy();
            // for every train time limit which is less than the target
            final List<TimeSpan>
                    timeSpans = config.getTrainTimeLimits().stream().filter(timeSpan -> target.compareTo(timeSpan) > 0).sorted(Comparator.reverseOrder()).collect(Collectors.toList());
            // examine the times in descending order, attempting to locate the most recent checkpoint
            for(TimeSpan timeSpan : timeSpans) {
                // set the dummy experiment's ttl
                otherConfig.setTrainTimeLimit(timeSpan);
                // lock the checkpoints to ensure we're the only user
                try(FileUtils.FileLock lock = new FileUtils.FileLock(otherConfig.getLockFilePath())) {
                    // if the checkpoint dir is empty then there's no usable checkpoints
                    if(!FileUtils.isEmptyDir(otherConfig.getCheckpointDirPath())) {
                        // otherwise this is the most recent usable checkpoint, no need to keep searching
                        mostRecentTrainTimeContract = trainTimeLimit;
                        break;
                    }
                } catch(Exception e) {
                    // failed to lock the checkpoint dir, in use by another process.Continue looking for other checkpoints
                }
            }
            // if a previous checkpoint has been located, copy the contents into the checkpoint dir for this contract time run
            if(mostRecentTrainTimeContract != null) {
                log.info("checkpoint found in " + otherConfig.getClassifierNameInResults() + " workspace");
                final String src = otherConfig.getCheckpointDirPath();
                final String dest = config.getCheckpointDirPath();
                log.info("coping checkpoint contents from " + src + " to " + dest);
                Files.copy(Paths.get(src), Paths.get(dest));
                // optionally remove the checkpoint now it's copied to a new location
                if(otherConfig.isRemoveCheckpoint()) {
                    Files.delete(Paths.get(otherConfig.getCheckpointDirPath()));   
                }
            } else {
                log.info("no checkpoints found from previous contracts");
            }
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

    private void writeResults(String label, ClassifierResults results, String path) throws Exception {
        // write the train results to file, overwriting if necessary
        results.setSplit(label);
        final boolean trainResultsFileExists = new File(path).exists();
        if(trainResultsFileExists && !config.isOverwriteResults()) {
            log.info(label + " results already exist");
            throw new IllegalStateException(label + " results exist");
        } else {
            log.info((trainResultsFileExists ? "overwriting" : "writing") + " " + label + " results");
            results.writeFullResultsToFile(path);
        }
    }

    private void setupTrainTimeContract() {
        if(trainTimeLimit != null) {
            // there is a contract
            if(classifier instanceof TrainTimeContractable) {
                log.info("setting " + config.getClassifierName() + " train contract to " + trainTimeLimit + " : " + config.getClassifierNameInResults());
                ((TrainTimeContractable) classifier).setTrainTimeLimit(trainTimeLimit.inNanos());
            } else {
                throw new IllegalStateException("classifier cannot handle train time contract");
            }
        }  // else there is no contract, proceed as is
    }

    private void configureClassifier() {
        // set estimate train error
        if(config.isEvaluateClassifier()) {
            if(classifier instanceof TrainEstimateable) {
                log.info("setting " + config.getClassifierName() + " to estimate train error");
                ((TrainEstimateable) classifier).setEstimateOwnPerformance(true);
            } else {
                throw new IllegalStateException("classifier cannot evaluate the train error");
            }
        }
        // set log level
        if(classifier instanceof Loggable) {
            log.info("setting " + config.getClassifierName() + " log level to " + config.getLogLevel());
            ((Loggable) classifier).setLogLevel(config.getLogLevel());
        } else if(classifier instanceof EnhancedAbstractClassifier) {
            boolean debug = !config.getLogLevel().equals(OFF);
            log.info("setting " + config.getClassifierName() + " debug to " + debug);
            ((EnhancedAbstractClassifier) classifier).setDebug(debug);
        } else {
            if(!config.getLogLevel().equals(Level.OFF)) {
                log.info("classifier does not support logging");
            }
        }
        // set seed
        if(classifier instanceof Randomizable) {
            log.info("setting " + config.getClassifierName() + " seed to " + config.getSeed());
            ((Randomizable) classifier).setSeed(config.getSeed());
        } else {
            log.info("classifier does not accept a seed");
        }
        // set threads
        if(classifier instanceof MultiThreadable) {
            log.info("setting " + config.getClassifierName() + " to use " + config.getNumThreads() + " threads");
            ((MultiThreadable) classifier).enableMultiThreading(config.getNumThreads());
        } else if(config.getNumThreads() != 1) {
            log.info("classifier cannot use multiple threads");
        }
    }
    
    private void train() throws Exception {
        // build the classifier
        log.info("training " + config.getClassifierNameInResults());
        timer.start();
        memoryWatcher.start();
        // prompt garbage collection to provide a clean slate before building
        System.gc();
        if(classifier instanceof TSClassifier) {
            ((TSClassifier) classifier).buildClassifier(split.getTrainDataTS());
        } else {
            classifier.buildClassifier(split.getTrainDataArff());
        }
        // prompt garbage collection to obtain at least one memory usage reading during training
        System.gc();
        timer.stop();
        memoryWatcher.stop();
        log.info("train time: " + timer.toDuration());
        log.info("train mem: " + memoryWatcher.getMaxMemoryUsage());
        // if estimating the train error then write out train results
        if(config.isEvaluateClassifier()) {
            final ClassifierResults trainResults = ((TrainEstimateable) classifier).getTrainResults();
            ResultUtils.setInfo(trainResults, classifier, split.getTrainDataArff()); // todo change to ts
            setResultInfo(trainResults);
            setTrainTime(trainResults, timer);
            setMemory(trainResults, memoryWatcher);
            trainResults.findAllStatsOnce();
            log.info("train results: ");
            log.info(trainResults.writeSummaryResultsToString());
            writeResults("train", trainResults, config.getTrainFilePath());
        }
    }
    
    private void test() throws Exception {
        // if only training then skip the test phase
        if(config.isTrainOnly()) {
            log.info("skipping testing classifier");
            return;
        }
        // test the classifier
        log.info("testing " + config.getClassifierNameInResults());
        timer.resetAndStart();
        memoryWatcher.resetAndStart();
        final ClassifierResults testResults = new ClassifierResults();
        // todo change to ts
        //            if(classifier instanceof TSClassifier) {
        //                ClassifierTools.addPredictions((TSClassifier) classifier, split.getTestDataTS(), testResults, new Random(seed));
        //            } else {
        ClassifierTools.addPredictions(classifier, split.getTestDataArff(), testResults, new Random(config.getSeed()));
        //            }
        timer.stop();
        memoryWatcher.stop();
        log.info("test time: " + timer.toDuration());
        log.info("test mem: " + memoryWatcher.getMaxMemoryUsage());
        ResultUtils.setInfo(testResults, classifier, split.getTrainDataArff()); // todo ts version
        setResultInfo(testResults);
        setTrainTime(testResults, timer);
        setMemory(testResults, memoryWatcher);
        log.info("test results: ");
        log.info(testResults.writeSummaryResultsToString());
        writeResults("test", testResults, config.getTestFilePath());
        if(trainTimeLimit == null) {
            log.info("experiment complete");
        } else {
            log.info("train time contract " + trainTimeLimit + " experiment complete");
        }
    }
    
    private void setupCheckpointing() throws FileUtils.FileLock.LockException, IOException {
        if(config.isCheckpoint()) {
            if(classifier instanceof Checkpointable) {
                // the copy over the most suitable checkpoint from another run if exists
                copyMostRecentCheckpoint();
                log.info("setting checkpoint path for " + config.getClassifierName() + " to " + config.getCheckpointDirPath());
                ((Checkpointable) classifier).setCheckpointPath(config.getCheckpointDirPath());
            } else {
                log.info(config.getClassifierName() + " cannot produce checkpoints");
            }
        }
    }
    
    private void optionalRemoveCheckpoint() {
        // if removing checkpoints then we can remove the most recent checkpoint as it has been copied over to the next contract
        if(config.isRemoveCheckpoint()) {
            log.info("removing previous checkpoint: " + config.getCheckpointDirPath());
            try {
                Files.delete(Paths.get(config.getCheckpointDirPath()));
            } catch(IOException e) {
                System.err.println("failed to remove checkpoint at " + config.getCheckpointDirPath());
                System.err.println(e);
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
}
