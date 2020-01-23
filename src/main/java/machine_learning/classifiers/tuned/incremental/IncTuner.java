package machine_learning.classifiers.tuned.incremental;

import com.google.common.primitives.Doubles;
import tsml.classifiers.*;
import utilities.*;
import utilities.params.ParamHandler;
import utilities.params.ParamSet;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.io.Serializable;
import java.util.*;
import java.util.concurrent.TimeUnit;

import static utilities.collections.Utils.replace;

public class IncTuner extends EnhancedAbstractClassifier implements TrainTimeContractable, MemoryWatchable,
                                                                    Checkpointable {

    public IncTuner() {
        super(true);
    }

    public interface InitFunction extends Serializable, ParamHandler {
        void init(Instances trainData);
    }

    private BenchmarkExplorer benchmarkExplorer = new BenchmarkExplorer() {
        @Override
        public Set<Benchmark> findFinalBenchmarks() {
            throw new UnsupportedOperationException();
        }

        @Override
        public boolean hasNext() {
            throw new UnsupportedOperationException();
        }

        @Override public Set<Benchmark> next() {
            throw new UnsupportedOperationException();
        }
    };
    protected transient Set<Benchmark> benchmarks = new HashSet<>();
    protected transient Set<Benchmark> benchmarksToCheckpoint = new HashSet<>();
    protected BenchmarkEnsembler benchmarkEnsembler = BenchmarkEnsembler.byScore(benchmark -> benchmark.getResults().getAcc());
    protected List<Double> ensembleWeights = new ArrayList<>();
    protected InitFunction initFunction = instances -> {};
    protected MemoryWatcher memoryWatcher = new MemoryWatcher();
    protected StopWatch trainTimer = new StopWatch();
    protected StopWatch trainEstimateTimer = new StopWatch();
    protected long trainTimeLimitNanos = -1;
    private static final String checkpointFileName = "checkpoint.ser";
    private static final String tempCheckpointFileName = checkpointFileName + ".tmp";
    private String checkpointDirPath;
    private String resultsDirPath; // for if you want to store the benchmarks in the results dir because they're also
    // legitimate classifiers in their own right (e.g. EE would do this to store DTWCV benchmarks as DTW_1, DTW_2,
    // DTW_3, etc. Then we can look at the results using current tools rather than creating something to handle the
    // output from tuning.
    private boolean independentBenchmarks = false;
    private long lastCheckpointTimeStamp = 0;
    private long minCheckpointIntervalNanos = TimeUnit.NANOSECONDS.convert(1, TimeUnit.HOURS);
    private final static String DEFAULT_RESULTS_DIR = "benchmarks";
    private boolean ignorePreviousCheckpoints = false;
    private boolean checkpointAfterEveryIteration = false;
    public static final String BENCHMARK_ITERATOR_FLAG = "b";
    public static final String INIT_FUNCTION_FLAG = "i";

    public StopWatch getTrainEstimateTimer() {
        return trainEstimateTimer;
    }

    @Override public ParamSet getParams() {
        return TrainTimeContractable.super.getParams()
                                    .add(BENCHMARK_ITERATOR_FLAG, benchmarkExplorer)
                                    .add(INIT_FUNCTION_FLAG, initFunction);
    }

    @Override public void setParams(final ParamSet params) {
        TrainTimeContractable.super.setParams(params);
        ParamHandler.setParam(params, BENCHMARK_ITERATOR_FLAG, this::setBenchmarkExplorer, BenchmarkExplorer.class);
        ParamHandler.setParam(params, INIT_FUNCTION_FLAG, this::setInitFunction, InitFunction.class);
    }

    @Override public boolean isIgnorePreviousCheckpoints() {
        return ignorePreviousCheckpoints;
    }

    @Override public void setIgnorePreviousCheckpoints(final boolean ignorePreviousCheckpoints) {
        this.ignorePreviousCheckpoints = ignorePreviousCheckpoints;
    }

    @Override public void setMinCheckpointIntervalNanos(final long nanos) {
        minCheckpointIntervalNanos = nanos;
    }

    @Override public long getMinCheckpointIntervalNanos() {
        return minCheckpointIntervalNanos;
    }

    @Override
    public boolean setSavePath(String path) {
        if(path == null) {
            return false;
        }
        checkpointDirPath = StrUtils.asDirPath(path);
        return true;
    }

    public boolean isSavingBenchmarksAsResults() {
        return resultsDirPath != null;
    }

    @Override
    public String getSavePath() {
        return checkpointDirPath;
    }

    private String getBenchmarksPath(String name) {
        String path;
        if(isSavingBenchmarksAsResults()) {
            path = resultsDirPath + name; // todo probs don't need this
        } else {
            path = StrUtils.asDirPath(checkpointDirPath, DEFAULT_RESULTS_DIR,  name);
        }
        return path;
    }

    private boolean withinCheckpointInterval() {
        return lastCheckpointTimeStamp + minCheckpointIntervalNanos < System.nanoTime();
    }

    public void checkpoint() throws Exception {
        trainTimer.suspend();
        trainEstimateTimer.suspend();
        memoryWatcher.suspend();
        if(isCheckpointing() && (built || withinCheckpointInterval() || isCheckpointAfterEveryIteration())) {
            String path = checkpointDirPath + tempCheckpointFileName;
            FileUtils.FileLocker locker = new FileUtils.FileLocker(new File(path));
            if(locker.isUnlocked()) {
                String msg = "failed to lock file: " + path;
                logger.severe(msg);
                throw new IllegalStateException(msg);
            }
            final String finalPath1 = path;
            logger.info(() -> "saving checkpoint to: " + finalPath1);
            saveToFile(path);
            boolean success = new File(path).renameTo(new File(checkpointDirPath + checkpointFileName));
            if(!success) {
                throw new IllegalStateException("could not rename checkpoint file");
            }
            locker.unlock();
            for(Benchmark benchmark : benchmarksToCheckpoint) {
                int id = benchmark.hashCode();
                Classifier classifier = benchmark.getClassifier();
                String name;
                if(classifier instanceof EnhancedAbstractClassifier) {
                    name = ((EnhancedAbstractClassifier) classifier).getClassifierName();
                } else {
                    name = classifier.getClass().getSimpleName();
                }
                name += "_" + id;
                path = getBenchmarksPath(name) + ".csv";
                final String finalPath = path;
                locker = new FileUtils.FileLocker(new File(path));
                if(locker.isUnlocked()) {
                    if(independentBenchmarks) {
                        logger.severe(() -> "failed to lock file: " + finalPath + " but that's ok as we're running " +
                                        "independently");
                        continue;
                    } else {
                        String msg = "failed to lock file: " + path + " another task must be using this file";
                        logger.severe(msg);
                        throw new IllegalStateException(msg);
                    }
                }
                String tmpPath = path + ".tmp";
                logger.info(() -> "saving benchmark checkpoint to: " + finalPath);
                benchmark.getResults().writeFullResultsToFile(tmpPath);
                success = new File(tmpPath).renameTo(new File(path));
                if(!success) {
                    throw new IllegalStateException("could not rename benchmark checkpoint file");
                }
                locker.unlock();
            }
            benchmarksToCheckpoint.clear();
            lastCheckpointTimeStamp = System.nanoTime();
        }
        memoryWatcher.unsuspend();
        trainTimer.unsuspend();
        trainEstimateTimer.unsuspend();
    }

    protected void loadFromCheckpoint() throws Exception {
        trainTimer.suspend();
        trainEstimateTimer.suspend();
        memoryWatcher.suspend();
        if(!isIgnorePreviousCheckpoints() && isCheckpointing() && isRebuild()) {
            String path = checkpointDirPath + checkpointFileName;
            logger.info(() -> "loading from checkpoint: " + path);
            loadFromFile(path);
        }
        memoryWatcher.unsuspend();
        trainTimer.unsuspend();
        trainEstimateTimer.unsuspend();
    }

    public StopWatch getTrainTimer() {
        return trainTimer;
    }

    @Override public void buildClassifier(final Instances trainData) throws Exception {
        preBuild(trainData);
        build(trainData);
        postBuild(trainData);
    }

    protected void preBuild(final Instances trainData) throws Exception {
        trainEstimateTimer.checkDisabled();
        trainTimer.enableAnyway();
        memoryWatcher.enableAnyway();
        if(rebuild) {
            super.buildClassifier(trainData);
            initFunction.init(trainData);
            rebuild = false;
            built = false;
        }
        trainTimer.disable();
        trainEstimateTimer.enable();
    }

    protected boolean hasNextBuildTick() throws Exception {
        return hasRemainingTraining();
    }

    protected void nextBuildTick() throws Exception {
        final Set<Benchmark> benchmarks = benchmarkExplorer.next();
        // either this or the benchmark iterator - should be the latter in most cases
        logger.info(() -> benchmarks.size() + " benchmark(s) changed");
        if(isCheckpointing()) {
            replace(benchmarksToCheckpoint, benchmarks);
        }
    }

    protected void build(Instances trainData) throws Exception {
        trainTimer.checkDisabled();
        trainEstimateTimer.checkEnabled();
        memoryWatcher.checkEnabled();
        while(hasNextBuildTick()) {
            nextBuildTick();
            checkpoint();
        }
    }

    protected void postBuild(Instances trainData) throws Exception {
        if(!independentBenchmarks) {
            trainTimer.checkDisabled();
            trainEstimateTimer.checkEnabled();
            memoryWatcher.checkEnabled();
            benchmarks = benchmarkExplorer.findFinalBenchmarks();
            if(benchmarks.isEmpty()) {
                logger.info(() -> "no benchmarks collected");
                ensembleWeights = new ArrayList<>();
                throw new UnsupportedOperationException("todo implement random guess here?");
            } else if(benchmarks.size() == 1) {
                logger.info(() -> "single benchmarks collected");
                ensembleWeights = new ArrayList<>(Collections.singletonList(1d));
                trainResults = benchmarks.iterator().next().getResults();
            } else {
                logger.info(() -> benchmarks.size() + " benchmarks collected");
                ensembleWeights = benchmarkEnsembler.weightVotes(benchmarks);
                throw new UnsupportedOperationException("todo apply ensemble weights to train results");
            }
            memoryWatcher.disable();
            trainEstimateTimer.disable();
            trainResults.setMemory(getMaxMemoryUsageInBytes());
            trainResults.setBuildTime(trainTimer.getTimeNanos());
            trainResults.setBuildPlusEstimateTime(getTrainTimeNanos());
            trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
            trainResults.setFoldID(seed);
            trainResults.setDetails(this, trainData);
        }
        memoryWatcher.disableAnyway();
        trainEstimateTimer.disableAnyway();
        trainTimer.disableAnyway();
        built = true;
        checkpoint();
    }

    public BenchmarkIterator getBenchmarkExplorer() {
        return benchmarkExplorer;
    }

    public void setBenchmarkExplorer(BenchmarkExplorer benchmarkExplorer) {
        this.benchmarkExplorer = benchmarkExplorer;
    }

    public Set<Benchmark> getBenchmarks() {
        return benchmarks;
    }

    public BenchmarkEnsembler getBenchmarkEnsembler() {
        return benchmarkEnsembler;
    }

    public void setBenchmarkEnsembler(BenchmarkEnsembler benchmarkEnsembler) {
        this.benchmarkEnsembler = benchmarkEnsembler;
    }

    public List<Double> getEnsembleWeights() {
        return ensembleWeights;
    }

    @Override
    public double[] distributionForInstance(Instance testCase) throws Exception {
        Iterator<Benchmark> benchmarkIterator = benchmarks.iterator();
        if(benchmarks.size() == 1) {
            return benchmarkIterator.next().getClassifier().distributionForInstance(testCase);
        }
        double[] distribution = new double[numClasses];
        for(int i = 0; i < benchmarks.size(); i++) {
            if(!benchmarkIterator.hasNext()) {
                throw new IllegalStateException("iterator incorrect");
            }
            Benchmark benchmark = benchmarkIterator.next();
            double[] constituentDistribution = benchmark.getClassifier().distributionForInstance(testCase);
            ArrayUtilities.multiplyInPlace(constituentDistribution, ensembleWeights.get(i));
            ArrayUtilities.addInPlace(distribution, constituentDistribution);
        }
        ArrayUtilities.normaliseInPlace(distribution);
        return distribution;
    }

    @Override
    public double classifyInstance(Instance testCase) throws Exception {
        return ArrayUtilities.bestIndex(Doubles.asList(distributionForInstance(testCase)), rand);
    }

    public InitFunction getInitFunction() {
        return initFunction;
    }

    public void setInitFunction(final InitFunction initFunction) {
        this.initFunction = initFunction;
    }

    @Override public void setTrainTimeLimitNanos(final long nanos) {
        trainTimeLimitNanos = nanos;
    }

    @Override public long predictNextTrainTimeNanos() {
        return benchmarkExplorer.predictNextTimeNanos();
    }

    @Override public boolean isDone() {
        return !benchmarkExplorer.hasNext();
    }

    @Override public long getTrainTimeNanos() {
        return trainTimer.getTimeNanos() + trainEstimateTimer.getTimeNanos();
    }

    @Override public MemoryWatcher getMemoryWatcher() {
        return memoryWatcher;
    }

    @Override public long getTrainTimeLimitNanos() {
        return trainTimeLimitNanos;
    }

    public String getResultsDirPath() {
        return resultsDirPath;
    }

    public void setResultsDirPath(final String resultsDirPath) {
        this.resultsDirPath = resultsDirPath;
    }

    public boolean isIndependentBenchmarks() {
        return independentBenchmarks;
    }

    public void setIndependentBenchmarks(final boolean independentBenchmarks) {
        this.independentBenchmarks = independentBenchmarks;
    }

    public boolean isCheckpointAfterEveryIteration() {
        return checkpointAfterEveryIteration;
    }

    public void setCheckpointAfterEveryIteration(final boolean checkpointAfterEveryIteration) {
        this.checkpointAfterEveryIteration = checkpointAfterEveryIteration;
    }

}
