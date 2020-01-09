package machine_learning.classifiers.tuned.incremental;

import com.google.common.primitives.Doubles;
import evaluation.storage.ClassifierResults;
import tsml.classifiers.*;
import utilities.ArrayUtilities;
import utilities.MemoryWatcher;
import utilities.StopWatch;
import utilities.StrUtils;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import static utilities.collections.Utils.replace;

public class IncTuner extends EnhancedAbstractClassifier implements ProgressiveBuildClassifier,
                                                                    TrainTimeContractable, MemoryWatchable,
                                                                    Checkpointable {

    public IncTuner() {
        super(true);
    }

    private BenchmarkIterator benchmarkIterator = new BenchmarkIterator() {
        @Override
        public boolean hasNext() {
            throw new UnsupportedOperationException();
        }

        @Override public Set<Benchmark> next() {
            throw new UnsupportedOperationException();
        }
    };
    protected transient Set<Benchmark> collectedBenchmarks = new HashSet<>();
    protected BenchmarkCollector benchmarkCollector = new BestBenchmarkCollector(benchmark -> benchmark.getResults().getAcc());
    protected BenchmarkEnsembler benchmarkEnsembler = BenchmarkEnsembler.byScore(benchmark -> benchmark.getResults().getAcc());
    protected List<Double> ensembleWeights = new ArrayList<>();
    protected Consumer<Instances> onTrainDataAvailable = instances -> {

    };
    protected MemoryWatcher memoryWatcher = new MemoryWatcher();
    protected StopWatch trainTimer = new StopWatch();
    protected Instances trainData;
    protected long trainTimeLimitNanos = -1;
    private static final String checkpointFileName = "checkpoint.ser";
    private static final String tempCheckpointFileName = checkpointFileName + ".tmp";
    private String checkpointDirPath;
    private String resultsDirPath; // for if you want to store the benchmarks in the results dir because they're also
    // legitimate classifiers in their own right (e.g. EE would do this to store DTWCV benchmarks as DTW_1, DTW_2,
    // DTW_3, etc. Then we can look at the results using current tools rather than creating something to handle the
    // output from tuning.
    private long lastCheckpointTimeStamp = 0;
    private long minCheckpointIntervalNanos = TimeUnit.NANOSECONDS.convert(1, TimeUnit.HOURS);
    private static String DEFAULT_RESULTS_DIR = "benchmarks";

    @Override
    public boolean setSavePath(final String path) {
        checkpointDirPath = StrUtils.asDirPath(path);
        return true;
    }

    @Override
    public String getSavePath() {
        return checkpointDirPath;
    }

    public void checkpoint(boolean force) throws
                                          Exception {
        trainTimer.pause();
        memoryWatcher.pause();
        if(isCheckpointing() && (force || lastCheckpointTimeStamp + minCheckpointIntervalNanos < System.nanoTime())) {
            logger.log("checkpointing");
            saveToFile(checkpointDirPath + tempCheckpointFileName);
            boolean success = new File(checkpointDirPath + tempCheckpointFileName).renameTo(new File(checkpointDirPath + checkpointDirPath));
            if(!success) {
                throw new IllegalStateException("could not rename checkpoint file");
            }
            
            lastCheckpointTimeStamp = System.nanoTime();
        }
        memoryWatcher.resume();
        trainTimer.resume();
    }

    public StopWatch getTrainTimer() {
        return trainTimer;
    }

    @Override public void buildClassifier(final Instances data) throws Exception {

        ProgressiveBuildClassifier.super.buildClassifier(data);
    }

    @Override public void startBuild(final Instances data) throws Exception {
        trainTimer.resume();
        memoryWatcher.resume();
        if(rebuild) {
            super.buildClassifier(data);
            onTrainDataAvailable.accept(data);
            rebuild = false;
        }
        trainData = data;
        memoryWatcher.pause();
        trainTimer.pause();
    }

    @Override
    public boolean hasNextBuildTick() throws Exception {
        trainTimer.resume();
        memoryWatcher.resume();
        boolean result = hasRemainingTraining();
        trainTimer.pause();
        memoryWatcher.pause();
        return result;
    }

    @Override
    public void nextBuildTick() throws Exception {
        trainTimer.resume();
        memoryWatcher.resume();
        Set<Benchmark> nextBenchmarks = benchmarkIterator.next();
        logger.log("benchmark batch produced:");
        for(Benchmark benchmark : nextBenchmarks) {
            logger.log(benchmark);
        }
        replace(collectedBenchmarks, nextBenchmarks);
        trainTimer.pause();
        memoryWatcher.pause();
    }

    @Override
    public void finishBuild() throws Exception {
        trainTimer.resume();
        memoryWatcher.resume();
        benchmarkCollector.addAll(collectedBenchmarks); // add all the current benchmarks to the filter
        collectedBenchmarks = benchmarkCollector.getCollectedBenchmarks(); // reassign the filtered benchmarks
        if(collectedBenchmarks.isEmpty()) {
            if(debug) {
                System.out.println("no benchmarks produced");
            }
            ensembleWeights = new ArrayList<>();
            trainResults = new ClassifierResults(); // todo random guess
        } else if(collectedBenchmarks.size() == 1) {
            ensembleWeights = new ArrayList<>(Collections.singletonList(1d));
            trainResults = collectedBenchmarks.iterator().next().getResults();
        } else {
            ensembleWeights = benchmarkEnsembler.weightVotes(collectedBenchmarks);
        }
        memoryWatcher.pause();
        trainTimer.pause();
        trainResults.setMemory(getMaxMemoryUsageInBytes()); // todo other fields
        trainResults.setBuildTime(getTrainTimeNanos()); // todo break down to estimate time also
        trainResults.setTimeUnit(TimeUnit.NANOSECONDS);
        trainResults.setFoldID(seed); // todo set other details
        trainResults.setDetails(this, trainData);
        trainData = null;
    }

    public BenchmarkIterator getBenchmarkIterator() {
        return benchmarkIterator;
    }

    public void setBenchmarkIterator(BenchmarkIterator benchmarkIterator) {
        this.benchmarkIterator = benchmarkIterator;
    }

    public Set<Benchmark> getCollectedBenchmarks() {
        return collectedBenchmarks;
    }

    public BenchmarkCollector getBenchmarkCollector() {
        return benchmarkCollector;
    }

    public void setBenchmarkCollector(BenchmarkCollector benchmarkCollector) {
        this.benchmarkCollector = benchmarkCollector;
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
        Iterator<Benchmark> benchmarkIterator = collectedBenchmarks.iterator();
        if(collectedBenchmarks.size() == 1) {
            return benchmarkIterator.next().getClassifier().distributionForInstance(testCase);
        }
        double[] distribution = new double[numClasses];
        for(int i = 0; i < collectedBenchmarks.size(); i++) {
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

    public Consumer<Instances> getOnTrainDataAvailable() {
        return onTrainDataAvailable;
    }

    public void setOnTrainDataAvailable(final Consumer<Instances> onTrainDataAvailable) {
        this.onTrainDataAvailable = onTrainDataAvailable;
    }

    @Override public void setTrainTimeLimitNanos(final long nanos) {
        trainTimeLimitNanos = nanos;
    }

    @Override public long predictNextTrainTimeNanos() {
        return benchmarkIterator.predictNextTimeNanos();
    }

    @Override public boolean isDone() {
        return !benchmarkIterator.hasNext();
    }

    @Override public long getTrainTimeNanos() {
        return trainTimer.getTimeNanos();
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

    // todo param handler + put lambdas / anon classes in full class for str representation in get/setoptions
}
