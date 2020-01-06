package machine_learning.classifiers.tuned.incremental;

import com.google.common.primitives.Doubles;
import evaluation.storage.ClassifierResults;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.MemoryWatchable;
import tsml.classifiers.ProgressiveBuildClassifier;
import tsml.classifiers.TrainTimeContractable;
import utilities.ArrayUtilities;
import utilities.MemoryWatcher;
import utilities.StopWatch;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import static utilities.collections.Utils.replace;

public class IncTunedClassifier extends EnhancedAbstractClassifier implements ProgressiveBuildClassifier,
                                                                              TrainTimeContractable, MemoryWatchable {

    private BenchmarkIterator benchmarkIterator = new BenchmarkIterator() {
        @Override
        public boolean hasNext() {
            return false;
        }
    };
    protected Set<Benchmark> collectedBenchmarks = new HashSet<>();
    protected BenchmarkCollector benchmarkCollector = new BestBenchmarkCollector(benchmark -> benchmark.getResults().getAcc());
    protected BenchmarkEnsembler benchmarkEnsembler = BenchmarkEnsembler.byScore(benchmark -> benchmark.getResults().getAcc());
    protected List<Double> ensembleWeights = new ArrayList<>();
    protected Consumer<Instances> onTrainDataAvailable = instances -> {

    };
    protected MemoryWatcher memoryWatcher = new MemoryWatcher();
    protected Instances trainData;

    @Override public void buildClassifier(final Instances data) throws Exception {
        super.buildClassifier(data);
        ProgressiveBuildClassifier.super.buildClassifier(data);
    }

    @Override public void startBuild(final Instances data) throws Exception {
        memoryWatcher.resume();
        trainData = data;
        onTrainDataAvailable.accept(data); // todo perhaps this should be obtained via a get? Not necessarily always
        // required
    }

    @Override
    public boolean hasNextBuildTick() throws Exception {
        return benchmarkIterator.hasNext();
    }

    @Override
    public void nextBuildTick() throws Exception {
        Set<Benchmark> nextBenchmarks = benchmarkIterator.next();
        benchmarkCollector.addAll(nextBenchmarks);
    }

    @Override
    public void finishBuild() throws Exception {
        collectedBenchmarks = benchmarkCollector.getCollectedBenchmarks();
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

    protected long trainTimeLimitNanos = -1;
    protected StopWatch trainTimer = new StopWatch();

    @Override public void setTrainTimeLimitNanos(final long nanos) {
        trainTimeLimitNanos = nanos;
    }

    @Override public long predictNextTrainTimeNanos() {
        return -1;
    }

    @Override public boolean isDone() {
        return benchmarkIterator.hasNext();
    }

    @Override public long getTrainTimeNanos() {
        return trainTimer.getTimeNanos();
    }

    @Override public long getMaxMemoryUsageInBytes() {
        return memoryWatcher.getMaxMemoryUsageInBytes();
    }

    @Override public long getMeanMemoryUsageInBytes() {
        return memoryWatcher.getMeanMemoryUsageInBytes();
    }

    @Override public long getVarianceMemoryUsageInBytes() {
        return memoryWatcher.getVarianceMemoryUsageInBytes();
    }

    @Override public long getGarbageCollectionTimeInMillis() {
        return memoryWatcher.getGarbageCollectionTimeInMillis();
    }

    @Override public long getTrainTimeLimitNanos() {
        return trainTimeLimitNanos;
    }

    // todo param handler + put lambdas / anon classes in full class for str representation in get/setoptions
}
