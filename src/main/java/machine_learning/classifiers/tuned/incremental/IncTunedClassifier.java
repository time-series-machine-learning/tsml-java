package machine_learning.classifiers.tuned.incremental;

import com.google.common.primitives.Doubles;
import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.ProgressiveBuildClassifier;
import utilities.ArrayUtilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.function.Consumer;

import static utilities.collections.Utils.replace;

public class IncTunedClassifier extends EnhancedAbstractClassifier implements ProgressiveBuildClassifier {

    private BenchmarkIterator benchmarkIterator = new BenchmarkIterator() {
        @Override
        public boolean hasNext() {
            return false;
        }
    };
    private Set<Benchmark> collectedBenchmarks = new HashSet<>();
    private BenchmarkCollector benchmarkCollector = new BestBenchmarkCollector(benchmark -> benchmark.getResults().getAcc());
    private BenchmarkEnsembler benchmarkEnsembler = BenchmarkEnsembler.byScore(benchmark -> benchmark.getResults().getAcc());
    private List<Double> ensembleWeights = null;
    private Consumer<Instances> onTrainDataAvailable = instances -> {

    };

    @Override public void startBuild(final Instances data) throws Exception {
        onTrainDataAvailable.accept(data);
    }

    @Override
    public boolean hasNextBuildTick() throws Exception {
        return benchmarkIterator.hasNext();
    }

    @Override
    public void nextBuildTick() throws Exception {
        Set<Benchmark> nextBenchmarks = benchmarkIterator.next();
        replace(collectedBenchmarks, nextBenchmarks);
    }

    @Override
    public void finishBuild() throws Exception {
        collectedBenchmarks = benchmarkCollector.getCollectedBenchmarks();
        if(collectedBenchmarks.isEmpty()) {
            throw new IllegalStateException("no benchmarks");
        } else if(collectedBenchmarks.size() == 1) {
            ensembleWeights = null;
        } else {
            ensembleWeights = benchmarkEnsembler.weightVotes(collectedBenchmarks);
        }
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

    // todo param handler + put lambdas / anon classes in full class for str representation in get/setoptions
}
