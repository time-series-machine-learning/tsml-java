package tsml.classifiers.distance_based.knn.configs;

import evaluation.storage.ClassifierResults;
import machine_learning.classifiers.tuned.incremental.*;
import org.apache.commons.collections4.Transformer;
import org.apache.commons.collections4.iterators.TransformIterator;
import tsml.classifiers.distance_based.knn.KnnLoocv;
import utilities.*;
import utilities.collections.PrunedMultimap;
import utilities.collections.box.Box;
import utilities.iteration.RandomIterator;
import utilities.params.ParamSet;
import utilities.params.ParamSpace;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.io.Serializable;
import java.util.*;
import java.util.function.Function;
import java.util.function.Supplier;

public class IncKnnTunerBuilder implements IncTuner.InitFunction {


    private IncTuner incTunedClassifier = new IncTuner();
    private ParamSpace paramSpace;
    private SourcedBenchmarkExplorer sourcedBenchmarkExplorer = new SourcedBenchmarkExplorer();
    // setup the param space to source classifiers
    private Iterator<ParamSet> paramSetIterator;
    // setup building classifier from param set
    private int maxParamSpaceSize = -1; // max number of params
    private int maxNeighbourhoodSize = -1; // max number of neighbours
    private Box<Integer> neighbourCount; // current number of neighbours
    private Box<Integer> paramCount; // current number of params
    private long longestParamTimeNanos; // track maximum time taken for a param to run
    private long longestNeighbourBatchTimeNanos; // track max time taken for an addition of
    // neighbours
    private Function<Instances, ParamSpace> paramSpaceFunction;
    private Iterator<Set<Benchmark>> paramSourceIterator;
    private BenchmarkIterator benchmarkSourceIterator;
    private BenchmarkImprover benchmarkImprover;
    private Optimiser optimiser;
    private Supplier<KnnLoocv> knnSupplier;
    private double maxNeighbourhoodSizePercentage = -1;
    private double maxParamSpaceSizePercentage = -1;
    private int fullParamSpaceSize = -1;
    private int fullNeighbourhoodSize = -1;
    private Set<Benchmark> improveableBenchmarks;
    private Set<Benchmark> unimprovableBenchmarks;
    private boolean limitedVersion = false;

    public Scorer getScorer() {
        return scorer;
    }

    public void setScorer(Scorer scorer) {
        this.scorer = scorer;
    }

    public boolean isLimitedVersion() {
        return limitedVersion;
    }

    public void setLimitedVersion(boolean limitedVersion) {
        this.limitedVersion = limitedVersion;
    }

    public interface Scorer extends Serializable {
        double score(ClassifierResults results);
    }

    private Scorer scorer = results -> {
        double acc = results.getAcc();
        results.cleanPredictionInfo();
        return acc;
    };

    // todo param handling

    public IncTuner build() {
        incTunedClassifier.setInitFunction(this);
        return incTunedClassifier;
    }

    private int findLimit(int size, int rawLimit, double percentageLimit) {
        if(size == 0) {
            throw new IllegalArgumentException();
        }
        int result = size;
        if(rawLimit >= 0) {
            result = rawLimit;
        }
        if(NumUtils.isPercentage(percentageLimit)) {
            result = (int) (size * percentageLimit);
        }
        if(result == 0) {
            result = 1;
        }
        return result;
    }

    private boolean hasLimits() {
        return hasLimitedNeighbourhoodSize() || hasLimitedParamSpaceSize();
    }

    private boolean hasLimitedParamSpaceSize() {
        return maxParamSpaceSize < 0 || !NumUtils.isPercentage(maxParamSpaceSizePercentage);
    }

    private boolean hasLimitedNeighbourhoodSize() {
        return maxNeighbourhoodSize < 0 || !NumUtils.isPercentage(maxNeighbourhoodSizePercentage);
    }

    private boolean withinParamSpaceSizeLimit() {
        return paramCount.get() < maxParamSpaceSize;
    }

    private boolean withinNeighbourhoodSizeLimit() {
        return neighbourCount.get() < maxNeighbourhoodSize;
    }

    public double getMaxParamSpaceSizePercentage() {
        return maxParamSpaceSizePercentage;
    }

    public IncKnnTunerBuilder setMaxParamSpaceSizePercentage(final double maxParamSpaceSizePercentage) {
        this.maxParamSpaceSizePercentage = maxParamSpaceSizePercentage;
        return this;
    }

    private void delegateResourceMonitoring(KnnLoocv knn) {
        knn.getTrainTimer().addListener(incTunedClassifier.getTrainEstimateTimer()); // because tuning is estimating
        knn.getTrainEstimateTimer().addListener(incTunedClassifier.getTrainEstimateTimer());
        knn.getMemoryWatcher().addListener(incTunedClassifier.getMemoryWatcher());
    }

    private void undelegateResourceMonitoring(KnnLoocv knn) {
        knn.getMemoryWatcher().removeListener(incTunedClassifier.getMemoryWatcher());
        knn.getTrainTimer().removeListener(incTunedClassifier.getTrainEstimateTimer());
        knn.getTrainEstimateTimer().removeListener(incTunedClassifier.getTrainEstimateTimer());
        incTunedClassifier.getTrainTimer().disableAnyway();
        incTunedClassifier.getTrainEstimateTimer().enableAnyway();
        incTunedClassifier.getMemoryWatcher().enableAnyway();
    }

    @Override
    public void init(Instances trainData) {
        neighbourCount = new Box<>(1); // must start at 1 otherwise the loocv produces no train estimate
        paramCount = new Box<>(0);
        longestParamTimeNanos = 0;
        longestNeighbourBatchTimeNanos = 0;
        improveableBenchmarks = new HashSet<>();
        unimprovableBenchmarks = new HashSet<>();
        final int seed = incTunedClassifier.getSeed();
        paramSpace = paramSpaceFunction.apply(trainData);
        paramSetIterator = new RandomIterator<>(this.paramSpace, seed);
        fullParamSpaceSize = this.paramSpace.size();
        fullNeighbourhoodSize = trainData.size();
        maxNeighbourhoodSize = findLimit(fullNeighbourhoodSize, maxNeighbourhoodSize, maxNeighbourhoodSizePercentage);
        maxParamSpaceSize = findLimit(fullParamSpaceSize, maxParamSpaceSize, maxParamSpaceSizePercentage);
        if(incTunedClassifier.hasTrainTimeLimit() && hasLimits()) {
            throw new IllegalStateException("cannot train under a contract with limits set");
        }
        // transform classifiers into benchmarks
        paramSourceIterator =
            new TransformIterator<>(paramSetIterator,
                                    new Transformer<ParamSet, Set<Benchmark>>() {
                                        private int id = 0;

                                        @Override public Set<Benchmark> transform(final ParamSet paramSet) {
                                            try {
                                                final StopWatch timer = new StopWatch();
                                                timer.enable();
                                                paramCount.set(paramCount.get() + 1);
                                                final KnnLoocv knn = knnSupplier.get();
                                                knn.setNeighbourLimit(neighbourCount.get());
                                                knn.setParams(paramSet);
                                                knn.setEstimateOwnPerformance(true);
                                                timer.disable();
                                                delegateResourceMonitoring(knn);
                                                knn.buildClassifier(trainData);
                                                undelegateResourceMonitoring(knn);
                                                timer.enable();
                                                final Benchmark benchmark = new Benchmark(knn, knn.getTrainResults(), id++);
                                                benchmark.score(scorer::score);
                                                final HashSet<Benchmark> benchmarks = new HashSet<>(
                                                    Collections.singletonList(benchmark));
                                                timer.disable();
                                                longestParamTimeNanos = Math.max(longestParamTimeNanos, timer.getTimeNanos());
                                                incTunedClassifier.getLogger().info(() -> "sourced " + paramSet.toString());
                                                return benchmarks;
                                            } catch(Exception e) {
                                                throw new IllegalStateException(e);
                                            }
                                        }
                                    }
            );
        benchmarkSourceIterator = new BenchmarkIterator() {
            @Override public long predictNextTimeNanos() {
                return longestParamTimeNanos;
            }

            @Override public Set<Benchmark> next() {
                return paramSourceIterator.next();
            }

            @Override public boolean hasNext() {
                return paramSourceIterator.hasNext() && withinParamSpaceSizeLimit();
            }
        };
        // setup an iterator to improve benchmarks
        benchmarkImprover = new BenchmarkImprover() {

            @Override public long predictNextTimeNanos() {
                return longestNeighbourBatchTimeNanos;
            }

            public Set<Benchmark> getImproveableBenchmarks() {
                return improveableBenchmarks;
            }

            public Set<Benchmark> getUnimprovableBenchmarks() {
                return unimprovableBenchmarks;
            }

            public Set<Benchmark> getAllBenchmarks() {
                final HashSet<Benchmark> benchmarks = new HashSet<>();
                benchmarks.addAll(unimprovableBenchmarks);
                benchmarks.addAll(improveableBenchmarks);
                return benchmarks;
            }

            @Override
            public void add(Benchmark benchmark) {
                if(isImproveable(benchmark)) {
                    improveableBenchmarks.add(benchmark);
                } else {
                    unimprovableBenchmarks.add(benchmark);
                }
            }

            @Override
            public Set<Benchmark> findFinalBenchmarks() {
                incTunedClassifier.getTrainTimer().checkDisabled();
                final StopWatch trainEstimateTimer = incTunedClassifier.getTrainEstimateTimer();
                final MemoryWatcher memoryWatcher = incTunedClassifier.getMemoryWatcher();
                final PrunedMultimap<Double, Benchmark> map = PrunedMultimap.desc(ArrayList::new);
                map.setSoftLimit(1);
                for(final Benchmark benchmark : getAllBenchmarks()) {
                    map.put(benchmark.getScore(), benchmark);
                }
                final Collection<Benchmark> benchmarks = map.values();
                final List<Benchmark> selectedBenchmarks = Utilities.randPickN(benchmarks, 1, incTunedClassifier.getRand());
                // train the selected classifier fully, i.e. all neighbours
                if(incTunedClassifier.isDebug() && selectedBenchmarks.size() > 1) {
                    throw new IllegalStateException("there shouldn't be more than 1");
                }
                if(limitedVersion) {
                    incTunedClassifier.getLogger().info("limited version, therefore training selected benchmark fully");
                    for(final Benchmark benchmark : selectedBenchmarks) {
                        final Classifier classifier = benchmark.getClassifier();
                        if(classifier instanceof KnnLoocv) {
                            final KnnLoocv knn = (KnnLoocv) classifier;
                            knn.setNeighbourLimit(-1);
                            knn.setRegenerateTrainEstimate(true);
                            delegateResourceMonitoring(knn);
                            try {
                                knn.buildClassifier(trainData);
                            } catch (Exception e) {
                                throw new IllegalStateException(e);
                            }
                            // todo should we use delegation for the resource monitoring or get it from benchmarks?
                            undelegateResourceMonitoring(knn);
                            benchmark.setResults(knn.getTrainResults());
                        } else {
                            throw new IllegalArgumentException("expected knn");
                        }
                    }
                }
                return new HashSet<>(selectedBenchmarks);
            }

            private boolean isImproveable(Benchmark benchmark) {
                try {
                    final KnnLoocv knn = (KnnLoocv) benchmark.getClassifier();
                    return knn.getNeighbourLimit() + 1 <= maxNeighbourhoodSize;
                } catch(Exception e) {
                    throw new IllegalStateException(e);
                }
            }

            @Override
            public Set<Benchmark> next() {
                final StopWatch timer = new StopWatch();
                timer.enable();
                final int origNeighbourCount = neighbourCount.get();
                final int nextNeighbourCount = origNeighbourCount + 1;
                incTunedClassifier.getLogger().info(() -> "neighbourhood " + origNeighbourCount + " --> " + nextNeighbourCount);
                neighbourCount.set(nextNeighbourCount);
                final Set<Benchmark> improvedBenchmarks = new HashSet<>();
                try {
                    final Iterator<Benchmark> benchmarkIterator = improveableBenchmarks.iterator();
                    while(benchmarkIterator.hasNext()) {
                        final Benchmark benchmark = benchmarkIterator.next();
                        final Classifier classifier = benchmark.getClassifier();
                        final KnnLoocv knn = (KnnLoocv) classifier;
                        if(incTunedClassifier.isDebug()) {
                            final int currentNeighbourLimit = knn.getNeighbourLimit();
                            if(nextNeighbourCount <= currentNeighbourLimit) {
                                throw new IllegalStateException("no improvement to the number of neighbours");
                            }
                        }
                        knn.setNeighbourLimit(nextNeighbourCount);
                        timer.disable();
                        delegateResourceMonitoring(knn);
                        knn.buildClassifier(trainData);
                        undelegateResourceMonitoring(knn);
                        timer.enable();
                        benchmark.setResults(knn.getTrainResults());
                        benchmark.score(scorer::score);
                        if(!isImproveable(benchmark)) {
                            benchmarkIterator.remove();
                            unimprovableBenchmarks.add(benchmark);
                        }
                        improvedBenchmarks.add(benchmark);
                    }
                } catch(Exception e) {
                    throw new IllegalStateException(e);
                }
                timer.disable();
                longestNeighbourBatchTimeNanos = Math.max(longestNeighbourBatchTimeNanos, timer.getTimeNanos());
                return improvedBenchmarks;
            }

            @Override
            public boolean hasNext() {
                return !improveableBenchmarks.isEmpty();
            }
        };
        sourcedBenchmarkExplorer.setBenchmarkImprover(benchmarkImprover);
        sourcedBenchmarkExplorer.setBenchmarkSource(benchmarkSourceIterator);
        optimiser = () -> {
            // only called when *both* improvements and source remain
            final int neighbours = neighbourCount.get();
            final int params = paramCount.get();
            if(params < maxParamSpaceSize / 10) {
                // 10% params, 0% neighbours
                return true;
            } else if(neighbours < maxNeighbourhoodSize / 10) {
                // 10% params, 10% neighbours
                return false;
            } else if(params < maxParamSpaceSize / 2) {
                // 50% params, 10% neighbours
                return true;
            } else if(neighbours < maxNeighbourhoodSize / 2) {
                // 50% params, 50% neighbours
                return false;
            } else if(params < maxParamSpaceSize) {
                // 100% params, 50% neighbours
                return true;
            }
            else {
                // by this point all params have been hit. Therefore, shouldSource should not be called at
                // all as only improvements will remain, if any.
                throw new IllegalStateException("invalid source / improvement state");
            }
        };
        sourcedBenchmarkExplorer.setOptimiser(optimiser);
        // set corresponding iterators in the incremental tuned classifier
        incTunedClassifier.setBenchmarkExplorer(sourcedBenchmarkExplorer);
        incTunedClassifier.setBenchmarkEnsembler(BenchmarkEnsembler.single());
        // todo make sure the seeds are set for everything
    }

    public IncTuner getIncTunedClassifier() {
        return incTunedClassifier;
    }

    public IncKnnTunerBuilder setIncTunedClassifier(final IncTuner incTunedClassifier) {
        this.incTunedClassifier = incTunedClassifier;
        return this;
    }

    public ParamSpace getParamSpace() {
        return paramSpace;
    }

    public IncKnnTunerBuilder setParamSpace(final ParamSpace paramSpace) {
        this.paramSpace = paramSpace;
        return this;
    }

    public SourcedBenchmarkExplorer getSourcedBenchmarkExplorer() {
        return sourcedBenchmarkExplorer;
    }

    public IncKnnTunerBuilder setSourcedBenchmarkExplorer(final SourcedBenchmarkExplorer sourcedBenchmarkExplorer) {
        this.sourcedBenchmarkExplorer = sourcedBenchmarkExplorer;
        return this;
    }

    public Iterator<ParamSet> getParamSetIterator() {
        return paramSetIterator;
    }

    public IncKnnTunerBuilder setParamSetIterator(final Iterator<ParamSet> paramSetIterator) {
        this.paramSetIterator = paramSetIterator;
        return this;
    }

    public int getMaxParamSpaceSize() {
        return maxParamSpaceSize;
    }

    public IncKnnTunerBuilder setMaxParamSpaceSize(final int maxParamSpaceSize) {
        this.maxParamSpaceSize = maxParamSpaceSize;
        return this;
    }

    public int getMaxNeighbourhoodSize() {
        return maxNeighbourhoodSize;
    }

    public IncKnnTunerBuilder setMaxNeighbourhoodSize(final int maxNeighbourhoodSize) {
        this.maxNeighbourhoodSize = maxNeighbourhoodSize;
        return this;
    }

    public Box<Integer> getNeighbourCount() {
        return neighbourCount;
    }

    public IncKnnTunerBuilder setNeighbourCount(final Box<Integer> neighbourCount) {
        this.neighbourCount = neighbourCount;
        return this;
    }

    public Box<Integer> getParamCount() {
        return paramCount;
    }

    public IncKnnTunerBuilder setParamCount(final Box<Integer> paramCount) {
        this.paramCount = paramCount;
        return this;
    }

    public long getLongestParamTimeNanos() {
        return longestParamTimeNanos;
    }

    public IncKnnTunerBuilder setLongestParamTimeNanos(final long longestParamTimeNanos) {
        this.longestParamTimeNanos = longestParamTimeNanos;
        return this;
    }

    public long getLongestNeighbourBatchTimeNanos() {
        return longestNeighbourBatchTimeNanos;
    }

    public IncKnnTunerBuilder setLongestNeighbourBatchTimeNanos(final long longestNeighbourBatchTimeNanos) {
        this.longestNeighbourBatchTimeNanos = longestNeighbourBatchTimeNanos;
        return this;
    }

    public Function<Instances, ParamSpace> getParamSpaceFunction() {
        return paramSpaceFunction;
    }

    public IncKnnTunerBuilder setParamSpaceFunction(
        final Function<Instances, ParamSpace> paramSpaceFunction) {
        this.paramSpaceFunction = paramSpaceFunction;
        return this;
    }

    public Iterator<Set<Benchmark>> getParamSourceIterator() {
        return paramSourceIterator;
    }

    public IncKnnTunerBuilder setParamSourceIterator(
        final Iterator<Set<Benchmark>> paramSourceIterator) {
        this.paramSourceIterator = paramSourceIterator;
        return this;
    }

    public BenchmarkIterator getBenchmarkSourceIterator() {
        return benchmarkSourceIterator;
    }

    public IncKnnTunerBuilder setBenchmarkSourceIterator(
        final BenchmarkIterator benchmarkSourceIterator) {
        this.benchmarkSourceIterator = benchmarkSourceIterator;
        return this;
    }

    public BenchmarkImprover getBenchmarkImprover() {
        return benchmarkImprover;
    }

    public IncKnnTunerBuilder setBenchmarkImprover(final BenchmarkImprover benchmarkImprover) {
        this.benchmarkImprover = benchmarkImprover;
        return this;
    }

    public Optimiser getOptimiser() {
        return optimiser;
    }

    public IncKnnTunerBuilder setOptimiser(final Optimiser optimiser) {
        this.optimiser = optimiser;
        return this;
    }

    public Supplier<KnnLoocv> getKnnSupplier() {
        return knnSupplier;
    }

    public IncKnnTunerBuilder setKnnSupplier(final Supplier<KnnLoocv> knnSupplier) {
        this.knnSupplier = knnSupplier;
        return this;
    }

    public IncKnnTunerBuilder setParamSpace(Function<Instances, ParamSpace> func) {
        return setParamSpaceFunction(func);
    }

    public IncKnnTunerBuilder setParamSpaceFunction(Supplier<ParamSpace> supplier) {
        return setParamSpace(i -> supplier.get());
    }

    public IncKnnTunerBuilder setParamSpace(Supplier<ParamSpace> supplier) {
        return setParamSpaceFunction(supplier);
    }

    public double getMaxNeighbourhoodSizePercentage() {
        return maxNeighbourhoodSizePercentage;
    }

    public IncKnnTunerBuilder setMaxNeighbourhoodSizePercentage(final double maxNeighbourhoodSizePercentage) {
        this.maxNeighbourhoodSizePercentage = maxNeighbourhoodSizePercentage;
        return this;
    }
}
