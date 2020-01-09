package machine_learning.classifiers.tuned.incremental.configs;

import machine_learning.classifiers.tuned.incremental.*;
import org.apache.commons.collections4.Transformer;
import org.apache.commons.collections4.iterators.TransformIterator;
import tsml.classifiers.distance_based.knn.KnnLoocv;
import utilities.NumUtils;
import utilities.collections.Best;
import utilities.collections.box.Box;
import utilities.iteration.RandomIterator;
import utilities.params.ParamSet;
import utilities.params.ParamSpace;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

public class IncKnnTunerBuilder implements Consumer<Instances> {


    private IncTuner incTunedClassifier = new IncTuner();
    private ParamSpace paramSpace;
    private BenchmarkExplorer benchmarkExplorer = new BenchmarkExplorer();
    // setup the param space to source classifiers
    private Iterator<ParamSet> paramSetIterator;
    // setup building classifier from param set
    private int maxParamSpaceSize = -1; // max number of params
    private int maxNeighbourhoodSize = -1; // max number of neighbours
    private Box<Integer> neighbourCount = new Box<>(0); // current number of neighbours
    private Box<Integer> paramCount = new Box<>(0); // current number of params
    private Best<Long> maxParamTimeNanos = new Best<>(0L); // track maximum time taken for a param to run
    private Best<Long> maxNeighbourBatchTimeNanos = new Best<>(0L); // track max time taken for an addition of
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

    public IncTuner build() {
        incTunedClassifier.setOnTrainDataAvailable(this);
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

    private boolean hasLimitedParamSpaceSize() {
        return maxParamSpaceSize < 0 || !NumUtils.isPercentage(maxParamSpaceSizePercentage);
    }

    private boolean hasLimitedNeighbourhoodSize() {
        return maxNeighbourhoodSize < 0 || !NumUtils.isPercentage(maxNeighbourhoodSize);
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

    @Override
    public void accept(Instances trainData) {
        improveableBenchmarks = new HashSet<>();
        unimprovableBenchmarks = new HashSet<>();
        final int seed = incTunedClassifier.getSeed();
        paramSpace = paramSpaceFunction.apply(trainData);
        paramSetIterator = new RandomIterator<>(this.paramSpace, seed);
        fullParamSpaceSize = this.paramSpace.size();
        fullNeighbourhoodSize = trainData.size();
        maxNeighbourhoodSize = findLimit(fullNeighbourhoodSize, maxNeighbourhoodSize, maxNeighbourhoodSizePercentage);
        maxParamSpaceSize = findLimit(fullParamSpaceSize, maxParamSpaceSize, maxParamSpaceSizePercentage);
        // transform classifiers into benchmarks
        paramSourceIterator =
            new TransformIterator<>(paramSetIterator,
                                    new Transformer<ParamSet, Set<Benchmark>>() {
                                        private int id = 0;

                                        @Override public Set<Benchmark> transform(final ParamSet paramSet) {
                                            try {
                                                long startTime = System.nanoTime();
                                                long timeTaken = 0;
                                                paramCount.set(paramCount.get() + 1);
                                                KnnLoocv knn = knnSupplier.get();
                                                knn.setNeighbourLimit(neighbourCount.get());
                                                knn.setParams(paramSet);
                                                knn.setEstimateOwnPerformance(true);
                                                timeTaken += System.nanoTime() - startTime;
                                                knn.buildClassifier(trainData);
                                                startTime = System.nanoTime();
                                                timeTaken += knn.getTrainTimeNanos();
                                                Benchmark benchmark = new Benchmark(knn, knn.getTrainResults(), id++);
                                                HashSet<Benchmark> benchmarks = new HashSet<>(
                                                    Collections.singletonList(benchmark));
                                                timeTaken += System.nanoTime() - startTime;
                                                maxParamTimeNanos.add(timeTaken);
                                                return benchmarks;
                                            } catch(Exception e) {
                                                throw new IllegalStateException(e);
                                            }
                                        }
                                    }
            );
        benchmarkSourceIterator = new BenchmarkIterator() {
            @Override public long predictNextTimeNanos() {
                return maxParamTimeNanos.get();
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
                return maxNeighbourBatchTimeNanos.get();
            }

            @Override
            public Set<Benchmark> getImproveableBenchmarks() {
                return improveableBenchmarks;
            }

            public Set<Benchmark> getUnimprovableBenchmarks() {
                return unimprovableBenchmarks;
            }

            public Set<Benchmark> getAllBenchmarks() {
                HashSet<Benchmark> benchmarks = new HashSet<>();
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

            private boolean isImproveable(Benchmark benchmark) {
                try {
                    KnnLoocv knn = (KnnLoocv) benchmark.getClassifier();
                    return knn.getNeighbourLimit() + 1 <= maxNeighbourhoodSize;
                } catch(Exception e) {
                    throw new IllegalStateException(e);
                }
            }

            @Override
            public Set<Benchmark> next() {
                long startTime = System.nanoTime();
                long timeTaken = 0;
                int nextNeighbourCount = neighbourCount.get() + 1;
                neighbourCount.set(nextNeighbourCount);
                Set<Benchmark> improvedBenchmarks = new HashSet<>();
                try {
                    Iterator<Benchmark> benchmarkIterator = improveableBenchmarks.iterator();
                    while(benchmarkIterator.hasNext()) {
                        Benchmark benchmark = benchmarkIterator.next();
                        Classifier classifier = benchmark.getClassifier();
                        KnnLoocv knn = (KnnLoocv) classifier;
                        if(incTunedClassifier.isDebug()) {
                            int currentNeighbourLimit = knn.getNeighbourLimit();
                            if(nextNeighbourCount <= currentNeighbourLimit) {
                                throw new IllegalStateException("no improvement to the number of neighbours");
                            }
                        }
                        knn.setNeighbourLimit(nextNeighbourCount);
                        timeTaken += System.nanoTime() - startTime;
                        long knnPrevTrainTime = knn.getTrainTimeNanos();
                        knn.buildClassifier(trainData);
                        timeTaken += knn.getTrainTimeNanos() - knnPrevTrainTime;
                        startTime = System.nanoTime();
                        benchmark.setResults(knn.getTrainResults());
                        if(!isImproveable(benchmark)) {
                            benchmarkIterator.remove();
                            unimprovableBenchmarks.add(benchmark);
                        }
                        improvedBenchmarks.add(benchmark);
                    }
                } catch(Exception e) {
                    throw new IllegalStateException(e);
                }
                timeTaken += System.nanoTime() - startTime;
                maxNeighbourBatchTimeNanos.add(timeTaken);
                return improvedBenchmarks;
            }

            @Override
            public boolean hasNext() {
                return !improveableBenchmarks.isEmpty();
            }
        };
        benchmarkExplorer.setBenchmarkImprover(benchmarkImprover);
        benchmarkExplorer.setBenchmarkSource(benchmarkSourceIterator);
        optimiser = () -> {
            // only called when *both* improvements and source remain
            int neighbours = neighbourCount.get();
            int params = paramCount.get();
            if(params < maxParamSpaceSize / 10 + 1) {
                // 10% params, 0% neighbours
                return true;
            } else if(neighbours < maxNeighbourhoodSize / 10 + 1) {
                // 10% params, 10% neighbours
                return false;
            } else if(params < maxParamSpaceSize / 2 + 1) {
                // 50% params, 10% neighbours
                return true;
            } else if(neighbours < maxNeighbourhoodSize / 2 + 1) {
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
        benchmarkExplorer.setOptimiser(optimiser);
        // set corresponding iterators in the incremental tuned classifier
        incTunedClassifier.setBenchmarkIterator(benchmarkExplorer);
        incTunedClassifier.setBenchmarkEnsembler(BenchmarkEnsembler.single());
        incTunedClassifier.setBenchmarkCollector(
            new BestBenchmarkCollector(seed, benchmark -> benchmark.getResults().getAcc()));
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

    public BenchmarkExplorer getBenchmarkExplorer() {
        return benchmarkExplorer;
    }

    public IncKnnTunerBuilder setBenchmarkExplorer(final BenchmarkExplorer benchmarkExplorer) {
        this.benchmarkExplorer = benchmarkExplorer;
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

    public Best<Long> getMaxParamTimeNanos() {
        return maxParamTimeNanos;
    }

    public IncKnnTunerBuilder setMaxParamTimeNanos(final Best<Long> maxParamTimeNanos) {
        this.maxParamTimeNanos = maxParamTimeNanos;
        return this;
    }

    public Best<Long> getMaxNeighbourBatchTimeNanos() {
        return maxNeighbourBatchTimeNanos;
    }

    public IncKnnTunerBuilder setMaxNeighbourBatchTimeNanos(final Best<Long> maxNeighbourBatchTimeNanos) {
        this.maxNeighbourBatchTimeNanos = maxNeighbourBatchTimeNanos;
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
