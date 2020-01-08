package machine_learning.classifiers.tuned.incremental.configs;

import machine_learning.classifiers.tuned.incremental.*;
import org.apache.commons.collections4.Transformer;
import org.apache.commons.collections4.iterators.TransformIterator;
import tsml.classifiers.distance_based.knn.KNNCV;
import utilities.collections.Best;
import utilities.collections.box.Box;
import utilities.iteration.RandomIterator;
import utilities.params.ParamSet;
import utilities.params.ParamSpace;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

import static tsml.classifiers.distance_based.knn.Configs.build1nnV1;

public class Inc1nnTuningSetup implements Consumer<Instances> {

    private IncTunedClassifier incTunedClassifier;
    private ParamSpace paramSpace;
    private BenchmarkExplorer benchmarkExplorer = new BenchmarkExplorer();
    // setup the param space to source classifiers
    private Iterator<ParamSet> paramSetIterator;
    // setup building classifier from param set
    private int maxParamCount; // max number of params
    private int maxNeighbourCount; // max number of neighbours
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
    private final Supplier<KNNCV> knnSupplier;

    public Inc1nnTuningSetup(IncTunedClassifier incTunedClassifier,
                             final Function<Instances, ParamSpace> paramSpaceFunction,
                             final Supplier<KNNCV> knnSupplier) {
        this.incTunedClassifier = incTunedClassifier;
        this.paramSpaceFunction = paramSpaceFunction;
        this.knnSupplier = knnSupplier;
    }

    public Inc1nnTuningSetup(IncTunedClassifier incTunedClassifier, ParamSpace paramSpace,
                             final Supplier<KNNCV> knnSupplier) {
        this(incTunedClassifier, (instances) -> paramSpace, knnSupplier);
    }

    @Override
    public void accept(Instances trainData) {
        final int seed = incTunedClassifier.getSeed();
        paramSpace = paramSpaceFunction.apply(trainData);
        paramSetIterator = new RandomIterator<>(this.paramSpace, seed);
        maxParamCount = this.paramSpace.size();
        maxNeighbourCount = trainData.size();
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
                                    KNNCV knn = knnSupplier.get();
                                    knn.setNeighbourLimit(neighbourCount.get());
                                    knn.setParams(paramSet);
                                    knn.setEstimateOwnPerformance(true);
                                    timeTaken += System.nanoTime() - startTime;
                                    knn.buildClassifier(trainData);
                                    startTime = System.nanoTime();
                                    timeTaken += knn.getTrainTimeNanos();
                                    Benchmark benchmark = new Benchmark(knn, knn.getTrainResults(), id++);
                                    HashSet<Benchmark> benchmarks = new HashSet<>(Collections.singletonList(benchmark));
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
                return paramSourceIterator.hasNext();
            }
        };
        // setup an iterator to improve benchmarks
        benchmarkImprover = new BenchmarkImprover() {

            private final Set<Benchmark> improveableBenchmarks = new HashSet<>();
            private final Set<Benchmark> unimprovableBenchmarks = new HashSet<>();

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
                    KNNCV knn = (KNNCV) benchmark.getClassifier();
                    return knn.getNeighbourLimit() + 1 <= maxNeighbourCount;
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
                        KNNCV knn = (KNNCV) classifier;
                        if(incTunedClassifier.isDebug()) {
                            int currentNeighbourLimit = knn.getNeighbourLimit() + 1;
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
//                    System.out.println("neighbour time: " + timeTaken + " vs " + maxNeighbourBatchTimeNanos.get());
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
            if(params < maxParamCount / 10 + 1) {
                // 10% params, 0% neighbours
                return true;
            } else if(neighbours < maxNeighbourCount / 10 + 1) {
                // 10% params, 10% neighbours
                return false;
            } else if(params < maxParamCount / 2 + 1) {
                // 50% params, 10% neighbours
                return true;
            } else if(neighbours < maxNeighbourCount / 2 + 1) {
                // 50% params, 50% neighbours
                return false;
            } else if(params < maxParamCount) {
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

    public IncTunedClassifier getIncTunedClassifier() {
        return incTunedClassifier;
    }

    public ParamSpace getParamSpace() {
        return paramSpace;
    }

    public BenchmarkExplorer getBenchmarkExplorer() {
        return benchmarkExplorer;
    }

    public Iterator<ParamSet> getParamSetIterator() {
        return paramSetIterator;
    }

    public int getMaxParamCount() {
        return maxParamCount;
    }

    public int getMaxNeighbourCount() {
        return maxNeighbourCount;
    }

    public Box<Integer> getNeighbourCount() {
        return neighbourCount;
    }

    public Box<Integer> getParamCount() {
        return paramCount;
    }

    public Best<Long> getMaxParamTimeNanos() {
        return maxParamTimeNanos;
    }

    public Best<Long> getMaxNeighbourBatchTimeNanos() {
        return maxNeighbourBatchTimeNanos;
    }

    public Function<Instances, ParamSpace> getParamSpaceFunction() {
        return paramSpaceFunction;
    }

    public Iterator<Set<Benchmark>> getParamSourceIterator() {
        return paramSourceIterator;
    }

    public BenchmarkIterator getBenchmarkSourceIterator() {
        return benchmarkSourceIterator;
    }

    public BenchmarkImprover getBenchmarkImprover() {
        return benchmarkImprover;
    }

    public Optimiser getOptimiser() {
        return optimiser;
    }

    public void setIncTunedClassifier(final IncTunedClassifier incTunedClassifier) {
        this.incTunedClassifier = incTunedClassifier;
    }

    public void setParamSpace(final ParamSpace paramSpace) {
        this.paramSpace = paramSpace;
    }

    public void setBenchmarkExplorer(final BenchmarkExplorer benchmarkExplorer) {
        this.benchmarkExplorer = benchmarkExplorer;
    }

    public void setParamSetIterator(final Iterator<ParamSet> paramSetIterator) {
        this.paramSetIterator = paramSetIterator;
    }

    public void setMaxParamCount(final int maxParamCount) {
        this.maxParamCount = maxParamCount;
    }

    public void setMaxNeighbourCount(final int maxNeighbourCount) {
        this.maxNeighbourCount = maxNeighbourCount;
    }

    public void setNeighbourCount(final Box<Integer> neighbourCount) {
        this.neighbourCount = neighbourCount;
    }

    public void setParamCount(final Box<Integer> paramCount) {
        this.paramCount = paramCount;
    }

    public void setMaxParamTimeNanos(final Best<Long> maxParamTimeNanos) {
        this.maxParamTimeNanos = maxParamTimeNanos;
    }

    public void setMaxNeighbourBatchTimeNanos(final Best<Long> maxNeighbourBatchTimeNanos) {
        this.maxNeighbourBatchTimeNanos = maxNeighbourBatchTimeNanos;
    }

    public void setParamSpaceFunction(
        final Function<Instances, ParamSpace> paramSpaceFunction) {
        this.paramSpaceFunction = paramSpaceFunction;
    }

    public void setParamSourceIterator(
        final Iterator<Set<Benchmark>> paramSourceIterator) {
        this.paramSourceIterator = paramSourceIterator;
    }

    public void setBenchmarkSourceIterator(
        final BenchmarkIterator benchmarkSourceIterator) {
        this.benchmarkSourceIterator = benchmarkSourceIterator;
    }

    public void setBenchmarkImprover(final BenchmarkImprover benchmarkImprover) {
        this.benchmarkImprover = benchmarkImprover;
    }

    public void setOptimiser(final Optimiser optimiser) {
        this.optimiser = optimiser;
    }
}
