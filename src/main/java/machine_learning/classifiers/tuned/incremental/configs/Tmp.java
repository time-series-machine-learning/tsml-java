package machine_learning.classifiers.tuned.incremental.configs;

import machine_learning.classifiers.tuned.incremental.*;
import org.apache.commons.collections4.Transformer;
import org.apache.commons.collections4.iterators.TransformIterator;
import tsml.classifiers.distance_based.knn.KNNCV;
import utilities.collections.Best;
import utilities.collections.Utils;
import utilities.collections.box.Box;
import utilities.iteration.RandomIterator;
import utilities.params.ParamSet;
import utilities.params.ParamSpace;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.util.*;
import java.util.function.Consumer;

import static tsml.classifiers.distance_based.distances.Configs.buildDtwSpaceV1;
import static tsml.classifiers.distance_based.knn.Configs.build1nnV1;

public class Tmp implements Consumer<Instances> {

    private final IncTunedClassifier incTunedClassifier;
    private final int seed;
    private ParamSpace paramSpace;
    private BenchmarkExplorer benchmarkExplorer = new BenchmarkExplorer();
    // setup the param space to source classifiers
    private Iterator<ParamSet> paramSetIterator;
    // setup building classifier from param set
    private final int maxParamCount; // max number of params
    private int maxNeighbourCount; // max number of neighbours
    private Box<Integer> neighbourCount = new Box<>(0); // current number of neighbours
    private Box<Integer> paramCount = new Box<>(0); // current number of params
    private Best<Long> maxParamTimeNanos = new Best<>(0L); // track maximum time taken for a param to run
    private Best<Long> maxNeighbourBatchTimeNanos = new Best<>(0L); // track max time taken for an addition of
    // neighbours


    public Tmp(IncTunedClassifier incTunedClassifier, int seed) {
        this.incTunedClassifier = incTunedClassifier;
        this.seed = seed;
        paramSetIterator = new RandomIterator<>(paramSpace, seed);
        maxParamCount = paramSpace.size();
    }

    @Override
    public void accept(Instances trainData) {
        maxNeighbourCount = trainData.size();
        // transform classifiers into benchmarks
        Iterator<Set<Benchmark>> paramSourceIterator =
                new TransformIterator<>(paramSetIterator,
                        new Transformer<ParamSet, Set<Benchmark>>() {
                            private int id = 0;

                            @Override public Set<Benchmark> transform(final ParamSet paramSet) {
                                try {
                                    long startTime = System.nanoTime();
                                    long timeTaken = 0;
                                    paramCount.set(paramCount.get() + 1);
                                    KNNCV knn = build1nnV1();
                                    knn.setNeighbourLimit(neighbourCount.get());
                                    knn.setParams(paramSet);
//                            System.out.println(StringUtils.join(paramSet.getOptions(), ", "));
                                    knn.setEstimateOwnPerformance(true);
                                    timeTaken += System.nanoTime() - startTime;
                                    knn.buildClassifier(trainData);
                                    startTime = System.nanoTime();
                                    timeTaken += knn.getTrainTimeNanos();
                                    Benchmark benchmark = new Benchmark(knn, knn.getTrainResults(), id++);
                                    HashSet<Benchmark> benchmarks = new HashSet<>(Collections.singletonList(benchmark));
                                    timeTaken += System.nanoTime() - startTime;
//                            System.out.println("param time: " + timeTaken + " vs " + maxParamTimeNanos.get());
                                    maxParamTimeNanos.add(timeTaken);
                                    return benchmarks;
                                } catch(Exception e) {
                                    throw new IllegalStateException(e);
                                }
                            }
                        }
                );
        BenchmarkIterator benchmarkSourceIterator = new BenchmarkIterator() {
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
        BenchmarkImprover benchmarkImprover = new BenchmarkImprover() {

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
        benchmarkExplorer.setOptimiser(new Optimiser() {
            @Override
            public boolean shouldSource() {
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
//                    else if(neighbours < maxNeighbourCount) {
//                        return false;
//                    }
                else {
                    // by this point all params have been hit. Therefore, shouldSource should not be called at
                    // all as only improvements will remain, if any.
                    throw new IllegalStateException("invalid source / improvement state");
                }
            }
        });
        // set corresponding iterators in the incremental tuned classifier
        incTunedClassifier.setBenchmarkIterator(benchmarkExplorer);
        incTunedClassifier.setBenchmarkEnsembler(benchmarks -> {
            if(Utils.size(benchmarks) != 1) {
                throw new IllegalArgumentException("was only expecting 1 benchmark");
            }
            return new ArrayList<>(Collections.singletonList(1.0));
        });
        incTunedClassifier.setBenchmarkCollector(
                new BestBenchmarkCollector(seed, benchmark -> benchmark.getResults().getAcc()));
    }
    }
}
