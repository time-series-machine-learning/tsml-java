package machine_learning.classifiers.tuned.incremental;

import org.apache.commons.collections4.Transformer;
import org.apache.commons.collections4.iterators.TransformIterator;
import tsml.classifiers.distance_based.knn.KNNCV;
import utilities.collections.Utils;
import utilities.collections.box.Box;
import utilities.iteration.RandomIterator;
import utilities.params.ParamSet;
import utilities.params.ParamSpace;
import weka.classifiers.Classifier;

import java.util.*;

import static tsml.classifiers.distance_based.distances.Configs.buildDtwSpaceV1;
import static tsml.classifiers.distance_based.knn.Configs.build1nnV1;

public class Configs {

    public static IncTunedClassifier buildTunedDtw1nnV1() {
        IncTunedClassifier incTunedClassifier = new IncTunedClassifier();
        incTunedClassifier.setOnTrainDataAvailable(trainData -> {
            int seed = 0;
            ParamSpace paramSpace = buildDtwSpaceV1(trainData);
            BenchmarkExplorer benchmarkExplorer = new BenchmarkExplorer();
            // setup the param space to source classifiers
            Iterator<ParamSet> paramSetIterator = new RandomIterator<>(paramSpace, seed);
            // setup building classifier from param set
            final int maxParamCount = paramSpace.size();
            final int maxNeighbourCount = trainData.size();
            Box<Integer> neighbourCount = new Box<>(0);
            Box<Integer> paramCount = new Box<>(0);
            Box<Integer> neighbourLimit = new Box<>(0);
            // transform classifiers into benchmarks
            Iterator<Set<Benchmark>> benchmarkSourceIterator =
                new TransformIterator<>(paramSetIterator, new Transformer<ParamSet, Set<Benchmark>>() {
                    private int id = 0;

                    @Override public Set<Benchmark> transform(final ParamSet paramSet) {
                        paramCount.set(paramCount.get() + 1);
                        KNNCV knn = build1nnV1();
                        knn.setNeighbourLimit(neighbourLimit.get());
                        knn.setParams(paramSet);
                        Benchmark benchmark = new Benchmark(knn, knn.getTrainResults(), id++);
                        return new HashSet<>(Collections.singletonList(benchmark));
                    }
                });
            // setup an iterator to improve benchmarks
            BenchmarkImprover benchmarkImprover = new BenchmarkImprover() {

                private final Set<Benchmark> improveableBenchmarks = new HashSet<>();
                private final Set<Benchmark> unimprovableBenchmarks = new HashSet<>();

                @Override
                public Set<Benchmark> getImproveableBenchmarks() {
                    return improveableBenchmarks;
                }

                public Set<Benchmark> getUnimprovableBenchmarks() {
                    return unimprovableBenchmarks;
                }

                @Override
                public void add(Benchmark benchmark) {
                    if (isImproveable(benchmark)) {
                        improveableBenchmarks.add(benchmark);
                    }
                }

                private boolean isImproveable(Benchmark benchmark) {
                    try {
                        KNNCV knn = (KNNCV) benchmark.getClassifier();
                        return knn.getNeighbourLimit() + 1 < neighbourCount.get();
                    } catch (Exception e) {
                        throw new IllegalStateException(e);
                    }
                }

                @Override
                public Set<Benchmark> next() {
                    neighbourCount.set(neighbourCount.get() + 1);
                    Set<Benchmark> improvedBenchmarks = new HashSet<>();
                    try {
                        for (Benchmark benchmark : improveableBenchmarks) {
                            Classifier classifier = benchmark.getClassifier();
                            KNNCV knn = (KNNCV) classifier;
                            int nextNeighbourLimit = neighbourLimit.get();
                            if(incTunedClassifier.isDebug()) {
                                int currentNeighbourLimit = knn.getNeighbourLimit() + 1;
                                if(nextNeighbourLimit <= currentNeighbourLimit) {
                                    throw new IllegalStateException("no improvement to the number of neighbours");
                                }
                            }
                            knn.setNeighbourLimit(nextNeighbourLimit);
                            knn.buildClassifier(trainData);
                            benchmark.setResults(knn.getTrainResults());
                            if (!isImproveable(benchmark)) {
                                improveableBenchmarks.remove(benchmark);
                                unimprovableBenchmarks.add(benchmark);
                            }
                            improvedBenchmarks.add(benchmark);
                        }
                    } catch (Exception e) {
                        throw new IllegalStateException(e);
                    }
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
                    if(params < maxParamCount / 10) {
                        // 10% params, 0% neighbours
                        return true;
                    } else if(neighbours < maxNeighbourCount / 10) {
                        // 10% params, 10% neighbours
                        return false;
                    } else if(params < maxParamCount / 2) {
                        // 50% params, 10% neighbours
                        return true;
                    } else if(neighbours < maxNeighbourCount / 2) {
                        // 50% params, 50% neighbours
                        return false;
                    } else if(params < maxParamCount) {
                        // 100% params, 50% neighbours
                        return true;
                    } else {
                        // by this point all params should have been hit. Therefore only improvements remain, so
                        // this question of whether to source a new benchmark or improve a current should evaluate to
                        // improvement as no source remains.
                        throw new IllegalStateException("invalid source / improvement state");
                    }
                }
            });
            // set corresponding iterators in the incremental tuned classifier
            incTunedClassifier.setBenchmarkIterator(benchmarkExplorer);
            incTunedClassifier.setBenchmarkEnsembler(new BenchmarkEnsembler() {
                @Override
                public List<Double> weightVotes(Iterable<Benchmark> benchmarks) {
                    if(Utils.size(benchmarks) != 1) {
                        throw new IllegalArgumentException("was only expecting 1 benchmark");
                    }
                    return new ArrayList<>(Collections.singletonList(1.0));
                }
            });
            incTunedClassifier.setBenchmarkCollector(new BestBenchmarkCollector(seed, benchmark -> benchmark.getResults().getAcc()));
        });
        return incTunedClassifier;
    }

}
