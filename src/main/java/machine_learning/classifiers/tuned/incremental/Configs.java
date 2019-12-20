package machine_learning.classifiers.tuned.incremental;

import org.apache.commons.collections4.Transformer;
import org.apache.commons.collections4.iterators.TransformIterator;
import tsml.classifiers.distance_based.knn.KNNCV;
import utilities.collections.Utils;
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
        incTunedClassifier.setOnDataFunction(trainData -> {
            int seed = 0;
            ParamSpace paramSpace = buildDtwSpaceV1(trainData);
            BenchmarkExplorer benchmarkExplorer = new BenchmarkExplorer();
            // setup the param space to source classifiers
            Iterator<ParamSet> paramSetIterator = new RandomIterator<ParamSet>(paramSpace, seed);
            // setup building classifier from param set
            TransformIterator<ParamSet, Classifier> classifierIterator =
                new TransformIterator<>(paramSetIterator, paramSet -> {
                    KNNCV knn = build1nnV1();
                    knn.setNeighbourLimit(1);
                    knn.setParams(paramSet);
                    // todo build
                    return knn;
                });
            // transform classifiers into benchmarks
            TransformIterator<Classifier, Set<Benchmark>> benchmarkSourceIterator =
                new TransformIterator<>(classifierIterator, new Transformer<Classifier, Set<Benchmark>>() {
                    private int id = 0;

                    @Override public Set<Benchmark> transform(final Classifier classifier) {
                        Benchmark benchmark = new Benchmark(classifier, null, id++);
                        return new HashSet<>(Collections.singletonList(benchmark));
                    }
                });
            // setup an iterator to improve benchmarks
            BenchmarkImprover benchmarkImprover = new BenchmarkImprover() {

                private final Set<Benchmark> improveableBenchmarks = new HashSet<>();

                @Override
                public Set<Benchmark> getImproveableBenchmarks() {
                    return improveableBenchmarks;
                }

                @Override
                public void add(Benchmark benchmark) {
                    if (improveable(benchmark)) {
                        improveableBenchmarks.add(benchmark);
                    }
                }

                private boolean improveable(Benchmark benchmark) {
                    try {
                        KNNCV knn = (KNNCV) benchmark.getClassifier();
                        return knn.getNeighbourLimit() + 1 < trainData.size();
                    } catch (Exception e) {
                        throw new IllegalStateException(e);
                    }
                }

                @Override
                public Set<Benchmark> next() {
                    Set<Benchmark> improvedBenchmarks = new HashSet<>();
                    try {
                        for (Benchmark benchmark : improveableBenchmarks) {
                            Classifier classifier = benchmark.getClassifier();
                            KNNCV knn = (KNNCV) classifier;
                            knn.setNeighbourLimit(knn.getNeighbourLimit() + 1);
                            knn.buildClassifier(trainData);
                            if (!improveable(benchmark)) {
                                improveableBenchmarks.remove(benchmark);
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
            benchmarkExplorer.setGuide(new BenchmarkExplorer.Guide() {
                @Override
                public boolean shouldSource() {
                    int count = 0;
                    if(count < trainData.size() / 10) {
                        return true;
                    } else {
                        // more than 10% of neighbours
                        if(benchmarkImprover.hasNext()) {
                            // improve those 10% neighbour versions
                            return false;
                        } else {
                            // no more 10% improvement available
                            if(count < trainData.size() / 2) {
                                // source until 50%
                                return true;
                            } else {
                                // more than 50% of neighbours
                                if(benchmarkImprover.hasNext()) {
                                    return true;
                                } else {
                                    // todo make sure knns resume build from last % of neighbours upon sourcing new param
                                    return true;
                                }
                            }
                        }
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
