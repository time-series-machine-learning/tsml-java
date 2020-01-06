package machine_learning.classifiers.tuned.incremental;

import evaluation.storage.ClassifierResults;
import experiments.data.DatasetLoading;
import org.apache.commons.collections4.Transformer;
import org.apache.commons.collections4.iterators.AbstractIteratorDecorator;
import org.apache.commons.collections4.iterators.TransformIterator;
import tsml.classifiers.distance_based.knn.KNNCV;
import utilities.ClassifierTools;
import utilities.Utilities;
import utilities.collections.Best;
import utilities.collections.Utils;
import utilities.collections.box.Box;
import utilities.iteration.RandomIterator;
import utilities.params.ParamSet;
import utilities.params.ParamSpace;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.concurrent.TimeUnit;

import static tsml.classifiers.distance_based.distances.Configs.buildDtwSpaceV1;
import static tsml.classifiers.distance_based.knn.Configs.build1nnV1;

public class Configs {

    public static void main(String[] args) throws Exception {
        int seed = 0;
        Instances[] instances = DatasetLoading.sampleItalyPowerDemand(seed);
        Instances train = instances[0];
        Instances test = instances[1];
        IncTunedClassifier incTunedClassifier = buildTunedDtw1nnV1();
        incTunedClassifier.setTrainTimeLimit(5, TimeUnit.MINUTES);
        incTunedClassifier.buildClassifier(train);
        ClassifierResults results = new ClassifierResults();
        results.setDetails(incTunedClassifier, train);
        for(Instance testCase : test) {
            long startTime = System.nanoTime();
            double[] distribution = incTunedClassifier.distributionForInstance(testCase);
            long timeTaken = System.nanoTime() - startTime;
            int prediction = Utilities.argMax(distribution, new Random(seed));
            results.addPrediction(testCase.classValue(), distribution, prediction, timeTaken, "");
        }
        System.out.println(results.getAcc());
    }

    public static IncTunedClassifier buildTunedDtw1nnV1() {
        IncTunedClassifier incTunedClassifier = new IncTunedClassifier();
        incTunedClassifier.setOnTrainDataAvailable(trainData -> {
            int seed = 0;
            ParamSpace paramSpace = buildDtwSpaceV1(trainData);
            BenchmarkExplorer benchmarkExplorer = new BenchmarkExplorer();
            // setup the param space to source classifiers
            Iterator<ParamSet> paramSetIterator = new RandomIterator<>(paramSpace, seed);
            // setup building classifier from param set
            final int maxParamCount = paramSpace.size(); // max number of params
            final int maxNeighbourCount = trainData.size(); // max number of neighbours
            Box<Integer> neighbourCount = new Box<>(0); // current number of neighbours
            Box<Integer> paramCount = new Box<>(0); // current number of params
            Best<Long> maxParamTimeNanos = new Best<>(0L); // track maximum time taken for a param to run
            Best<Long> maxNeighbourBatchTimeNanos = new Best<>(0L); // track max time taken for an addition of
            // neighbours
            // transform classifiers into benchmarks
            Iterator<Set<Benchmark>> benchmarkSourceIterator =
                new TransformIterator<>(new AbstractIteratorDecorator<ParamSet>(paramSetIterator) {
                    @Override public boolean hasNext() {
                        boolean result = super.hasNext();
                        if(result) {
                            result = incTunedClassifier.getRemainingTrainTimeNanos() > maxParamTimeNanos.get();
                        }
                        return result;
                    }
                },
                new Transformer<ParamSet, Set<Benchmark>>() {
                    private int id = 0;

                    @Override public Set<Benchmark> transform(final ParamSet paramSet) {
                        try {
                            long startTime = System.nanoTime();
                            paramCount.set(paramCount.get() + 1);
                            KNNCV knn = build1nnV1();
                            knn.setNeighbourLimit(neighbourCount.get());
                            knn.setParams(paramSet);
                            knn.buildClassifier(trainData);
                            Benchmark benchmark = new Benchmark(knn, knn.getTrainResults(), id++);
                            HashSet<Benchmark> benchmarks = new HashSet<>(Collections.singletonList(benchmark));
                            long timeTaken = System.nanoTime() - startTime;
                            maxParamTimeNanos.add(timeTaken);
                            return benchmarks;
                        } catch(Exception e) {
                            throw new IllegalStateException(e);
                        }
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
                    if(isImproveable(benchmark)) {
                        improveableBenchmarks.add(benchmark);
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
                            knn.buildClassifier(trainData);
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
                    long timeTaken = System.nanoTime() - startTime;
                    maxNeighbourBatchTimeNanos.add(timeTaken);
                    return improvedBenchmarks;
                }

                @Override
                public boolean hasNext() {
                    return !improveableBenchmarks.isEmpty() &&
                        incTunedClassifier.getRemainingTrainTimeNanos() > maxNeighbourBatchTimeNanos.get();
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
            incTunedClassifier.setBenchmarkEnsembler(new BenchmarkEnsembler() {
                @Override
                public List<Double> weightVotes(Iterable<Benchmark> benchmarks) {
                    if(Utils.size(benchmarks) != 1) {
                        throw new IllegalArgumentException("was only expecting 1 benchmark");
                    }
                    return new ArrayList<>(Collections.singletonList(1.0));
                }
            });
            incTunedClassifier.setBenchmarkCollector(
                new BestBenchmarkCollector(seed, benchmark -> benchmark.getResults().getAcc()));
        });
        return incTunedClassifier;
    }

}
