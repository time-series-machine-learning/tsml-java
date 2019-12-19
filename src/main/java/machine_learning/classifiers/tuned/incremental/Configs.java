package machine_learning.classifiers.tuned.incremental;

import org.apache.commons.collections4.Transformer;
import org.apache.commons.collections4.iterators.TransformIterator;
import tsml.classifiers.distance_based.knn.KNNCV;
import utilities.iteration.RandomIterator;
import utilities.params.ParamSet;
import utilities.params.ParamSpace;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;

import static tsml.classifiers.distance_based.distances.Configs.buildDtwSpaceV1;
import static tsml.classifiers.distance_based.knn.Configs.build1nnV1;

public class Configs {

    public static IncTunedClassifier buildTunedDtw1nnV1() {
        IncTunedClassifier incTunedClassifier = new IncTunedClassifier();
        incTunedClassifier.setOnDataFunction(instances1 -> {
            int seed = 0;
            ParamSpace paramSpace = buildDtwSpaceV1(instances1);
            BenchmarkExplorer benchmarkExplorer = new BenchmarkExplorer();
            Iterator<ParamSet> paramSetIterator = new RandomIterator<ParamSet>(paramSpace, seed);
            TransformIterator<ParamSet, Classifier> classifierIterator =
                new TransformIterator<>(paramSetIterator, paramSet -> {
                    KNNCV knn = build1nnV1();
                    knn.setParams(paramSet);
                    return knn;
                });
            TransformIterator<Classifier, Set<Benchmark>> benchmarkSourceIterator =
                new TransformIterator<>(classifierIterator, new Transformer<Classifier, Set<Benchmark>>() {
                    private int id = 0;

                    @Override public Benchmark transform(final Classifier classifier) {
                        return new HashSet<Benchmark>(Collections.singletonList(new Benchmark(classifier, null, id++)));
                    }
                });
            benchmarkExplorer.setBenchmarkSource(classifierIterator);
            incTunedClassifier.setBenchmarkIterator(benchmarkExplorer);
        });
        return incTunedClassifier;
    }

}
