package classifiers.distance_based.knn.sampling;

import classifiers.distance_based.elastic_ensemble.iteration.RandomIterator;
import weka.core.Instance;
import weka.core.Instances;

import java.util.List;
import java.util.Random;

public class RandomSampler extends RandomIterator<Instance> implements Sampler {

    public RandomSampler(final List<? extends Instance> values, final Random random) {
        super(values, random);
    }
}
