package classifiers.distance_based.knn.sampling;

import classifiers.distance_based.elastic_ensemble.iteration.random.AbstractRandomIterator;
import classifiers.distance_based.elastic_ensemble.iteration.random.RandomIterator;
import weka.core.Instance;

import java.util.List;
import java.util.Random;

public class RandomSampler extends RandomIterator<Instance> {

    public RandomSampler(final List<? extends Instance> values, final Random random) {
        super(values, random);
    }

    public RandomSampler(Random random) {
        super(random);
    }
}
