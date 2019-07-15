package classifiers.distance_based.knn.sampling;

import classifiers.distance_based.elastic_ensemble.iteration.random.RandomIterator;
import weka.core.Instance;

import java.util.List;

public class RandomSampler extends RandomIterator<Instance> {

    public RandomSampler(final List<Instance> values, final long seed) {
        super(values, seed);
    }

    public RandomSampler(long seed) {
        super(seed);
    }

    public RandomSampler(RandomSampler other) {
        super(other);
    }

    @Override
    public RandomSampler iterator() {
        return new RandomSampler(this);
    }
}
