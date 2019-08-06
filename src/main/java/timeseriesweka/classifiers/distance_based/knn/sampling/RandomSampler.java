package timeseriesweka.classifiers.distance_based.knn.sampling;

import utilities.iteration.random.RandomIterator;
import weka.core.Instance;

import java.util.List;
import java.util.Random;

public class RandomSampler extends RandomIterator<Instance> {

    public RandomSampler(final long seed, final List<Instance> values) {
        super(seed, values);
    }

    public RandomSampler(final Random random, final List<Instance> values) {
        super(random, values);
    }

    public RandomSampler(long seed) {
        super(seed);
    }

    public RandomSampler(Random random) {
        super(random);
    }

    public RandomSampler(RandomSampler other) {
        super(other);
    }

    @Override
    public RandomSampler iterator() {
        return new RandomSampler(this);
    }
}
