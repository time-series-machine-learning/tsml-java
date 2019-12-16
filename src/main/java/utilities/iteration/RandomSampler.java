package utilities.iteration;

import weka.core.Instance;

public class RandomSampler extends RandomIterator<Instance> {
    public RandomSampler() {
    }

    public RandomSampler(final int seed) {
        super(seed);
    }
}
