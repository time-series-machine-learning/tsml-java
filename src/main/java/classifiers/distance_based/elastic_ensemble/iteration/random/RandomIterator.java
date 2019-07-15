package classifiers.distance_based.elastic_ensemble.iteration.random;


import classifiers.distance_based.elastic_ensemble.iteration.linear.LinearIterator;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class RandomIterator<A> extends LinearIterator<A> {
    protected final Random random = new Random();
    protected long seed;

    public RandomIterator(final List<A> values, final long seed) {
        super(values);
        random.setSeed(seed);
    }

    public RandomIterator(final List<A> values, final Random random) {
        this(values, random.nextLong());
    }

    public RandomIterator(RandomIterator<A> other) {
        this(other.values, other.seed);
        index = other.index;
    }

    public RandomIterator(long seed) {
        random.setSeed(seed);
    }

    @Override
    public RandomIterator<A> iterator() {
        return new RandomIterator<>(this);
    }

    @Override
    public A next() {
        index = random.nextInt(values.size());
        seed = random.nextLong();
        random.setSeed(seed);
        return values.get(index);
    }

}
