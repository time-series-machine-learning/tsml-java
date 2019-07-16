package classifiers.distance_based.elastic_ensemble.iteration.random;


import classifiers.distance_based.elastic_ensemble.iteration.linear.LinearIterator;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class RandomIterator<A> extends LinearIterator<A> {
    protected final Random random = new Random();
    protected long seed;

    public RandomIterator(final long seed, final List<A> values) {
        super(values);
        random.setSeed(seed);
        this.seed = seed;
    }

    public RandomIterator(final Random random, final List<A> values) {
        this(random.nextLong(), values);
    }

    public RandomIterator(RandomIterator<A> other) {
        this(other.seed, other.values);
        index = other.index;
    }

    public RandomIterator(long seed) {
        this(seed, new ArrayList<>());
    }

    public RandomIterator(Random random) {
        this(random.nextLong(), new ArrayList<>());
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
