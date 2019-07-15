package classifiers.distance_based.elastic_ensemble.iteration.random;

import java.util.Collection;
import java.util.List;
import java.util.Random;

public class RandomIterator<A> extends AbstractRandomIterator<A, RandomIterator<A>> {

    public RandomIterator(final List<A> values, final long seed) {
        super(values, seed);
    }


    public RandomIterator(final List<A> values, final Random random) {
        this(values, random.nextLong());
    }

    public RandomIterator(RandomIterator<A> other) {
        this(other.values, other.seed);
        index = other.index;
    }

    public RandomIterator(long seed) {
        super(seed);
    }

    public RandomIterator(Random random) {
        this(random.nextLong());
    }

    @Override
    public RandomIterator<A> iterator() {
        return new RandomIterator<>(this);
    }

    @Override
    public boolean hasNext() {
        return !values.isEmpty();
    }
}
