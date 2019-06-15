package classifiers.distance_based.elastic_ensemble.iteration.random;

import java.util.Collection;
import java.util.Random;

public class RandomIterator<A> extends AbstractRandomIterator<A, RandomIterator<A>> {

    public RandomIterator(final Collection<? extends A> values, final Random random) {
        super(values, random);
    }

    public RandomIterator(RandomIterator<A> other) {
        this(other.values, other.random);
        index = other.index;
    }

    public RandomIterator(Random random) {
        super(random);
    }

    @Override
    public RandomIterator<A> iterator() {
        return new RandomIterator<>(this);
    }
}
