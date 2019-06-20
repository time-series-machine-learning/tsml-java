package classifiers.distance_based.elastic_ensemble.iteration.random;

import classifiers.distance_based.elastic_ensemble.iteration.linear.AbstractLinearIterator;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

public abstract class AbstractRandomIterator<A, B extends AbstractRandomIterator<A, B>> extends AbstractLinearIterator<A, AbstractRandomIterator<A, B>> {
    protected final Random random;
    protected long seed;

    public AbstractRandomIterator(final List<A> values, final long seed) {
        super(values);
        this.random = new Random(seed);
    }

    public AbstractRandomIterator(AbstractRandomIterator<A, B> other) {
        this(other.values, other.seed);
        index = other.index;
    }

    public AbstractRandomIterator(long seed) {
        this.random = new Random(seed);
    }

    @Override
    public A next() {
        index = random.nextInt(values.size());
        seed = random.nextLong();
        random.setSeed(seed);
        return values.get(index);
    }

}
