package classifiers.distance_based.elastic_ensemble.iteration.random;

import classifiers.distance_based.elastic_ensemble.iteration.linear.AbstractLinearIterator;

import java.util.Collection;
import java.util.Random;

public abstract class AbstractRandomIterator<A, B extends AbstractRandomIterator<A, B>> extends AbstractLinearIterator<A, AbstractRandomIterator<A, B>> {
    protected final Random random;

    public AbstractRandomIterator(final Collection<? extends A> values, final Random random) {
        super(values);
        this.random = random;
    }

    public AbstractRandomIterator(AbstractRandomIterator<A, B> other) {
        this(other.values, other.random);
        index = other.index;
    }

    public AbstractRandomIterator(Random random) {
        this.random = random;
    }

    @Override
    public void remove() {
        values.remove(index--);
    }


    @Override
    public A next() {
        return values.get(index = random.nextInt(values.size()));
    }

}
