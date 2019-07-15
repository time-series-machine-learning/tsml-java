package classifiers.distance_based.elastic_ensemble.iteration.limited;

import classifiers.distance_based.elastic_ensemble.iteration.DynamicIterator;
import classifiers.distance_based.elastic_ensemble.iteration.linear.AbstractLinearIterator;

public class AbstractLimitedIterator<A, B extends AbstractLimitedIterator<A, B>> extends DynamicIterator<A, AbstractLimitedIterator<A, B>> {

    private final DynamicIterator<A, ?> iterator;
    private final int limit;
    private int count = 0;

    public AbstractLimitedIterator(final DynamicIterator<A, ?> iterator, final int limit) {this.iterator = iterator;
        this.limit = limit;
    }

    public AbstractLimitedIterator(AbstractLimitedIterator<A, B> other) {
        this(other.iterator, other.limit);
        count = other.count;
    }

    @Override
    public AbstractLimitedIterator<A, B> iterator() {
        return new AbstractLimitedIterator<>(this);
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext() && count < limit;
    }

    @Override
    public A next() {
        count++;
        return iterator.next();
    }

    @Override
    public void remove() {
        iterator.remove();
    }

    @Override
    public void add(final A a) {
        iterator.add(a);
    }

    public void resetCount() {
        count = 0;
    }
}
