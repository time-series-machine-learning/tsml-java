package classifiers.distance_based.elastic_ensemble.iteration.wrapped.limited;

import classifiers.distance_based.elastic_ensemble.iteration.AbstractIterator;
import classifiers.distance_based.elastic_ensemble.iteration.wrapped.AbstractWrappedIterator;

public class LimitedIterator<A> extends AbstractWrappedIterator<A> {

    private final AbstractIterator<A> iterator;
    private final int limit;
    private int count = 0;

    @Override
    public AbstractIterator<A> getWrappedIterator() {
        return iterator;
    }

    public LimitedIterator(final AbstractIterator<A> iterator, final int limit) {this.iterator = iterator;
        this.limit = limit;
    }

    public LimitedIterator(LimitedIterator<A> other) {
        this(other.iterator, other.limit);
        count = other.count;
    }

    @Override
    public LimitedIterator<A> iterator() {
        return new LimitedIterator<>(this);
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
