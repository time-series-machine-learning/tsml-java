package classifiers.distance_based.elastic_ensemble.iteration.wrapped.feedback;

import classifiers.distance_based.elastic_ensemble.iteration.AbstractIterator;
import classifiers.distance_based.elastic_ensemble.iteration.wrapped.limited.LimitedIterator;

public class ThresholdIterator<A> extends AbstractFeedbackIterator<A, Double, Boolean> {

    private Double best = null;
    private final LimitedIterator<A> iterator;

    public ThresholdIterator(final AbstractIterator<A> iterator,
                                     final int threshold) {
        this.iterator = new LimitedIterator<>(iterator, threshold);
    }

    public ThresholdIterator(ThresholdIterator<A> other) {
        this.iterator = other.iterator.iterator();
        this.best = other.best;
    }

    public void resetCount() {
        iterator.resetCount();
    }

    public void resetBest() {
        best = null;
    }

    @Override
    public Boolean feedback(final Double value) {
        if(best == null || value > best) {
            best = value;
            resetCount();
            return true;
        }
        return false;
    }

    @Override
    public AbstractIterator<A> getWrappedIterator() {
        return iterator;
    }

    @Override
    public ThresholdIterator<A> iterator() {
        return new ThresholdIterator<>(this);
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public A next() {
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
}
