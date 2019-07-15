package classifiers.distance_based.elastic_ensemble.iteration.feedback;

import classifiers.distance_based.elastic_ensemble.iteration.DynamicIterator;
import classifiers.distance_based.elastic_ensemble.iteration.limited.LimitedIterator;

public class AbstractThresholdIterator<A, B extends AbstractThresholdIterator<A, B>> extends AbstractFeedbackIterator<A, AbstractThresholdIterator<A, B>, Double, Boolean> {

    private Double best = null;
    private final LimitedIterator<A> iterator;

    public AbstractThresholdIterator(final DynamicIterator<A, ?> iterator,
                                     final int threshold) {
        this.iterator = new LimitedIterator<>(iterator, threshold);
    }

    public AbstractThresholdIterator(AbstractThresholdIterator<A, B> other) {
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
    public AbstractThresholdIterator<A, B> iterator() {
        return new AbstractThresholdIterator<>(this);
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
