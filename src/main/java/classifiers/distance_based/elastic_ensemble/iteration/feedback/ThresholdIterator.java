package classifiers.distance_based.elastic_ensemble.iteration.feedback;

import classifiers.distance_based.elastic_ensemble.iteration.DynamicIterator;
import classifiers.distance_based.elastic_ensemble.iteration.limited.AbstractLimitedIterator;
import classifiers.distance_based.elastic_ensemble.iteration.limited.LimitedIterator;

public class ThresholdIterator<A> extends AbstractThresholdIterator<A, ThresholdIterator<A>> {
    public ThresholdIterator(final DynamicIterator<A, ?> iterator,
                             final int threshold) {
        super(iterator, threshold);
    }

    public ThresholdIterator(final AbstractThresholdIterator<A, ThresholdIterator<A>> other) {
        super(other);
    }

    @Override
    public ThresholdIterator<A> iterator() {
        return new ThresholdIterator<>(this);
    }
}
