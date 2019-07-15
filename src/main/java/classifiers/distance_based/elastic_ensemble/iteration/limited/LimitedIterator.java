package classifiers.distance_based.elastic_ensemble.iteration.limited;

import classifiers.distance_based.elastic_ensemble.iteration.DynamicIterator;
import classifiers.distance_based.elastic_ensemble.iteration.linear.AbstractLinearIterator;
import classifiers.distance_based.elastic_ensemble.iteration.linear.LinearIterator;

public class LimitedIterator<A> extends AbstractLimitedIterator<A, LimitedIterator<A>> {
    public LimitedIterator(final DynamicIterator<A, ?> iterator,
                           final int limit) {
        super(iterator, limit);
    }

    public LimitedIterator(final AbstractLimitedIterator<A, LimitedIterator<A>> other) {
        super(other);
    }

    @Override
    public LimitedIterator<A> iterator() {
        return new LimitedIterator<>(this);
    }
}
