package classifiers.distance_based.elastic_ensemble.iteration.wrapped;

import classifiers.distance_based.elastic_ensemble.iteration.AbstractIterator;

public interface WrappedIterator<A> {
    AbstractIterator<A> getWrappedIterator();
}
