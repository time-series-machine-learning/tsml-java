package classifiers.distance_based.elastic_ensemble.iteration.feedback;

import classifiers.distance_based.elastic_ensemble.iteration.AbstractIterator;

public abstract class AbstractFeedbackIterator<A, B, C>
    extends AbstractIterator<A>
    implements FeedbackIterator<A, B, C> {
    @Override
    public abstract AbstractFeedbackIterator<A, B, C> iterator();
}
