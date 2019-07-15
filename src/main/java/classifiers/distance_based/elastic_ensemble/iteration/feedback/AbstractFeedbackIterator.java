package classifiers.distance_based.elastic_ensemble.iteration.feedback;

import classifiers.distance_based.elastic_ensemble.iteration.DynamicIterator;
import classifiers.distance_based.elastic_ensemble.iteration.FeedbackIterator;
import classifiers.distance_based.elastic_ensemble.iteration.limited.AbstractLimitedIterator;

public abstract class AbstractFeedbackIterator<A, B extends AbstractFeedbackIterator<A, B, C, D>, C, D> extends DynamicIterator<A, AbstractFeedbackIterator<A, B, C, D>> implements FeedbackIterator<A, C, D> {
}
