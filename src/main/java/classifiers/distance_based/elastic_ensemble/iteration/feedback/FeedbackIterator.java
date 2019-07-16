package classifiers.distance_based.elastic_ensemble.iteration.feedback;

import java.util.Iterator;

public interface FeedbackIterator<A, B, C> extends Iterator<A> {
    C feedback(B value);
}
