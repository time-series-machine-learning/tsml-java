package classifiers.distance_based.elastic_ensemble.iteration;

import java.util.Iterator;

public interface FeedbackIterator<A, B, C> extends Iterator<A> {
    C feedback(B value);
}
