package tsml.classifiers.distance_based.utils.collections.iteration;

/**
 * A FeedbackIterator iterates over items and receives feedback to guide further next() calls. The feedback should be called after next() as another step in iteration methodology, i.e.:
 * 1) check hasNext()
 * 2) call next()
 * 3) feedback the result from (2) to feedback() (the result should have been modified to include some useful information to guide subsequent next() calls)
 * @param <A>
 */
public interface FeedbackIterator<A> extends DefaultIterator<A>, FeedbackLoop<A> {
    
}
