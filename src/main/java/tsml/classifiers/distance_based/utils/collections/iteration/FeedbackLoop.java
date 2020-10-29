package tsml.classifiers.distance_based.utils.collections.iteration;

public interface FeedbackLoop<A> {

    /**
     * Feedback some information on the last next() call to guide further next() calls.
     * @param feedback the feedback to be acted upon.
     */
    default void feedback(A feedback) {

    }
    
    default void feedback(Iterable<A> iterable) {
        for(A feedback : iterable) {
            feedback(feedback);
        }
    }
}
