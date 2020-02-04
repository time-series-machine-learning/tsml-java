package machine_learning.classifiers.tuned.incremental;

import tsml.classifiers.EnhancedAbstractClassifier;
import utilities.collections.DefaultIterator;
import weka.core.Randomizable;

import java.util.Set;

/*
    Explores the tuning space (usually a parameter space). It is an iterator selects the next classifier to examine.
    The classifier is examined by the IncTuner and returned using the feedback function, i.e. reinforcement learning
    style.
 */
public interface Agent extends DefaultIterator<EnhancedAbstractClassifier> {
    default long predictNextTimeNanos() {
        return -1;
    }
    Set<EnhancedAbstractClassifier> findFinalClassifiers();

    boolean feedback(EnhancedAbstractClassifier classifier); // true == can be exploited (again)

    default boolean isExploringOrExploiting() { // must be called after hasNext !!
        return true; // explore == true; exploit == false
    }

}
