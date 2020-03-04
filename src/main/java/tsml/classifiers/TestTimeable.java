package tsml.classifiers;

/**
 * Purpose: get the test time. Implement if the classifier tracks the time taken to classify.
 *
 * Contributors: goastler
 */
public interface TestTimeable {
    default long getTestTimeNanos() { return -1; };
}
