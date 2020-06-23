package tsml.classifiers.distance_based.utils.classifiers;

/**
 * Purpose: get the test time. Implement if the classifier tracks the time taken to classify.
 *
 * Contributors: goastler
 */
public interface TestTimeable {
    long getTestTime();
}
