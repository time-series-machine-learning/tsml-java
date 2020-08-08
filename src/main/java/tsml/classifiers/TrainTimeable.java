package tsml.classifiers;

/**
 * Purpose: track the time taken to train a classifier.
 *
 * Contributors: goastler
 */
public interface TrainTimeable {
    default long getTrainTime() { return -1; }
}
