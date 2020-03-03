package tsml.classifiers;

/**
 * Purpose: check whether the classifier is fully trained or not. This is especially important for contracting /
 * progressive classifiers which may only partially build for whatever reason.
 *
 * Contributors: goastler
 */
public interface Trainable {

    boolean isFullyTrained();
}
