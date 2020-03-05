package tsml.classifiers.distance_based.utils.classifier_mixins;

/**
 * Purpose: manage parallelism. At the moment this is just to detect whether we've yielded to another process, but it
 * could control parallelisation further in future.
 *
 * Contributors: goastler
 */
public interface Parallelisable {
    boolean hasYielded();
}
