package tsml.classifiers.distance_based.utils.classifier_mixins;

/**
 * Purpose: is the classifier fully built yet? This is especially important for contracted / restricted classifiers
 * which may not build fully for whatever reason. Other classifiers (e.g. ensembles) may adjust their behaviour based
 * upon whether the constituent is built or not.
 *
 * Contributors: goastler
 */
public interface Buildable {
    boolean isBuilt();
}
