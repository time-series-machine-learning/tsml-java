package tsml.classifiers;

import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Purpose: whether to rebuild the classifier upon buildClassifier() call. This is useful for incremental builders
 * where every call to buildClassifier may not be a full rebuild from scratch.
 *
 * Contributors: goastler
 */
public interface Retrainable {

    boolean isRebuild();

    void setRebuild(boolean state);
}
