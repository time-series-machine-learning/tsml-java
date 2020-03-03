package tsml.classifiers;

import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Purpose: whether to rebuild the classifier on buildClassifier call. Incremental / progressive / contracted
 * classifiers may continue their build every time buildClassifier is called (say, if the contract time has been
 * increased). We may not want this to be the case, and to in fact do a fresh build.
 *
 * Contributors: goastler
 */
public interface Rebuildable {

    boolean isRebuild();

    void setRebuild(boolean state);
}
