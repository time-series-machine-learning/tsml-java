package tsml.classifiers;

import weka.classifiers.Classifier;
import weka.core.Instances;

public interface RebuildableClassifier extends Classifier {

    boolean isRebuild();

    void setRebuild(boolean state);

    default void enableRebuild() {
        setRebuild(true);
    }

    default void disableRebuild() {
        setRebuild(false);
    }

    default void rebuildClassifier(Instances data) throws
                                                   Exception {
        setRebuild(true);
        buildClassifier(data);
    }
}
