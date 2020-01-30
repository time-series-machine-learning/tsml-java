package tsml.classifiers;

import weka.classifiers.Classifier;
import weka.core.Instances;

public interface Rebuildable {

    boolean isRebuild();

    void setRebuild(boolean state);

    boolean isBuilt();
}
