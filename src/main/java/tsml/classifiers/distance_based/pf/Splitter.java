package tsml.classifiers.distance_based.pf;

import weka.core.Instances;

public interface Splitter {
    Split split(Instances data);
}
