package tsml.classifiers.distance_based.pf;

import weka.core.Instances;

public interface Splittable {
    default Split split(Instances data) {
        setSplitInputData(data);
        return split();
    }
    Split split();
    void setSplitInputData(Instances data);
    Instances getSplitInputData();
}
