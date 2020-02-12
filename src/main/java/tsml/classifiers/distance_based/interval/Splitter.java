package tsml.classifiers.distance_based.interval;

import weka.core.Instances;

import java.util.List;

public interface Splitter {
    Split split(Instances data);
}
