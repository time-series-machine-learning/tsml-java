package tsml.classifiers.distance_based.pf.relative;

import weka.core.Instance;
import weka.core.Instances;

import java.util.List;

public interface ExemplarPicker {
    List<Instance> pickExemplars(Instances data);
}
