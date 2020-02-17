package tsml.classifiers.distance_based.pf.relative;

import utilities.Rand;
import weka.core.Instance;
import weka.core.Instances;

import java.util.List;

public interface ExemplarPicker extends Rand {
    List<Instance> pickExemplars(Instances data);
}
