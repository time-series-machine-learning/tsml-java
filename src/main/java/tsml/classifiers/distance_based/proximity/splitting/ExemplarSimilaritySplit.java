package tsml.classifiers.distance_based.proximity.splitting;

import tsml.classifiers.distance_based.proximity.Split;
import weka.core.DistanceFunction;
import weka.core.Instance;

import java.util.List;

public class ExemplarSimilaritySplit extends Split {

    private DistanceFunction distanceFunction;
    private List<Instance> exemplars;
}
