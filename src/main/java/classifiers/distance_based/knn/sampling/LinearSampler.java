package classifiers.distance_based.knn.sampling;

import classifiers.distance_based.elastic_ensemble.iteration.LinearIterator;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Iterator;
import java.util.List;

public class LinearSampler extends LinearIterator<Instance>
    implements Sampler {
    public LinearSampler(final List<? extends Instance> values) {
        super(values);
    }
}
