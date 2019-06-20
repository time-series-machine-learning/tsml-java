package classifiers.distance_based.knn.sampling;

import classifiers.distance_based.elastic_ensemble.iteration.linear.AbstractLinearIterator;
import classifiers.distance_based.elastic_ensemble.iteration.linear.LinearIterator;
import weka.core.Instance;

import java.util.List;

public class LinearSampler extends LinearIterator<Instance> {
    public LinearSampler(final List<Instance> values) {
        super(values);
    }

    public LinearSampler() {
        super();
    }
}
