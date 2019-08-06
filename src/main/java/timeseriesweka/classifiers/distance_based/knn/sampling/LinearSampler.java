package timeseriesweka.classifiers.distance_based.knn.sampling;

import utilities.iteration.linear.LinearIterator;
import weka.core.Instance;

import java.util.List;

public class LinearSampler extends LinearIterator<Instance> {
    public LinearSampler(final List<Instance> values) {
        super(values);
    }

    public LinearSampler() {
        super();
    }

    public LinearSampler(LinearSampler other) {
        super(other);
    }

    @Override
    public LinearSampler iterator() {
        return new LinearSampler(this);
    }
}
