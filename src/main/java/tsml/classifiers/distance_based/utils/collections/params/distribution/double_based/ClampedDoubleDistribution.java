package tsml.classifiers.distance_based.utils.collections.params.distribution.double_based;

import tsml.classifiers.distance_based.utils.collections.params.distribution.ClampedDistribution;

public abstract class ClampedDoubleDistribution extends ClampedDistribution<Double> implements DoubleDistribution {
    
    public ClampedDoubleDistribution() {
        this(0d, 1d);
    }
    
    public ClampedDoubleDistribution(Double end) {
        this(0d, end);
    }
    
    public ClampedDoubleDistribution(Double start, Double end) {
        super(start, end);
    }
}
