package tsml.classifiers.distance_based.utils.collections.params.distribution.int_based;

import tsml.classifiers.distance_based.utils.collections.params.distribution.ClampedDistribution;

public abstract class ClampedIntDistribution extends ClampedDistribution<Integer> implements IntDistribution {
    public ClampedIntDistribution() {
        this(0, 1);
    }
    
    public ClampedIntDistribution(Integer end) {
        this(0, end);
    }
    
    public ClampedIntDistribution(Integer start, Integer end) {
        super(start, end);
    }
}
