package tsml.classifiers.distance_based.utils.collections.params.distribution.double_based;

import tsml.classifiers.distance_based.utils.collections.intervals.DoubleInterval;
import tsml.classifiers.distance_based.utils.collections.params.distribution.ClampedDistribution;

import java.util.Random;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class UniformDoubleDistribution extends ClampedDistribution<Double> {

    public UniformDoubleDistribution() {
        this(0d, 1d);
    }
    
    public UniformDoubleDistribution(final Double start, final Double end) {
        super(new DoubleInterval(start, end));
    }

    @Override
    public Double sample(Random random) {
        double start = getStart();
        double end = getEnd();
        return random.nextDouble() * Math.abs(end - start) + Math.min(start, end);
    }
}
