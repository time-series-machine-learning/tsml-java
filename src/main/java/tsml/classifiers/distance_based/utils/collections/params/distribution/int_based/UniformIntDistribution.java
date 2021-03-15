package tsml.classifiers.distance_based.utils.collections.params.distribution.int_based;

import tsml.classifiers.distance_based.utils.collections.intervals.IntInterval;
import tsml.classifiers.distance_based.utils.collections.params.distribution.ClampedDistribution;

import java.util.Random;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class UniformIntDistribution extends ClampedDistribution<Integer> {

    public UniformIntDistribution() {
        this(0, 1);
    }

    public UniformIntDistribution(final Integer start, final Integer end) {
        super(new IntInterval(start, end));
    }
    
    public Integer sample(Random random) {
        int end = getEnd();
        int start = getStart();
        return random.nextInt(Math.abs(end - start) + 1) + Math.min(start, end);
    }
}
