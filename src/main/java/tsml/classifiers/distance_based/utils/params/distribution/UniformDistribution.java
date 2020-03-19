package tsml.classifiers.distance_based.utils.params.distribution;

import java.util.Random;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class UniformDistribution extends RangedDistribution {

    public UniformDistribution() {
    }

    public UniformDistribution(final Random random) {
        super(random);
    }

    public UniformDistribution(final double min, final double max) {
        super(min, max);
    }

    public UniformDistribution(final double min, final double max, final Random random) {
        super(min, max, random);
    }

    @Override
    public Double uncheckedSample() {
        return getRandom().nextDouble() * size() + getMin();
    }
}
