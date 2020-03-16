package tsml.classifiers.distance_based.utils.params.distribution;

import java.util.Random;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class UniformDistribution extends RangedDistribution {

    @Override
    public Double uncheckedSample() {
        return getRandom().nextDouble() * size() + getMin();
    }
}
