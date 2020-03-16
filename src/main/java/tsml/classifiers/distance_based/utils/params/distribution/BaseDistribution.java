package tsml.classifiers.distance_based.utils.params.distribution;

import java.util.Random;
import tsml.classifiers.distance_based.utils.BaseRandom;

/**
 * Purpose: sample values from a distribution.
 */
public abstract class BaseDistribution<A> extends BaseRandom implements Distribution<A> {

    public BaseDistribution() {}

    public BaseDistribution(Random random) {
        super(random);
    }

    @Override
    public String toString() {
        return getClass().getSimpleName();
    }
}
