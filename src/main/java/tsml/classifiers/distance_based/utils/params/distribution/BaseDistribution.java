package tsml.classifiers.distance_based.utils.params.distribution;

import java.util.Random;
import tsml.classifiers.distance_based.utils.random.BaseRandom;

/**
 * Purpose: sample values from a distribution.
 */
public abstract class BaseDistribution<A> extends BaseRandom implements Distribution<A> {

    private static final Random DEFAULT_RANDOM = new Random();

    public static Random getDefaultRandom() {
        return DEFAULT_RANDOM;
    }

    public void setDefaultRandom() {
        setRandom(getDefaultRandom());
    }

    public BaseDistribution() {
        this(DEFAULT_RANDOM);
    }

    public BaseDistribution(int seed) {
        super(seed);
    }

    public BaseDistribution(Random random) {
        super(random);
    }

    @Override
    public String toString() {
        return getClass().getSimpleName();
    }
}
