package tsml.classifiers.distance_based.utils.params.distribution;

import java.util.Random;
import tsml.classifiers.distance_based.utils.random.BaseRandom;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */

public abstract class Distribution<A> extends BaseRandom {

    public Distribution(final Random random) {
        super(random);
    }

    public Distribution() {
        super(null);
    }

    public abstract A sample();

    public A sample(Random random) {
        Random origRandom = getRandom();
        setRandom(random);
        A sample = sample();
        setRandom(origRandom);
        return sample;
    }
}
