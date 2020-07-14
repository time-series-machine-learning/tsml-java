package tsml.classifiers.distance_based.utils.collections.params.distribution;

import java.io.Serializable;
import java.util.Random;
import tsml.classifiers.distance_based.utils.system.random.BaseRandom;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */

public abstract class Distribution<A> extends BaseRandom implements Serializable {

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
