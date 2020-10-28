package tsml.classifiers.distance_based.utils.collections.params.distribution;

import java.io.Serializable;
import java.util.Random;

public interface Distribution<A> extends Serializable {
    A sample();

    default A sample(Random random) {
        Random origRandom = getRandom();
        setRandom(random);
        A sample = sample();
        setRandom(origRandom);
        return sample;
    }

    Random getRandom();

    void setRandom(Random random);
}
