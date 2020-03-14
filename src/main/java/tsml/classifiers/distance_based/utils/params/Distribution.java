package tsml.classifiers.distance_based.utils.params;

import java.util.Random;
import tsml.classifiers.distance_based.proximity.RandomSource;

public abstract class Distribution<A> implements RandomSource {

    private Random random = null;

    public Distribution() {

    }

    public Distribution(Random random) {
        setRandom(random);
    }

    public abstract A sample();

    @Override
    public Random getRandom() {
        return random;
    }

    @Override
    public void setRandom(Random random) {
        this.random = random;
    }

}
