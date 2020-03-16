package tsml.classifiers.distance_based.utils;

import java.util.Random;
import org.junit.Assert;
import tsml.classifiers.distance_based.proximity.RandomSource;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class BaseRandom implements RandomSource {

    private Random random = new Random();

    public BaseRandom() {

    }

    public BaseRandom(final Random random) {
        this.random = random;
    }

    @Override
    public void setRandom(final Random random) {
        Assert.assertNotNull(random);
        this.random = random;
    }

    @Override
    public Random getRandom() {
        return random;
    }
}
