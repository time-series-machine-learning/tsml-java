package tsml.classifiers.distance_based.utils.system.random;


import java.util.Random;

/**
 * Purpose: allow the setting / getting of the random source.
 * <p>
 * Contributors: goastler
 */

public interface RandomSource extends ReadOnlyRandomSource {
    void setRandom(Random random);
}
