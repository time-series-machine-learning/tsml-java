package tsml.classifiers.distance_based.utils.system.random;


import java.io.Serializable;
import java.util.Random;

/**
 * Purpose: allow the setting / getting of the random source.
 * <p>
 * Contributors: goastler
 */

public interface RandomSource extends Serializable {
    Random getRandom();
}
