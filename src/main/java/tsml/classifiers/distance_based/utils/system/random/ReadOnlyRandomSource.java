package tsml.classifiers.distance_based.utils.system.random;

import java.io.Serializable;
import java.util.Random;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */

public interface ReadOnlyRandomSource extends Serializable {
    Random getRandom();
}
