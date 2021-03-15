package tsml.classifiers.distance_based.utils.system.random;

import weka.core.Randomizable;

import java.util.Random;

/**
 * Purpose: allow the setting / getting of the random source.
 * <p>
 * Contributors: goastler
 */

public interface Randomised extends RandomSource, Randomizable {
    void setRandom(Random random);
    void setSeed(int seed);
    
    default void checkRandom() {
        if(getRandom() == null) {
            // random should be set by either calling setRandom or setSeed, the latter of which will automatically build a random with the specified seed
            throw new IllegalStateException("random not set");
        }
    }

    /**
     * Copy the random config (i.e. rng object and the seed) onto another instance which requires randomisation
     * @param obj
     */    
    default void copyRandomTo(Object obj) {
        copySeedTo(obj);
        if(obj instanceof Randomised) {
            // pass on the already initialised (with the seed) random
            // note that the seed has already been passed on because Randomised extends Randomizable
            ((Randomised) obj).setRandom(getRandom());
        }
        // else not a user of randomisation so don't worry about setting random / seed
    }
    
    default void copySeedTo(Object obj) {
        if(obj instanceof Randomizable) {
            ((Randomizable) obj).setSeed(getSeed());
        }
    }
}
