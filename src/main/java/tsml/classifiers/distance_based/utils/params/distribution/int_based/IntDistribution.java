package tsml.classifiers.distance_based.utils.params.distribution.int_based;

import java.util.Random;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.params.distribution.NumericDistribution;
import tsml.classifiers.distance_based.utils.random.BaseRandom;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public abstract class IntDistribution extends NumericDistribution<Integer> {

    public IntDistribution(final Integer min, final Integer max) {
        super(min, max);
    }

    public abstract Integer sample();

}
