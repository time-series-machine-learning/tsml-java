package tsml.classifiers.distance_based.utils.collections.params.distribution.int_based;

import tsml.classifiers.distance_based.utils.collections.params.distribution.NumericDistribution;

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
