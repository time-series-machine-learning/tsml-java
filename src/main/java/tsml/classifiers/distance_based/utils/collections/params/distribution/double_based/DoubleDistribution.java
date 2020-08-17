package tsml.classifiers.distance_based.utils.collections.params.distribution.double_based;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.params.distribution.NumericDistribution;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public abstract class DoubleDistribution extends NumericDistribution<Double> {

    public DoubleDistribution(final double min, final double max) {
        super(min, max);
    }

    @Override
    protected void checkSetMinAndMax(Double min, Double max) {
        Assert.assertNotNull(min);
        Assert.assertNotNull(max);
        Assert.assertTrue(min <= max);
    }

    public abstract Double sample();

}
