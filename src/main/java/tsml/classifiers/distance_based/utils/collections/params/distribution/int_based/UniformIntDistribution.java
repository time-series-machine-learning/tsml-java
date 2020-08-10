package tsml.classifiers.distance_based.utils.collections.params.distribution.int_based;

import org.junit.Assert;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class UniformIntDistribution extends IntDistribution {

    public UniformIntDistribution(final Integer max) {
        this(0, max);
    }

    public UniformIntDistribution(final Integer min, final Integer max) {
        super(min, max);
    }

    @Override
    protected void checkSetMinAndMax(final Integer min, final Integer max) {
        Assert.assertNotNull(min);
        Assert.assertNotNull(max);
        Assert.assertTrue(min <= max);
    }

    @Override
    public Integer sample() {
        int max = getMax();
        int min = getMin();
        return getRandom().nextInt(max - min + 1) + min;
    }
}
