package tsml.classifiers.distance_based.utils.params.distribution;

import java.util.Random;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.BaseRandom;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public abstract class RangedDistribution extends BaseRandom implements Distribution<Double> {
    private double min;
    private double max;

    public RangedDistribution() {
        this(0, 1);
    }

    public RangedDistribution(final Random random) {
        this(0, 1, random);
    }

    public RangedDistribution(double min, double max) {
        setMinAndMax(min, max);
    }

    public RangedDistribution(double min, double max, Random random) {
        this(min, max);
        setRandom(random);
    }

    public double getMin() {
        return min;
    }

    public void setMin(final double min) {
        Assert.assertTrue(min <= max);
        this.min = min;
    }

    public double getMax() {
        return max;
    }

    public void setMax(final double max) {
        Assert.assertTrue(min <= max);
        this.max = max;
    }

    public void setMinAndMax(final double min, final double max) {
        Assert.assertTrue(min <= max);
        this.min = min;
        this.max = max;
    }

    public double size() {
        return max - min;
    }

    protected abstract Double uncheckedSample();

    public Double sample() {
        Double value = uncheckedSample();
        Assert.assertTrue(value <= getMax());
        Assert.assertTrue(value >= getMin());
        return value;
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "{" +
            "min=" + min +
            ", max=" + max +
            '}';
    }
}
