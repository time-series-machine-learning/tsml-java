package tsml.classifiers.distance_based.utils.params;

import com.google.common.collect.Range;
import java.util.Random;
import org.junit.Assert;

class RangedDistribution extends Distribution {
    private Range<Double> range = null;
    private Distribution distribution = new UniformDistribution();

    public RangedDistribution() {

    }

    public RangedDistribution(Distribution distribution) {
        setDistribution(distribution);
    }

    public RangedDistribution(Distribution distribution, Range<Double> range) {
        this(distribution);
        setRange(range);
    }

    public RangedDistribution(Distribution distribution, Random random) {
        this(distribution);
        setRandom(random);
    }

    public RangedDistribution(Distribution distribution, Range<Double> range, Random random) {
        this(distribution, range);
        setRandom(random);
    }

    public Distribution getDistribution() {
        return distribution;
    }

    public RangedDistribution setDistribution(
        final Distribution distribution) {
        Assert.assertNotNull(distribution);
        distribution.setRandom(getRandom());
        this.distribution = distribution;
        return this;
    }

    /**
     * produce a sample from the distribution
     * @return a double in the range of 0 to 1.
     */
    public final double sample() {
        final Distribution distribution = getDistribution();
        final Random origRandom = distribution.getRandom();
        distribution.setRandom(getRandom());
        double value = distribution.sample();
        distribution.setRandom(origRandom);
        if(isPdf()) {
            return value;
        } else {
            final Range<Double> range = getRange();
            return range.lowerEndpoint() + value * (range.upperEndpoint() - range.lowerEndpoint());
        }
    }

    public boolean isPdf() {
        return getRange() == null;
    }

    public Range<Double> getRange() {
        return range;
    }

    public Distribution setRange(final Range<Double> range) {
        this.range = range;
        return this;
    }
}
