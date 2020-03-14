package tsml.classifiers.distance_based.utils.params;

import java.io.Serializable;
import java.util.Random;
import org.junit.Assert;

class DiscreteDistribution {

    public Discretiser getDiscretiser() {
        return discretiser;
    }

    public DiscreteDistribution setDiscretiser(
        final Discretiser discretiser) {
        Assert.assertNotNull(discretiser); // todo do asserts work in production? There's no <if running in test
        // mode>, right?!
        this.discretiser = discretiser;
        return this;
    }

    public interface Discretiser extends Serializable {
        double discretise(double value);
    }

    private Distribution distribution = new UniformDistribution();
    private Random random = null;
    private Discretiser discretiser = Math::round;

    public DiscreteDistribution() {

    }

    public DiscreteDistribution(Distribution distribution, Discretiser discretiser) {
        setDiscretiser(discretiser);
        setDistribution(distribution);
    }

    public DiscreteDistribution(Distribution distribution, Discretiser discretiser, Random random) {
        this(distribution, discretiser);
        setRandom(random);
    }

    public int sample() {
        final Distribution distribution = getDistribution();
        final Random origRandom = distribution.getRandom();
        distribution.setRandom(getRandom());
        final double value = distribution.sample();
        distribution.setRandom(origRandom);
        double discretisedValue = getDiscretiser().discretise(value);
        return (int) discretisedValue;
    }

    public Distribution getDistribution() {
        return distribution;
    }

    public void setDistribution(
        final Distribution distribution) {
        Assert.assertNotNull(distribution);
        this.distribution = distribution;
    }

    public Random getRandom() {
        return random;
    }

    public void setRandom(final Random random) {
        this.random = random;
    }
}
