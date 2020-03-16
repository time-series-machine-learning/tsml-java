package tsml.classifiers.distance_based.utils.params.tmp;

import java.util.ArrayList;
import java.util.List;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.params.distribution.BaseDistribution;
import tsml.classifiers.distance_based.utils.params.distribution.Distribution;

/**
 * class to represent a continuous set of parameter values, i.e. a range of doubles between 0 and 1, say. The
 * range is sampled randomly and the probability density function should be controlled using the distribution
 * class functions (e.g. setting the range and type of pdf).
 * @param <A>
 */
public class ContinuousParameterDimension<A> extends ParameterDimension<A> {

    private Distribution<A> distribution;

    public ContinuousParameterDimension(Distribution<A> distribution, List<ParameterSpace> subSpaces) {
        setDistribution(distribution);
        setSubSpaces(subSpaces);
    }

    public ContinuousParameterDimension(Distribution<A> distribution) {
        this(distribution, new ArrayList<>());
    }

    @Override
    public A getParameterValue(final int index) {
        return distribution.sample();
    }

    @Override
    public int getParameterDimensionSize() {
        // -1 because there's an infinite amount of values in the distribution
        return -1;
    }

    public Distribution<A> getDistribution() {
        return distribution;
    }

    public void setDistribution(final Distribution<A> distribution) {
        Assert.assertNotNull(distribution);
        this.distribution = distribution;
    }

    @Override
    public String toString() {
        return "{distribution=" + distribution + super.buildSubSpacesString() + "}";
    }

}
