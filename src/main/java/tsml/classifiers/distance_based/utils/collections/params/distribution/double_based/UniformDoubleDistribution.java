package tsml.classifiers.distance_based.utils.collections.params.distribution.double_based;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class UniformDoubleDistribution extends DoubleDistribution {

    public UniformDoubleDistribution() {
        this(0d, 1d);
    }

    public UniformDoubleDistribution(final Double min, final Double max) {
        super(min, max);
    }

    @Override
    public Double sample() {
        double min = getMin();
        double max = getMax();
        return getRandom().nextDouble() * (max - min) + min;
    }
}
