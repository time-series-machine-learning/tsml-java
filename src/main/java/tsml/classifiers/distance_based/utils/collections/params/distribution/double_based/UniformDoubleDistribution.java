package tsml.classifiers.distance_based.utils.collections.params.distribution.double_based;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class UniformDoubleDistribution extends ClampedDoubleDistribution {

    public UniformDoubleDistribution() {
        super();
    }
    
    public UniformDoubleDistribution(Double end) {
        super(end);
    }

    public UniformDoubleDistribution(final Double start, final Double end) {
        super(start, end);
    }

    @Override
    public Double sample() {
        double start = getStart();
        double end = getEnd();
        return getRandom().nextDouble() * (end - start) + start;
    }
}
