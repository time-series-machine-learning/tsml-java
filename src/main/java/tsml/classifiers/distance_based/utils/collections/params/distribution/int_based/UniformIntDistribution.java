package tsml.classifiers.distance_based.utils.collections.params.distribution.int_based;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class UniformIntDistribution extends ClampedIntDistribution {

    public UniformIntDistribution() {}
    
    public UniformIntDistribution(final Integer end) {
        super(end);
    }

    public UniformIntDistribution(final Integer start, final Integer end) {
        super(start, end);
    }

    @Override
    public Integer sample() {
        int end = getEnd();
        int start = getStart();
        return getRandom().nextInt(end - start + 1) + start;
    }
}
