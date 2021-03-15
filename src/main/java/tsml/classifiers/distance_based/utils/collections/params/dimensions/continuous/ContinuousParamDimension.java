package tsml.classifiers.distance_based.utils.collections.params.dimensions.continuous;

import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.ParamDimension;
import tsml.classifiers.distance_based.utils.collections.params.distribution.Distribution;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ContinuousParamDimension<A> extends ParamDimension<Distribution<A>> {

    public ContinuousParamDimension(final Distribution<A> values) {
        this(values, new ParamSpace());
    }

    public ContinuousParamDimension(final Distribution<A> distribution,
        final ParamSpace subSpace) {
        super(subSpace);
        this.distribution = Objects.requireNonNull(distribution);
    }
    
    private final Distribution<A> distribution;

    public Distribution<A> getDistribution() {
        return distribution;
    }

    @Override public String toString() {
        return "dist=" + distribution + super.toString();
    }
}
