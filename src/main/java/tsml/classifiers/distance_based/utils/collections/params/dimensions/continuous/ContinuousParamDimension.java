package tsml.classifiers.distance_based.utils.collections.params.dimensions.continuous;

import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.ParamDimension;
import tsml.classifiers.distance_based.utils.collections.params.distribution.Distribution;

import java.util.List;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ContinuousParamDimension<A> extends ParamDimension<Distribution<A>> {

    public ContinuousParamDimension(final Distribution<A> values) {
        super(values);
    }

    public ContinuousParamDimension(final Distribution<A> values,
        final List<ParamSpace> subSpaces) {
        super(values, subSpaces);
    }
}
