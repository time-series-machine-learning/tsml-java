package tsml.classifiers.distance_based.utils.collections.params.dimensions;

import java.util.List;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.distribution.Distribution;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class ContinuousParameterDimension<A> extends ParameterDimension<Distribution<A>> {

    public ContinuousParameterDimension(final Distribution<A> values) {
        super(values);
    }

    public ContinuousParameterDimension(final Distribution<A> values,
        final List<ParamSpace> subSpaces) {
        super(values, subSpaces);
    }
}
