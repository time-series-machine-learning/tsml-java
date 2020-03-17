package tsml.classifiers.distance_based.utils.params.tmp;

import java.util.List;
import tsml.classifiers.distance_based.utils.params.distribution.Distribution;

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
        final List<ParameterSpace> subSpaces) {
        super(values, subSpaces);
    }
}
