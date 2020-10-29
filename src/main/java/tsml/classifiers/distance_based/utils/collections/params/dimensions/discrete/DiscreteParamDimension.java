package tsml.classifiers.distance_based.utils.collections.params.dimensions.discrete;

import tsml.classifiers.distance_based.utils.collections.DefaultList;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.ParamDimension;
import tsml.classifiers.distance_based.utils.collections.params.iteration.PermutationUtils;

import java.util.List;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class DiscreteParamDimension<A> extends ParamDimension<List<A>> implements DefaultList<Object> {

    public DiscreteParamDimension(final List<A> values) {
        super(values);
    }

    public DiscreteParamDimension(final List<A> values, final List<ParamSpace> subSpaces) {
        super(values, subSpaces);
    }

    public int getValuesSize() {
        return getValues().size();
    }

    @Override
    public Object get(final int index) {
        return IndexedParamSpace.get(this, index);
    }

    @Override
    public int size() {
        return PermutationUtils.numPermutations(getAllSizes());
    }

    public List<Integer> getAllSizes() {
        final List<Integer> sizes = getSubSpaceSizes();
        // put values size on the front
        sizes.add(0, getValuesSize());
        return sizes;
    }

    public List<Integer> getSubSpaceSizes() {
        return IndexedParamSpace.sizesParameterSpace(getSubSpaces());
    }

    public A getValue(int index) {
        return getValues().get(index);
    }

}
