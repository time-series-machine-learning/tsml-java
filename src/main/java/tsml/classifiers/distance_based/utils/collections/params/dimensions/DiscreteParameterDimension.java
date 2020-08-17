package tsml.classifiers.distance_based.utils.collections.params.dimensions;

import java.util.List;
import tsml.classifiers.distance_based.utils.collections.IndexedCollection;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.iteration.PermutationUtils;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class DiscreteParameterDimension<A> extends ParameterDimension<List<A>> implements IndexedCollection<Object> {

    public DiscreteParameterDimension(final List<A> values) {
        super(values);
    }

    public DiscreteParameterDimension(final List<A> values,
        final List<ParamSpace> subSpaces) {
        super(values, subSpaces);
    }

    public int getValuesSize() {
        return getValues().size();
    }

    @Override
    public Object get(final int index) {
        return IndexedParameterSpace.get(this, index);
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
        return IndexedParameterSpace.sizesParameterSpace(getSubSpaces());
    }

    public A getValue(int index) {
        return getValues().get(index);
    }

}
