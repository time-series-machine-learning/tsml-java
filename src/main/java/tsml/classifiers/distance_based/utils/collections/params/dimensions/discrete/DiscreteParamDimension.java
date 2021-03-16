package tsml.classifiers.distance_based.utils.collections.params.dimensions.discrete;

import tsml.classifiers.distance_based.utils.system.copy.CopierUtils;
import tsml.classifiers.distance_based.utils.collections.DefaultList;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.ParamDimension;
import tsml.classifiers.distance_based.utils.collections.params.iteration.PermutationUtils;

import java.util.List;
import java.util.Objects;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class DiscreteParamDimension<A> extends ParamDimension<List<A>> implements DefaultList<Object> {

    private List<A> values;
    
    public DiscreteParamDimension(final List<A> values) {
        this(values, new ParamSpace());
    }

    public DiscreteParamDimension(final List<A> values, ParamSpace subSpaces) {
        super(subSpaces);
        setValues(values);
    }

    public int getNumValues() {
        return getValues().size();
    }

    @Override
    public Object get(final int index) {
        final List<Integer> binSizes = getBinSizes();
        final List<Integer> indices = PermutationUtils.fromPermutation(index, binSizes);
        final Integer valueIndex = indices.remove(0);
        // must copy objects otherwise every paramset uses the same object reference!
        final Object value = CopierUtils.deepCopy(values.get(valueIndex));
        final ParamSpace subSpace = getSubSpace();
        if(!subSpace.isEmpty()) {
            // remove the first size as used
            binSizes.remove(0);
            int subIndex = PermutationUtils.toPermutation(indices, binSizes);
            ParamSet subParamSet = new GridParamSpace(subSpace).get(subIndex);
            subParamSet.applyTo(value);
        }
        return value;
    }

    @Override
    public int size() {
        return PermutationUtils.numPermutations(getBinSizes());
    }
    
    public List<Integer> getBinSizes() {
        final List<Integer> sizes = getSubSpaceBinSizes();
        // put values size on the front
        sizes.add(0, getNumValues());
        return sizes;
    }
    
    public List<Integer> getSubSpaceBinSizes() {
        return new GridParamSpace(getSubSpace()).getParamMapSizes();
    }

    public List<A> getValues() {
        return values;
    }

    public void setValues(final List<A> values) {
        this.values = Objects.requireNonNull(values);
    }

    @Override public String toString() {
        return "{" +
                       "values=" + values + super.toString() +
                       '}';
    }
}
