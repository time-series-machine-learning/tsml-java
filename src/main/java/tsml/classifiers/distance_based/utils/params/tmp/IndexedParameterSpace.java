package tsml.classifiers.distance_based.utils.params.tmp;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.IndexedCollection;
import tsml.classifiers.distance_based.utils.params.ParamHandler;
import tsml.classifiers.distance_based.utils.params.ParamSet;
import utilities.ArrayUtilities;
import utilities.Utilities;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class IndexedParameterSpace implements IndexedCollection<ParamSet> {

    private ParameterSpace parameterSpace;

    public IndexedParameterSpace(
        final ParameterSpace parameterSpace) {
        setParameterSpace(parameterSpace);
    }

    public ParameterSpace getParameterSpace() {
        return parameterSpace;
    }

    public IndexedParameterSpace setParameterSpace(
        final ParameterSpace parameterSpace) {
        Assert.assertNotNull(parameterSpace);
        this.parameterSpace = parameterSpace;
        return this;
    }

    @Override
    public ParamSet get(final int index) {
        return get(getParameterSpace(), index);
    }

    @Override
    public int size() {
        return size(getParameterSpace());
    }


    /**
     * get the value of the parameter dimension at the given index
     * @param dimension
     * @param index
     * @return
     */
    public static Object get(ParameterDimension<?> dimension, int index) {
        if(dimension instanceof DiscreteParameterDimension<?>) { // todo make this into an interface for looser coupling
            final List<Integer> allSizes = ((DiscreteParameterDimension<?>) dimension).getAllSizes();
            final List<Integer> indices = Permutations.fromPermutation(index, allSizes);
            final Integer valueIndex = indices.remove(0);
            Object value = ((DiscreteParameterDimension<?>) dimension).getValue(valueIndex);
            try {
                value = Utilities.deepCopy(value); // must copy objects otherwise every paramset
                // uses the same object reference!
            } catch(Exception e) {
                throw new IllegalStateException("cannot copy value");
            }
            ParamSet subParamSet = get(dimension.getSubSpaces(), indices);
            if(value instanceof ParamHandler) {
                ((ParamHandler) value).setParams(subParamSet);
            } else {
                throw new IllegalStateException("{" + value.toString() + "} isn't an instance of ParamHandler, cannot "
                    + "set params");
            }
            return value;
        } else {
            throw new IllegalArgumentException();
        }
    }

    /**
     * get the paramset corresponding to the given indices
     * @param spaces
     * @param indices
     * @return
     */
    public static ParamSet get(final List<ParameterSpace> spaces, final List<Integer> indices) {
        Assert.assertEquals(spaces.size(), indices.size());
        final ParamSet overallParamSet = new ParamSet();
        for(int i = 0; i < spaces.size(); i++) {
            final ParamSet paramSet = get(spaces.get(i), indices.get(i));
            overallParamSet.addAll(paramSet);
        }
        return overallParamSet;
    }

    /**
     * get the paramset given the permutation index
     * @param space
     * @param index
     * @return
     */
    public static ParamSet get(ParameterSpace space, int index) {
        final List<Integer> sizes = sizes(space);
        final List<Integer> indices = ArrayUtilities.fromPermutation(index, sizes);
        int i = 0;
        ParamSet param = new ParamSet();
        for(Map.Entry<String, List<ParameterDimension<?>>> entry : space.getDimensionMap().entrySet()) {
            index = indices.get(i);
            List<ParameterDimension<?>> dimensions = entry.getValue();
            Object value = get(dimensions, index);
            param.add(entry.getKey(), value);
            i++;
        }
        return param;
    }

    /**
     * get the object at the given index across several dimensions
     * @param dimensions
     * @param index
     * @return
     */
    public static Object get(List<ParameterDimension<?>> dimensions, int index) {
        for(ParameterDimension<?> dimension : dimensions) {
            int size = size(dimension);
            index -= size;
            if(index < 0) {
                index += size;
                return get(dimension, index);
            }
        }
        throw new IndexOutOfBoundsException();
    }

    public static int size(ParameterSpace space) {
        final List<Integer> sizes = sizes(space);
        return Permutations.numPermutations(sizes);
    }

    public static int size(ParameterDimension<?> dimension) {
        if(dimension instanceof DiscreteParameterDimension<?>) {
            return Permutations.numPermutations(sizes(dimension));
        } else {
            throw new IllegalArgumentException("dimension not instance of DiscreteParameterDimension");
        }
    }

    public static List<Integer> sizesParameterSpace(List<ParameterSpace> spaces) {
        return Utilities.convert(spaces, IndexedParameterSpace::size);
    }

    public static int size(List<ParameterDimension<?>> dimensions) {
        return Permutations.numPermutations(sizesParameterDimension(dimensions));
    }

    public static List<Integer> sizesParameterDimension(List<ParameterDimension<?>> dimensions) {
        return Utilities.convert(dimensions, IndexedParameterSpace::size);
    }

    public static List<Integer> sizes(ParameterSpace space) {
        final Map<String, List<ParameterDimension<?>>> dimensionMap = space.getDimensionMap();
        final List<Integer> sizes = new ArrayList<>();
        for(final Map.Entry<String, List<ParameterDimension<?>>> entry : dimensionMap.entrySet()) {
            final List<ParameterDimension<?>> dimensions = entry.getValue();
            int size = 0;
            for(ParameterDimension<?> dimension : dimensions) {
                size += size(dimension);
            }
            sizes.add(size);
        }
        return sizes;
    }

    public static List<Integer> sizes(ParameterDimension<?> dimension) {
        List<Integer> sizes = sizesParameterSpace(dimension.getSubSpaces());
        if(dimension instanceof DiscreteParameterDimension<?>) {
            int size = ((DiscreteParameterDimension<?>) dimension).getValues().size();
            sizes.add(0, size);
        } else {
            throw new IllegalArgumentException("cannot handle dimension type");
        }
        return sizes;
    }
}
