package tsml.classifiers.distance_based.utils.collections.params.dimensions.discrete;

import org.junit.Assert;
import tsml.classifiers.distance_based.utils.classifiers.CopierUtils;
import tsml.classifiers.distance_based.utils.collections.DefaultList;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.ParamDimension;
import tsml.classifiers.distance_based.utils.collections.params.iteration.PermutationUtils;
import utilities.ArrayUtilities;
import utilities.Utilities;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class IndexedParamSpace implements DefaultList<ParamSet> {

    private ParamSpace paramSpace;
    private int size = -1;

    public IndexedParamSpace(
        final ParamSpace paramSpace) {
        setParamSpace(paramSpace);
    }

    public ParamSpace getParamSpace() {
        return paramSpace;
    }

    public void setParamSpace(
        final ParamSpace paramSpace) {
        if(paramSpace == null) throw new IllegalArgumentException("param space cannot be null");
        this.paramSpace = paramSpace;
        // calling size checks that the space contains only discrete dimensions
        size = size(getParamSpace());
    }

    @Override
    public boolean equals(final Object o) {
        if(this == o) {
            return true;
        }
        if(o == null || getClass() != o.getClass()) {
            return false;
        }
        final IndexedParamSpace space = (IndexedParamSpace) o;
        return getParamSpace().equals(space.getParamSpace());
    }

    @Override
    public int hashCode() {
        return Objects.hash(getParamSpace());
    }

    @Override
    public ParamSet get(final int index) {
        return get(getParamSpace(), index);
    }

    @Override
    public int size() {
        return size;
    }
    
    /**
     * get the value of the parameter dimension at the given index
     * @param dimension
     * @param index
     * @return
     */
    public static Object get(ParamDimension<?> dimension, int index) {
        Object values = dimension.getValues();
        if(values instanceof List<?>) {
            final List<Integer> allSizes = sizes(dimension);
            final List<Integer> indices = PermutationUtils.fromPermutation(index, allSizes);
            final Integer valueIndex = indices.remove(0);
            List<?> valuesList = (List<?>) values;
            Object value = valuesList.get(valueIndex);
            try {
                value = CopierUtils.deepCopy(value); // must copy objects otherwise every paramset
                // uses the same object reference!
            } catch(Exception e) {
                throw new IllegalArgumentException(e);
            }
            List<ParamSpace> subSpaces = dimension.getSubSpaces();
            if(!subSpaces.isEmpty()) {
                ParamSet subParamSet = get(subSpaces, indices);
                ParamSet.setParams(value, subParamSet);
            }
            return value;
        } else {
            throw new IllegalArgumentException("expected finite list of options");
        }
    }

    /**
     * get the paramset corresponding to the given indices
     * @param spaces
     * @param indices
     * @return
     */
    public static ParamSet get(final List<ParamSpace> spaces, final List<Integer> indices) {
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
    public static ParamSet get(ParamSpace space, int index) {
        final List<Integer> sizes = sizes(space);
        final List<Integer> indices = ArrayUtilities.fromPermutation(index, sizes);
        int i = 0;
        ParamSet param = new ParamSet();
        for(Map.Entry<String, List<ParamDimension<?>>> entry : space.getDimensionMap().entrySet()) {
            index = indices.get(i);
            List<ParamDimension<?>> dimensions = entry.getValue();
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
    public static Object get(List<ParamDimension<?>> dimensions, int index) {
        for(ParamDimension<?> dimension : dimensions) {
            int size = size(dimension);
            index -= size;
            if(index < 0) {
                index += size;
                return get(dimension, index);
            }
        }
        throw new IndexOutOfBoundsException();
    }

    public static int size(ParamSpace space) {
        final List<Integer> sizes = sizes(space);
        return PermutationUtils.numPermutations(sizes);
    }

    public static int size(ParamDimension<?> dimension) {
        return PermutationUtils.numPermutations(sizes(dimension));
    }

    public static List<Integer> sizesParameterSpace(List<ParamSpace> spaces) {
        return spaces.stream().map(IndexedParamSpace::size).collect(Collectors.toList());
    }

    public static int size(List<ParamDimension<?>> dimensions) {
        return PermutationUtils.numPermutations(sizesParameterDimension(dimensions));
    }

    public static List<Integer> sizesParameterDimension(List<ParamDimension<?>> dimensions) {
        return dimensions.stream().map(IndexedParamSpace::size).collect(Collectors.toList());
    }

    public static List<Integer> sizes(ParamSpace space) {
        final Map<String, List<ParamDimension<?>>> dimensionMap = space.getDimensionMap();
        final List<Integer> sizes = new ArrayList<>();
        for(final Map.Entry<String, List<ParamDimension<?>>> entry : dimensionMap.entrySet()) {
            final List<ParamDimension<?>> dimensions = entry.getValue();
            int size = 0;
            for(ParamDimension<?> dimension : dimensions) {
                size += size(dimension);
            }
            sizes.add(size);
        }
        return sizes;
    }

    public static List<Integer> sizes(ParamDimension<?> dimension) {
        List<Integer> sizes = sizesParameterSpace(dimension.getSubSpaces());
        Object values = dimension.getValues();
        if(values instanceof List<?>) {
            int size = ((List<?>) values).size();
            sizes.add(0, size);
        } else {
            throw new IllegalArgumentException("cannot handle dimension type");
        }
        return sizes;
    }

}
