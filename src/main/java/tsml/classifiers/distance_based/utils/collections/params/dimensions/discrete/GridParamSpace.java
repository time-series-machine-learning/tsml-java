package tsml.classifiers.distance_based.utils.collections.params.dimensions.discrete;

import tsml.classifiers.distance_based.utils.collections.DefaultList;
import tsml.classifiers.distance_based.utils.collections.params.ParamMap;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.iteration.PermutationUtils;
import utilities.ArrayUtilities;

import java.util.*;

/**
 * Purpose: given a paramspace which contains only discrete dimensions, index each permutation of the parameters to appear as a list of paramsets.
 * <p>
 * Contributors: goastler
 */
public class GridParamSpace implements DefaultList<ParamSet> {

    private final ParamSpace paramSpace;

    public GridParamSpace(final ParamSpace paramSpace) {
        this.paramSpace = Objects.requireNonNull(paramSpace);
    }

    public ParamSpace getParamSpace() {
        return paramSpace;
    }

    @Override
    public boolean equals(final Object o) {
        if(this == o) {
            return true;
        }
        if(o == null || getClass() != o.getClass()) {
            return false;
        }
        final GridParamSpace space = (GridParamSpace) o;
        return getParamSpace().equals(space.getParamSpace());
    }

    @Override
    public int hashCode() {
        return Objects.hash(getParamSpace());
    }
    
    /**
     * get the paramset given the permutation index
     * @param index
     * @return
     */
    @Override
    public ParamSet get(final int index) {
        final List<Integer> a = new ArrayList<>();
        for(ParamMap map : paramSpace) {
            Integer size = new GridParamMap(map).size();
            a.add(size);
        }
        final int paramMapIndex = PermutationUtils.spannedIndexOf(a, index);
        final ParamMap paramMap = paramSpace.get(paramMapIndex);
        // now we have the parammap, we need to get the paramset from that using the index
        // discretise the param map
        final GridParamMap gridParamMap = new GridParamMap(paramMap);
        // find the bin sizes for each dimension inside the param map
        final List<Integer> sizes = gridParamMap.getBinSizes();
        // get the indices for each dimension given the index and bin sizes of each dimension
        final List<Integer> indices = ArrayUtilities.fromPermutation(index, sizes);
        return gridParamMap.get(indices);
    }

    @Override
    public int size() {
        return getParamMapSizes().stream().mapToInt(i -> i).sum();
    }
    
    public List<Integer> getParamMapSizes() {
        List<Integer> list = new ArrayList<>();
        for(ParamMap paramMap : paramSpace) {
            Integer size = new GridParamMap(paramMap).size();
            list.add(size);
        }
        return list;
    }
    
    public static boolean isDiscrete(ParamSpace paramSpace) {
        for(ParamMap paramMap : paramSpace) {
            if(!GridParamMap.isDiscrete(paramMap)) {
                return false;
            }
        }
        return true;
    }
    
    
    
    
//    
//    /**
//     * get the value of the parameter dimension at the given index
//     * @param dimension
//     * @param index
//     * @return
//     */
//    private Object getValue(ParamDimension<?> dimension, int index) {
//        if(dimension instanceof DiscreteParamDimension<?>) {
//            
//        } else {
//            throw new IllegalArgumentException("expected finite list of options");
//        }
//    }
//
//    /**
//     * get the paramset corresponding to the given indices
//     * @param spaces
//     * @param indices
//     * @return
//     */
//    public ParamSet get(final List<ParamSpace> spaces, final List<Integer> indices) {
//        Assert.assertEquals(spaces.size(), indices.size());
//        final ParamSet overallParamSet = new ParamSet();
//        for(int i = 0; i < spaces.size(); i++) {
//            final ParamSet paramSet = get(spaces.get(i), indices.get(i));
//            overallParamSet.addAll(paramSet);
//        }
//        return overallParamSet;
//    }
//
//    /**
//     * get the object at the given index across several dimensions
//     * @param dimensions
//     * @param index
//     * @return
//     */
//    private Object get(List<ParamDimension<?>> dimensions, int index) {
//        for(ParamDimension<?> dimension : dimensions) {
//            int size = size(dimension);
//            index -= size;
//            if(index < 0) {
//                index += size;
//                return get(dimension, index);
//            }
//        }
//        throw new IndexOutOfBoundsException();
//    }
//
//    public static int size(ParamDimension<?> dimension) {
//        return PermutationUtils.numPermutations(getBinSizes(dimension));
//    }
//
//    public static List<Integer> sizesParameterSpace(List<ParamSpace> spaces) {
//        return spaces.stream().map(IndexedParamSpace::size).collect(Collectors.toList());
//    }
//
//    public static int size(List<ParamDimension<?>> dimensions) {
//        return PermutationUtils.numPermutations(sizesParameterDimension(dimensions));
//    }
//
//    public static List<Integer> sizesParameterDimension(List<ParamDimension<?>> dimensions) {
//        return dimensions.stream().map(IndexedParamSpace::size).collect(Collectors.toList());
//    }
//
//    public static List<Integer> getBinSizes(ParamDimension<?> dimension) {
//        List<Integer> sizes = sizesParameterSpace(dimension.getSubSpace());
//        Object values = dimension.getValues();
//        if(values instanceof List<?>) {
//            int size = ((List<?>) values).size();
//            sizes.add(0, size);
//        } else {
//            throw new IllegalArgumentException("cannot handle dimension type");
//        }
//        return sizes;
//    }

}
