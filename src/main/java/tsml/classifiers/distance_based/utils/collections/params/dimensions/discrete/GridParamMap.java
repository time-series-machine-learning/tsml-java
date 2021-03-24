package tsml.classifiers.distance_based.utils.collections.params.dimensions.discrete;

import tsml.classifiers.distance_based.utils.collections.DefaultMap;
import tsml.classifiers.distance_based.utils.collections.params.ParamMap;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.ParamDimension;
import tsml.classifiers.distance_based.utils.collections.params.iteration.PermutationUtils;
import tsml.classifiers.distance_based.utils.collections.views.WrappedList;

import java.util.*;

public class GridParamMap implements DefaultMap<String, List<ParamDimension<?>>> {

    public GridParamMap(final ParamMap paramMap) {
        this.paramMap = Objects.requireNonNull(paramMap);
        if(!isDiscrete(paramMap)) {
            throw new IllegalArgumentException("param space not discrete");
        }
    }

    private final ParamMap paramMap;
    
    public int size() {
        return PermutationUtils.numPermutations(getBinSizes());
    }

    @Override public List<ParamDimension<?>> get(final Object name) {
        return paramMap.get(name);
    }
    
    public ParamSet get(List<Integer> indices) {
        final ParamSet paramSet = new ParamSet();
        int i = 0;
        // loop through dimensions
        for(Map.Entry<String, List<ParamDimension<?>>> entry : entrySet()) {
            // get the dimension
            final int subIndex = indices.get(i);
            // find which dimension the index lands in
            final List<ParamDimension<?>> paramDimensions = entry.getValue();
            final int paramDimensionIndex =
                    PermutationUtils.spannedIndexOf(new WrappedList<>(paramDimensions, paramDimension -> ((DiscreteParamDimension<?>) paramDimension).size()), subIndex);
            final ParamDimension<?> paramDimension = paramDimensions.get(paramDimensionIndex);
            // get the value from that dimension at the given index
            final Object value = ((DiscreteParamDimension<?>) paramDimension).get(subIndex);
            // add to param set
            paramSet.add(entry.getKey(), value);
            i++;
        }
        return paramSet;
    }

    public List<Integer> getBinSizes() {
        final List<Integer> sizes = new ArrayList<>();
        for(final Map.Entry<String, List<ParamDimension<?>>> entry : paramMap.entrySet()) {
            final List<ParamDimension<?>> dimensions = entry.getValue();
            int size = 0;
            for(ParamDimension<?> dimension : dimensions) {
                if(dimension instanceof DiscreteParamDimension<?>) {
                    size += ((DiscreteParamDimension<?>) dimension).size();
                } else {
                    throw new IllegalArgumentException("dimension not discrete: " + dimension);
                }
            }
            sizes.add(size);
        }
        return sizes;
    }

    @Override public boolean containsKey(final Object o) {
        return paramMap.containsKey(o);
    }

    @Override public boolean containsValue(final Object o) {
        return paramMap.containsValue(o);
    }

    @Override public Set<String> keySet() {
        return paramMap.keySet();
    }

    @Override public Collection<List<ParamDimension<?>>> values() {
        return paramMap.values();
    }

    @Override public Set<Entry<String, List<ParamDimension<?>>>> entrySet() {
        return paramMap.entrySet();
    }
    
    
    public static boolean isDiscrete(ParamMap paramMap) {
        for(final Map.Entry<String, List<ParamDimension<?>>> entry : paramMap.entrySet()) {
            for(ParamDimension<?> dimension : entry.getValue()) {
                if(!(dimension instanceof DiscreteParamDimension<?>)) {
                    return false;
                }
                if(!GridParamSpace.isDiscrete(dimension.getSubSpace())) {
                    return false;
                }
            }
        }
        return true;
    }
}
