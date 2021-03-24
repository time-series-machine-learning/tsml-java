package tsml.classifiers.distance_based.utils.collections.params;

import tsml.classifiers.distance_based.utils.collections.DefaultMap;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.ParamDimension;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.continuous.ContinuousParamDimension;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.discrete.DiscreteParamDimension;
import tsml.classifiers.distance_based.utils.collections.params.distribution.Distribution;

import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

public class ParamMap implements Serializable, DefaultMap<String, List<ParamDimension<?>>> {

    // 1 to many mapping of param name to list of param dimensions
    private final Map<String, List<ParamDimension<?>>> dimensionMap = new LinkedHashMap<>();

    @Override
    public String toString() {
        return String.valueOf(dimensionMap);
    }

    @Override public List<ParamDimension<?>> get(final Object name) {
        return dimensionMap.get(name);
    }

    @Override public int size() {
        return dimensionMap.size();
    }

    @Override public Set<String> keySet() {
        return Collections.unmodifiableMap(dimensionMap).keySet();
    }

    @Override public Collection<List<ParamDimension<?>>> values() {
        return Collections.unmodifiableMap(dimensionMap).values();
    }

    @Override public Set<Entry<String, List<ParamDimension<?>>>> entrySet() {
        return Collections.unmodifiableMap(dimensionMap).entrySet();
    }

    public ParamMap addDimension(String name, ParamDimension<?> dimension) {
        dimensionMap.computeIfAbsent(name, s -> new ArrayList<>()).add(dimension);
        return this;
    }

    public <A> ParamMap add(String name, double[] values) {
        return add(name, Arrays.stream(values).boxed().collect(Collectors.toList()));
    }

    public <A> ParamMap add(String name, int[] values) {
        return add(name, Arrays.stream(values).boxed().collect(Collectors.toList()));
    }

    public <A> ParamMap add(String name, long[] values) {
        return add(name, Arrays.stream(values).boxed().collect(Collectors.toList()));
    }
    
    public <A> ParamMap add(String name, List<A> values) {
        final DiscreteParamDimension<A> dimension;
        if(values instanceof DiscreteParamDimension) {
            dimension = (DiscreteParamDimension<A>) values;
        } else {
            dimension = new DiscreteParamDimension<>(values);
        }
        return addDimension(name, dimension);
    }

    public <A> ParamMap add(String name, List<A> values, ParamSpace subSpaces) {
        return addDimension(name, new DiscreteParamDimension<>(values, subSpaces));
    }

    public <A> ParamMap add(String name, List<A> values, ParamMap subMap) {
        return add(name, values, new ParamSpace(subMap));
    }

    public <A> ParamMap add(String name, Distribution<A> values) {
        return addDimension(name, new ContinuousParamDimension<A>(values));
    }

    public <A> ParamMap add(String name, Distribution<A> values, ParamSpace subSpaces) {
        return addDimension(name, new ContinuousParamDimension<>(values, subSpaces));
    }

    public <A> ParamMap add(String name, Distribution<A> values, ParamMap subMap) {
        return add(name, values, new ParamSpace(subMap));
    }

}
