package tsml.classifiers.distance_based.utils.collections.params;

import tsml.classifiers.distance_based.utils.collections.params.dimensions.ParamDimension;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.continuous.ContinuousParamDimension;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.discrete.DiscreteParamDimension;
import tsml.classifiers.distance_based.utils.collections.params.distribution.Distribution;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

public class ParamSpace implements Serializable {

    // 1 to many mapping of param name to list of param dimensions
    private final Map<String, List<ParamDimension<?>>> dimensionMap = new LinkedHashMap<>();

    public Map<String, List<ParamDimension<?>>> getDimensionMap() {
        return dimensionMap;
    }

    @Override
    public String toString() {
        return String.valueOf(dimensionMap);
    }

    public ParamSpace addDimension(String name, ParamDimension<?> dimension) {
        getDimensionMap().computeIfAbsent(name, s -> new ArrayList<>()).add(dimension);
        return this;
    }

    public <A> ParamSpace add(String name, List<A> values) {
        final DiscreteParamDimension<A> dimension;
        if(values instanceof DiscreteParamDimension) {
            dimension = (DiscreteParamDimension<A>) values;
        } else {
            dimension = new DiscreteParamDimension<>(values);
        }
        return addDimension(name, dimension);
    }

    public <A> ParamSpace add(String name, List<A> values, List<ParamSpace> subSpaces) {
        return addDimension(name, new DiscreteParamDimension<>(values, subSpaces));
    }

    public <A> ParamSpace add(String name, List<A> values, ParamSpace subSpace) {
        return add(name, values, newArrayList(subSpace));
    }

    public <A> ParamSpace add(String name, Distribution<A> values) {
        return addDimension(name, new ContinuousParamDimension<A>(values));
    }

    public <A> ParamSpace add(String name, Distribution<A> values, List<ParamSpace> subSpaces) {
        return addDimension(name, new ContinuousParamDimension<>(values, subSpaces));
    }

    public <A> ParamSpace add(String name, Distribution<A> values, ParamSpace subSpace) {
        return add(name, values, newArrayList(subSpace));
    }

    public List<ParamDimension<?>> get(String name) {
        return getDimensionMap().get(name);
    }

}
