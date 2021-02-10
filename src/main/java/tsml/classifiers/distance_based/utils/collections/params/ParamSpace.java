/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.classifiers.distance_based.utils.collections.params;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import tsml.classifiers.distance_based.utils.collections.params.dimensions.ContinuousParameterDimension;
import tsml.classifiers.distance_based.utils.collections.params.distribution.Distribution;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.DiscreteParameterDimension;
import tsml.classifiers.distance_based.utils.collections.params.dimensions.ParameterDimension;

public class ParamSpace implements Serializable {

    // 1 to many mapping of param name to list of param dimensions
    private Map<String, List<ParameterDimension<?>>> dimensionMap = new LinkedHashMap<>();

    public Map<String, List<ParameterDimension<?>>> getDimensionMap() {
        return dimensionMap;
    }

    @Override
    public String toString() {
        return String.valueOf(dimensionMap);
    }

    public ParamSpace add(String name, ParameterDimension<?> dimension) {
        getDimensionMap().computeIfAbsent(name, s -> new ArrayList<>()).add(dimension);
        return this;
    }

    public <A> ParamSpace add(String name, List<A> values) {
        return add(name, new DiscreteParameterDimension<A>(values));
    }

    public <A> ParamSpace add(String name, List<A> values, List<ParamSpace> subSpaces) {
        return add(name, new DiscreteParameterDimension<>(values, subSpaces));
    }

    public <A> ParamSpace add(String name, List<A> values, ParamSpace subSpace) {
        List<ParamSpace> list = new ArrayList<>(Collections.singletonList(subSpace));
        return add(name, values, list);
    }

    public <A> ParamSpace add(String name, Distribution<A> values) {
        return add(name, new ContinuousParameterDimension<A>(values));
    }

    public <A> ParamSpace add(String name, Distribution<A> values, List<ParamSpace> subSpaces) {
        return add(name, new ContinuousParameterDimension<>(values, subSpaces));
    }

    public <A> ParamSpace add(String name, Distribution<A> values, ParamSpace subSpace) {
        List<ParamSpace> list = new ArrayList<>(Collections.singletonList(subSpace));
        return add(name, values, list);
    }

    public List<ParameterDimension<?>> get(String name) {
        return getDimensionMap().get(name);
    }

}
