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
 
package tsml.classifiers.distance_based.utils.collections.params.dimensions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import org.junit.Assert;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;

/**
 * hold the parameter dimension. In here should be a method of retreiving values for the given parameter along
 * with sub parameter spaces to explore
 * @param <A>
 */
public abstract class ParameterDimension<A> implements Serializable {

    // list of subspaces to explore
    private List<ParamSpace> subSpaces;
    // some type holding the values for this dimension. This would usually be a list.
    private A values;

    public ParameterDimension(A values) {
        this(values, new ArrayList<>());
    }

    public ParameterDimension(A values, List<ParamSpace> subSpaces) {
        setValues(values);
        setSubSpaces(subSpaces);
    }

    @Override
    public String toString() {
        String subSpacesString = "";
        if(!getSubSpaces().isEmpty()) {
            subSpacesString = ", subSpaces=" + subSpaces;
        }
        return "{" +
            "values=" + values +
            subSpacesString +
            '}';
    }

    public List<ParamSpace> getSubSpaces() {
        return subSpaces;
    }

    public void setSubSpaces(final List<ParamSpace> subSpaces) {
        Assert.assertNotNull(subSpaces);
        this.subSpaces = subSpaces;
    }


    public A getValues() {
        return values;
    }

    public ParameterDimension<A> setValues(final A values) {
        Assert.assertNotNull(values);
        this.values = values;
        return this;
    }

    public void addSubSpace(ParamSpace subSpace) {
        getSubSpaces().add(subSpace);
    }

    public void addValue(A values) {
        this.values = values;
    }

}
