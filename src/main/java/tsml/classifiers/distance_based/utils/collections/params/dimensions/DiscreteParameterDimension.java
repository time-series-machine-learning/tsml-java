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
