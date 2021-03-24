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

import tsml.classifiers.distance_based.utils.collections.DefaultList;
import tsml.classifiers.distance_based.utils.collections.checks.Checks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class ParamSpace implements DefaultList<ParamMap> {

    public ParamSpace(final List<ParamMap> paramMaps) {
        addAll(paramMaps);
    }
    
    public ParamSpace() {
        
    }
    
    public ParamSpace(ParamMap... paramMap) {
        this(Arrays.asList(paramMap));
    }

    private final List<ParamMap> paramMaps = new ArrayList<>();

    public boolean add(ParamMap paramMap) {
        paramMaps.add(Objects.requireNonNull(paramMap));
        return true;
    }

    @Override public ParamMap get(final int i) {
        return paramMaps.get(i);
    }

    @Override public int size() {
        return paramMaps.size();
    }

    @Override public String toString() {
        return paramMaps.toString();
    }
    
    public ParamMap getSingle() {
        return Checks.requireSingle(paramMaps);
    }

}
