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

import tsml.classifiers.distance_based.utils.strings.StrUtils;

import java.util.List;
import java.util.function.Consumer;

public class ParamHandlerUtils {

    public static <A> boolean setParam(ParamSet paramSet, String name, Consumer<A> setter,
                                       Class<A> clazz) throws Exception {
        final List<Object> list = paramSet.get(name);
        if(list == null) {
            return false;
        }
        for(Object value : list) {
            if(value instanceof String) {
                value = StrUtils.fromOptionValue((String) value, clazz);
            }
            setter.accept((A) value);
        }
        return true;
    }
}
