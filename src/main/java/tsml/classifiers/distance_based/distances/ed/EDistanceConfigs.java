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
 
package tsml.classifiers.distance_based.distances.ed;

import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import weka.core.Instances;

import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

public class EDistanceConfigs {
    /**
     * param space containing ED
     *
     * @return
     */
    public static ParamSpace buildEDSpace() {
        return new ParamSpace().add(DistanceMeasure.DISTANCE_MEASURE_FLAG, newArrayList(new EDistance()));
    }

    public static class EDSpaceBuilder implements ParamSpaceBuilder {

        @Override public ParamSpace build(final Instances data) {
            return buildEDSpace();
        }
    }
}
