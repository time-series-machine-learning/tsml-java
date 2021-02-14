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
 
package tsml.classifiers.distance_based.distances;
/**
 * @author George Oastler
 */

import java.io.Serializable;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;

public interface DistanceMeasure extends Serializable, DistanceFunction, ParamHandler {

    String DISTANCE_MEASURE_FLAG = "d";

    // the maximum distance the distance measure could produce
    double getMaxDistance();

    // whether the distance measure is symmetric (i.e. dist from inst A to inst B == dist from inst B to inst A
    boolean isSymmetric();

    double distance(final Instance a, final Instance b);

    double distance(final Instance a, final Instance b, final double limit);

    @Override
    double distance(final Instance a, final Instance b, final PerformanceStats stats)
        throws Exception;

    String getName();

    void setName(String name);

    // the fit function
    void setInstances(Instances data);


    static String getName(DistanceFunction df) {
        if(df instanceof DistanceMeasure) {
            return ((DistanceMeasure) df).getName();
        } else {
            return df.getClass().getSimpleName();
        }
    }
}
