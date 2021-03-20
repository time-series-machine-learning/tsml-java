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
 
package tsml.classifiers.distance_based.distances.transformed;
/*

Purpose: // todo - docs - type the purpose of the code here

Contributors: goastler
    
*/

import tsml.classifiers.distance_based.distances.DistanceMeasure;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.transformers.TrainableTransformer;
import tsml.transformers.Transformer;
import weka.core.DistanceFunction;

import java.util.Collections;
import java.util.List;

public interface TransformDistanceMeasure extends DistanceMeasure {
    DistanceMeasure getDistanceMeasure();
    Transformer getTransformer();
    void setDistanceMeasure(DistanceMeasure distanceMeasure);
    void setTransformer(Transformer transformer);
    void setName(String name);
    
    default void buildDistanceMeasure(TimeSeriesInstances data, boolean transformed) {
        // transform the data if not already
        if(!transformed) {
            final Transformer transformer = getTransformer();
            if(transformer instanceof TrainableTransformer) {
                ((TrainableTransformer) transformer).fit(data);
            }
            data = transformer.transform(data);
        }
        getDistanceMeasure().buildDistanceMeasure(data);
    }
    
    default void buildDistanceMeasure(TimeSeriesInstances data) {
        buildDistanceMeasure(data, true);
    }
    
    default double distance(TimeSeriesInstance a, TimeSeriesInstance b, double limit) {
        return distance(a, true, b, true, limit);
    }
    
    double distance(TimeSeriesInstance a, boolean transformA, TimeSeriesInstance b, boolean transformB, double limit);
    
    default double distance(TimeSeriesInstance a, boolean transformA, TimeSeriesInstance b, boolean transformB) {
        return distance(a, transformA, b, transformB, Double.POSITIVE_INFINITY);
    }
    
    default double distanceUnivariate(double[] a, boolean transformA, double[] b, boolean transformB, double limit) {
        return distance(new TimeSeriesInstance(a), transformA, new TimeSeriesInstance(b), transformB, limit);
    }

    default double distanceUnivariate(double[] a, boolean transformA, double[] b, boolean transformB) {
        return distanceUnivariate(a, transformA, b, transformB, Double.POSITIVE_INFINITY);
    }

    default double distanceMultivariate(double[][] a, boolean transformA, double[][] b, boolean transformB, double limit) {
        return distance(new TimeSeriesInstance(a), transformA, new TimeSeriesInstance(b), transformB, limit);
    }

    default double distanceMultivariate(double[][] a, boolean transformA, double[][] b, boolean transformB) {
        return distanceMultivariate(a, transformA, b, transformB, Double.POSITIVE_INFINITY);
    }

    default double distanceUnivariate(List<Double> a, boolean transformA, List<Double> b, boolean transformB, double limit) {
        return distanceMultivariate(Collections.singletonList(a), transformA, Collections.singletonList(b), transformB, limit);
    }

    default double distanceUnivariate(List<Double> a, boolean transformA, List<Double> b, boolean transformB) {
        return distanceUnivariate(a, transformA, b, transformB, Double.POSITIVE_INFINITY);
    }

    default double distanceMultivariate(List<List<Double>> a, boolean transformA, List<List<Double>> b, boolean transformB, double limit) {
        return distance(new TimeSeriesInstance(a), transformA, new TimeSeriesInstance(b), transformB, limit);
    }

    default double distanceMultivariate(List<List<Double>> a, boolean transformA, List<List<Double>> b, boolean transformB) {
        return distanceMultivariate(a, transformA, b, transformB, Double.POSITIVE_INFINITY);
    }
}
