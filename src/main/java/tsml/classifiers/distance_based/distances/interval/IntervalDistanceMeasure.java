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
 
package tsml.classifiers.distance_based.distances.interval;

import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.utils.collections.intervals.Interval;
import tsml.classifiers.distance_based.utils.collections.intervals.IntervalInstance;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;

public class IntervalDistanceMeasure extends BaseDistanceMeasure {

    private Interval interval;
    private Interval altInterval;
    private DistanceFunction distanceFunction;

    public IntervalDistanceMeasure(Interval interval, DistanceFunction distanceFunction) {
        this(interval, null, distanceFunction);
    }

    public IntervalDistanceMeasure(Interval interval, Interval altInterval, DistanceFunction distanceFunction) {
        setInterval(interval);
        setAltInterval(altInterval);
        setDistanceFunction(distanceFunction);
    }

    public IntervalDistanceMeasure() {}

    public Interval getInterval() {
        return interval;
    }

    public void setInterval(final Interval interval) {
        this.interval = interval;
    }

    public Interval getAltInterval() {
        return altInterval;
    }

    public void setAltInterval(final Interval altInterval) {
        this.altInterval = altInterval;
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public void setDistanceFunction(final DistanceFunction distanceFunction) {
        this.distanceFunction = distanceFunction;
        setName(distanceFunction.toString() + "I");
    }

    @Override protected double findDistance(Instance a, Instance b, final double limit) {
        if(interval != null) {
            a = new IntervalInstance(interval, a);
            if(altInterval == null) {
                b = new IntervalInstance(interval, b);
            } else {
                b = new IntervalInstance(altInterval, b);
            }
        }
        return distanceFunction.distance(a, b, limit);
    }

    @Override public void setInstances(final Instances data) {
        super.setInstances(data);
        distanceFunction.setInstances(data);
    }
}
