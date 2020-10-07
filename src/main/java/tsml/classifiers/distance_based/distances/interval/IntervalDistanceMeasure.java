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
