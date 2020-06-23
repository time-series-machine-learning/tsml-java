package tsml.classifiers.distance_based.distances.interval;

import org.junit.Assert;
import tsml.classifiers.distance_based.distances.BaseDistanceMeasure;
import tsml.classifiers.distance_based.distances.ed.EuclideanDistance;
import tsml.classifiers.distance_based.utils.collections.intervals.Interval;
import tsml.classifiers.distance_based.utils.collections.intervals.IntervalInstance;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import weka.core.DistanceFunction;
import weka.core.Instance;

public class IntervalDistanceMeasure extends BaseDistanceMeasure {

    private Interval intervalA = new Interval();
    private Interval intervalB = new Interval();
    public static final String INTERVAL_A_FLAG = "a";
    public static final String INTERVAL_B_FLAG = "b";
    public static final String DISTANCE_FUNCTION_FLAG = "d";
    private DistanceFunction distanceFunction = new EuclideanDistance();

    public IntervalDistanceMeasure() {

    }

    @Override protected double findDistance(final Instance a, final Instance b, final double limit) {
        final IntervalInstance ia = new IntervalInstance(intervalA, a);
        final IntervalInstance ib = new IntervalInstance(intervalB, b);
        return distanceFunction.distance(ia, ib, limit);
    }

    public Interval getIntervalA() {
        return intervalA;
    }

    public void setIntervalA(final Interval intervalA) {
        Assert.assertNotNull(intervalA);
        this.intervalA = intervalA;
    }

    public Interval getIntervalB() {
        return intervalB;
    }

    public void setIntervalB(final Interval intervalB) {
        Assert.assertNotNull(intervalB);
        this.intervalB = intervalB;
    }

    public void setInterval(Interval interval) {
        setIntervalA(interval);
        setIntervalB(interval);
    }

    @Override public ParamSet getParams() {
        return super.getParams().add(INTERVAL_A_FLAG, intervalA).add(INTERVAL_B_FLAG, intervalB).add(DISTANCE_FUNCTION_FLAG, distanceFunction);
    }

    @Override public void setParams(final ParamSet param) throws Exception {
        super.setParams(param);
        ParamHandlerUtils.setParam(param, INTERVAL_A_FLAG, this::setIntervalA, Interval.class);
        ParamHandlerUtils.setParam(param, INTERVAL_B_FLAG, this::setIntervalB, Interval.class);
        ParamHandlerUtils.setParam(param, DISTANCE_FUNCTION_FLAG, this::setDistanceFunction, DistanceFunction.class);
    }

    public DistanceFunction getDistanceFunction() {
        return distanceFunction;
    }

    public void setDistanceFunction(final DistanceFunction distanceFunction) {
        this.distanceFunction = distanceFunction;
    }
}
