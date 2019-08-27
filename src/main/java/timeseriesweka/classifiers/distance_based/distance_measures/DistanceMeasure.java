package timeseriesweka.classifiers.distance_based.distance_measures;

import timeseriesweka.classifiers.Loggable;
import utilities.Options;
import weka.core.Instance;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.logging.Logger;

public abstract class DistanceMeasure
    implements Serializable,
               Options, Loggable {

    protected Logger logger;

    public Logger getLogger() {
        return logger;
    }

    public void setLogger(Logger logger) {
        this.logger = logger;
    }

    public DistanceMeasure() { }

    private double limit = Double.POSITIVE_INFINITY;
    private Instance firstInstance;
    private Instance secondInstance;

    protected abstract double measureDistance();

    public final double distance() {
        if(firstInstance.classIndex() != firstInstance.numAttributes() - 1) {
            throw new IllegalStateException("class value must be at the end");
        }
        if(secondInstance.classIndex() != secondInstance.numAttributes() - 1) {
            throw new IllegalStateException("class value must be at the end");
        }
        double distance = measureDistance();
        return distance;
    }

    public final double distance(final Instance first, final Instance second) {
        setFirstInstance(first);
        setSecondInstance(second);
        return distance();
    }

    public final double distance(final Instance first, final Instance second, final double limit) {
        setLimit(limit);
        return distance(first, second);
    }

    public static DistanceMeasure fromString(String str) {
        switch(str) {
            case Dtw.NAME: return new Dtw();
            case Ddtw.NAME: return new Ddtw();
            case Wdtw.NAME: return new Wdtw();
            case Wddtw.NAME: return new Wddtw();
            case Twed.NAME: return new Twed();
            case Msm.NAME: return new Msm();
            case Lcss.NAME: return new Lcss();
            case Erp.NAME: return new Erp();
            default: throw new IllegalArgumentException("unknown distance measure: " + str);
        }
    }

    public static final String DISTANCE_MEASURE_KEY = "distanceMeasure";

    @Override
    public abstract String toString();

    public final double getLimit() {
        return limit;
    }

    public final void setLimit(final double limit) {
        this.limit = limit;
    }

    @Override
    public Enumeration listOptions() {
        throw new UnsupportedOperationException();
    }

    public final Instance getSecondInstance() {
        return secondInstance;
    }

    public final void setSecondInstance(final Instance second) {
        this.secondInstance = second;

    }

    public final Instance getFirstInstance() {
        return firstInstance;
    }

    public final void setFirstInstance(final Instance first) {
        this.firstInstance = first;
    }

    public boolean isSymmetric() {
        return true;
    }

    protected static double transformedDistanceMeasure(DistanceMeasure distanceMeasure, Function<Instance, Instance> transformFunction, Supplier<Double> distanceSupplier) {
        Instance origFirst = distanceMeasure.getFirstInstance();
        Instance origSecond = distanceMeasure.getSecondInstance();
        Instance first = transformFunction.apply(origFirst);
        Instance second = transformFunction.apply(origSecond);
        distanceMeasure.setFirstInstance(first);
        distanceMeasure.setSecondInstance(second);
        double distance = distanceSupplier.get();
        distanceMeasure.setFirstInstance(origFirst);
        distanceMeasure.setSecondInstance(origSecond);
        return distance;
    }
}
