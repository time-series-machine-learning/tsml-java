package timeseriesweka.classifiers.distance_based.distances;

import timeseriesweka.classifiers.distance_based.distances.ddtw.Ddtw;
import timeseriesweka.classifiers.distance_based.distances.dtw.Dtw;
import timeseriesweka.classifiers.distance_based.distances.erp.Erp;
import timeseriesweka.classifiers.distance_based.distances.lcss.Lcss;
import timeseriesweka.classifiers.distance_based.distances.msm.Msm;
import timeseriesweka.classifiers.distance_based.distances.twed.Twed;
import timeseriesweka.classifiers.distance_based.distances.wddtw.Wddtw;
import timeseriesweka.classifiers.distance_based.distances.wdtw.Wdtw;
import utilities.Options;
import utilities.Utilities;
import weka.core.Instance;

import java.io.Serializable;
import java.util.Enumeration;

public abstract class DistanceMeasure implements Serializable,
                                                 Options {

    public DistanceMeasure() { }

    private double limit = Double.POSITIVE_INFINITY;
    private double[] target;
    private Instance targetInstance;
    private double[] candidate;
    private Instance candidateInstance;

    public abstract double distance();

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

    public double getLimit() {
        return limit;
    }

    public void setLimit(final double limit) {
        this.limit = limit;
    }

    @Override
    public Enumeration listOptions() {
        throw new UnsupportedOperationException();
    }

    public double[] getCandidate() {
        return candidate;
    }

    public void setCandidate(final double[] candidate) {
        this.candidate = candidate;
    }

    public double[] getTarget() {
        return target;
    }

    public void setTarget(final double[] target) {
        this.target = target;
    }

    public Instance getCandidateInstance() {
        return candidateInstance;
    }

    public void setCandidate(final Instance candidateInstance) {
        this.candidateInstance = candidateInstance;
        setCandidate(Utilities.extractTimeSeries(candidateInstance));
    }

    public Instance getTargetInstance() {
        return targetInstance;
    }

    public void setTarget(final Instance targetInstance) {
        this.targetInstance = targetInstance;
        setTarget(Utilities.extractTimeSeries(targetInstance));
    }
}
