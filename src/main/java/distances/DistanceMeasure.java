package distances;

import distances.derivative_time_domain.ddtw.CachedDdtw;
import distances.derivative_time_domain.ddtw.Ddtw;
import distances.time_domain.dtw.Dtw;
import distances.time_domain.erp.Erp;
import distances.time_domain.lcss.Lcss;
import distances.time_domain.msm.Msm;
import distances.time_domain.twe.Twe;
import distances.derivative_time_domain.wddtw.CachedWddtw;
import distances.derivative_time_domain.wddtw.Wddtw;
import distances.time_domain.wdtw.Wdtw;
import weka.core.Instance;
import weka.core.NormalizableDistance;
import weka.core.OptionHandler;

import java.io.Serializable;

public abstract class DistanceMeasure extends NormalizableDistance implements Serializable, OptionHandler {

    public DistanceMeasure() {
        setDontNormalize(true); // disable WEKA's normalisation - shouldn't use it anyway but just in case!
    }

    @Override
    public String globalInfo() {
        throw new UnsupportedOperationException();
    }

    @Override
    protected double updateDistance(double currDist, double diff) {
        throw new UnsupportedOperationException();
    }

    /**
     * find distance between two instances
     * @param instanceA first instance
     * @param instanceB second instance
     * @return distance between the two instances
     */
    public final double distance(Instance instanceA, Instance instanceB) {
        return distance(instanceA, instanceB, Double.POSITIVE_INFINITY);
    }

    /**
     * find distance between two instances
     * @param instanceA first instance
     * @param instanceB second instance
     * @param cutOff cut off value to abandon distance measurement early
     * @return distance between the two instances
     */
    public abstract double distance(Instance instanceA, Instance instanceB, double cutOff);

    @Override
    public String getRevision() {
        throw new UnsupportedOperationException();
    }

    public static DistanceMeasure fromString(String str) {
        str = str.toLowerCase();
        switch(str) {
            case "dtw": return new Dtw();
            case "ddtw": return new Ddtw();
            case "cddtw": return new CachedDdtw();
            case "wdtw": return new Wdtw();
            case "wddtw": return new Wddtw();
            case "cwddtw": return new CachedWddtw();
            case "twe": return new Twe();
            case "msm": return new Msm();
            case "lcss": return new Lcss();
            case "erp": return new Erp();
            default: throw new IllegalArgumentException("unknown distance measure: " + str);
        }
    }

    public static final String DISTANCE_MEASURE_KEY = "dm";

    @Override
    public String toString() {
        throw new UnsupportedOperationException();
    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }

    @Override
    public void setOptions(final String[] options) throws
                                                   Exception {

    }
}
