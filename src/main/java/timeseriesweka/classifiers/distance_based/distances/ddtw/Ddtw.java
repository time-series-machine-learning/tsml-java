package timeseriesweka.classifiers.distance_based.distances.ddtw;

import timeseriesweka.classifiers.distance_based.distances.dtw.Dtw;
import timeseriesweka.filters.Cache;
import timeseriesweka.filters.DerivativeCache;

public class Ddtw extends Dtw {

    private final Cache<double[], double[]> cache = new Cache<double[], double[]>(DerivativeCache::getDerivative);

    @Override
    public double distance() {
        double[] originalA = getTarget();
        double[] originalB = getCandidate();
        double[] a = cache.get(originalA);
        double[] b = cache.get(originalB);
        setTarget(a);
        setCandidate(b);
        double distance = super.distance();
        setTarget(originalA);
        setCandidate(originalB);
        return distance;
    }

    @Override
    public String toString() {
        return NAME;
    }

    public static final String NAME = "DDTW";
}
