package timeseriesweka.classifiers.distance_based.distances.ddtw;

import timeseriesweka.classifiers.distance_based.distances.dtw.Dtw;
import timeseriesweka.filters.DerivativeFilter;
import timeseriesweka.filters.cache.CachedFunction;
import utilities.FilterUtilities;
import weka.core.Instance;

public class Ddtw extends Dtw {

    private CachedFunction<Instance, Integer, Instance> derivativeCache = new CachedFunction<>(instance -> {
        try {
            return FilterUtilities.filter(instance, DerivativeFilter.INSTANCE);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }, this::hash);

    public CachedFunction<Instance, Integer, Instance> getDerivativeCache() {
        return derivativeCache;
    }

    public void setDerivativeCache(final CachedFunction<Instance, Integer, Instance> derivativeCache) {
        this.derivativeCache = derivativeCache;
    }

    @Override
    public double measureDistance() {
        Instance origFirst = getFirstInstance();
        Instance origSecond = getSecondInstance();
        Instance first = derivativeCache.get(origFirst);
        Instance second = derivativeCache.get(origSecond);
        setFirstInstance(first);
        setSecondInstance(second);
        double distance = super.measureDistance();
        setFirstInstance(origFirst);
        setSecondInstance(origSecond);
        return distance;
    }

    @Override
    public String toString() {
        return NAME;
    }

    public static final String NAME = "DDTW";
}
