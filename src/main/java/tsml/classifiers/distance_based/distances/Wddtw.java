package tsml.classifiers.distance_based.distances;

import weka.core.Instance;
import weka.core.neighboursearch.PerformanceStats;

import static tsml.classifiers.distance_based.distances.Ddtw.DERIVATIVE_CACHE;

public class Wddtw extends Wdtw {

    private utilities.cache.CachedFunction<Instance, Instance> derivativeCache = DERIVATIVE_CACHE;

    public utilities.cache.CachedFunction<Instance, Instance> getDerivativeCache() {
        return derivativeCache;
    }

    public void setDerivativeCache(final utilities.cache.CachedFunction<Instance, Instance> derivativeCache) {
        this.derivativeCache = derivativeCache;
    }

    @Override
    public double distance(final Instance first,
                           final Instance second,
                           final double limit,
                           final PerformanceStats stats) {
        Instance transformedFirst = derivativeCache.apply(first);
        Instance transformedSecond = derivativeCache.apply(second);
        return super.distance(transformedFirst, transformedSecond, limit, stats);
    }

}
