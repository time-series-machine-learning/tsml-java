package tsml.classifiers.distance_based.distances;

import utilities.cache.CachedFunction;
import weka.core.Instance;
import weka.core.neighboursearch.PerformanceStats;

import static tsml.filters.Derivative.INSTANCE_DERIVATIVE_FUNCTION;


public class DDTWDistance extends DTWDistance {
    public static final CachedFunction<Instance, Instance> DERIVATIVE_CACHE = new CachedFunction<>(INSTANCE_DERIVATIVE_FUNCTION);

    private CachedFunction<Instance, Instance> derivativeCache = DERIVATIVE_CACHE;

    public CachedFunction<Instance, Instance> getDerivativeCache() {
        return derivativeCache;
    }

    public void setDerivativeCache(final CachedFunction<Instance, Instance> derivativeCache) {
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

    public DDTWDistance() {
        super();
    }

    public DDTWDistance(int warpingWindow) {
        super(warpingWindow);
    }

}
