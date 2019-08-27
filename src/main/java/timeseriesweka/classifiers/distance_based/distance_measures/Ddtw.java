package timeseriesweka.classifiers.distance_based.distance_measures;

import utilities.cache.CachedFunction;
import weka.core.Instance;

import static timeseriesweka.filters.DerivativeFilter.INSTANCE_DERIVATIVE_FUNCTION;

public class Ddtw extends Dtw {

    private CachedFunction<Instance, Instance> derivativeCache = new CachedFunction<>(INSTANCE_DERIVATIVE_FUNCTION);

    public CachedFunction<Instance, Instance> getDerivativeCache() {
        return derivativeCache;
    }

    public void setDerivativeCache(final CachedFunction<Instance, Instance> derivativeCache) {
        this.derivativeCache = derivativeCache;
    }

    @Override
    public double measureDistance() {
        return transformedDistanceMeasure(this, derivativeCache, super::measureDistance);
    }

    @Override
    public String toString() {
        return NAME;
    }

    public static final String NAME = "DDTW";
}
