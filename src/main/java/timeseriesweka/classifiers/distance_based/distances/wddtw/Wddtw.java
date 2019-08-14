package timeseriesweka.classifiers.distance_based.distances.wddtw;

import timeseriesweka.classifiers.distance_based.distances.wdtw.Wdtw;
import timeseriesweka.filters.DerivativeFilter;
import timeseriesweka.filters.cache.CachedFunction;
import utilities.FilterUtilities;
import weka.core.Instance;

import static timeseriesweka.filters.DerivativeFilter.INSTANCE_DERIVATIVE_FUNCTION;

public class Wddtw extends Wdtw {

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

    public static final String NAME = "WDDTW";
}
