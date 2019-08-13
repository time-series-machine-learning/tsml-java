package timeseriesweka.classifiers.distance_based.distances;

import timeseriesweka.filters.cache.CachedFunction;
import weka.core.Instance;

public abstract class FilteredDistanceMeasure {

    abstract CachedFunction<Instance, Integer, Instance> getCachedFunction();

    abstract void setCachedFunction(final CachedFunction<Instance, Integer, Instance> derivativeCache);
}
