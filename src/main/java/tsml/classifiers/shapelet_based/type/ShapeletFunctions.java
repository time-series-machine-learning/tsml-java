package tsml.classifiers.shapelet_based.type;

import tsml.data_containers.TimeSeriesInstance;

public interface ShapeletFunctions<T extends ShapeletMV> {
    T[] getShapeletsOverInstance(int shapeletSize, int instanceIndex, double classIndex, TimeSeriesInstance instance);
    T getRandomShapelet(int shapeletSize, int instanceIndex, double classIndex, TimeSeriesInstance instance);
    double getDistanceFunction(T t1, T t2);
}
