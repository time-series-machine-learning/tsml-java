package tsml.classifiers.shapelet_based.dev.functions;

import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstance;

public interface ShapeletFunctions<T extends ShapeletMV> {
    T[] getShapeletsOverInstance(int shapeletSize, int instanceIndex, double classIndex, TimeSeriesInstance instance);
    T getRandomShapelet(int shapeletSize, int instanceIndex, double classIndex, TimeSeriesInstance instance);
    double getDistanceFunction(T t1, T t2);
    boolean selfSimilarity(T t1, T t2);
}
