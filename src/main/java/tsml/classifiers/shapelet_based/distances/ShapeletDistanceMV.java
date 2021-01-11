package tsml.classifiers.shapelet_based.distances;

import tsml.classifiers.shapelet_based.classifiers.ShapeletMV;

public interface ShapeletDistanceMV {

    public double distance(ShapeletMV shapelet, double[][] instance, int seriesLength);
}
