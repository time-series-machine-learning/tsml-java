package tsml.classifiers.shapelet_based.distances;

import tsml.classifiers.shapelet_based.type.ShapeletMV;

public interface ShapeletDistanceMV {

    public double calculate(ShapeletMV shapelet, double[][] instance);
    public double calculate(double[][] a, double[][] b, double shapeletLength);

}
