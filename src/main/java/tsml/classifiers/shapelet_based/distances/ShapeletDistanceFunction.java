package tsml.classifiers.shapelet_based.distances;

import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstance;

public interface ShapeletDistanceFunction {

    public double calculate(ShapeletMV shapelet, TimeSeriesInstance instance);

}
