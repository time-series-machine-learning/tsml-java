package tsml.classifiers.shapelet_based.filter;

import tsml.classifiers.shapelet_based.classifiers.MultivariateShapelet;
import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;

public interface ShapeletFilterMV {
    public ArrayList<ShapeletMV> findShapelets(MultivariateShapelet.ShapeletParams params, TimeSeriesInstances instances);

}
