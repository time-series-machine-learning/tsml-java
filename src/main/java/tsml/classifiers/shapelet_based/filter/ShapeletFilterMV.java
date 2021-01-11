package tsml.classifiers.shapelet_based.filter;

import tsml.classifiers.shapelet_based.classifiers.MultivariateShapelet;
import tsml.classifiers.shapelet_based.classifiers.ShapeletInterface;
import tsml.classifiers.shapelet_based.classifiers.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;
import tsml.transformers.shapelet_tools.quality_measures.ShapeletQuality;

import java.util.ArrayList;

public interface ShapeletFilterMV {
    public ArrayList<ShapeletMV> findShapelets(MultivariateShapelet.ShapeletParams params, TimeSeriesInstances instances, ShapeletQuality quality);

}
