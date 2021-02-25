package tsml.classifiers.shapelet_based.filter;

import tsml.classifiers.shapelet_based.classifiers.MultivariateShapelet;
import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;

public class RandomFilter implements ShapeletFilterMV {


    @Override
    public ArrayList<ShapeletMV> findShapelets(MultivariateShapelet.ShapeletParams params, TimeSeriesInstances instances) {
        return null;
    }
}
