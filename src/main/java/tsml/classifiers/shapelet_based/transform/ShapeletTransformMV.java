package tsml.classifiers.shapelet_based.transform;

import tsml.classifiers.shapelet_based.classifiers.ShapeletDataMV;
import tsml.classifiers.shapelet_based.classifiers.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;

public interface ShapeletTransformMV {

    TimeSeriesInstances transform(TimeSeriesInstances instances, ArrayList<ShapeletMV> shapelets);
}
