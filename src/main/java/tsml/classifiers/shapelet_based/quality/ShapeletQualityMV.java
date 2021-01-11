package tsml.classifiers.shapelet_based.quality;

import tsml.classifiers.shapelet_based.classifiers.ShapeletMV;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.transformers.shapelet_tools.OrderLineObj;
import tsml.transformers.shapelet_tools.Shapelet;

public interface ShapeletQualityMV {
    double getQuality(TimeSeriesInstances instances, ShapeletMV shapelet, ShapeletQualityMeasureMV measure);
}
