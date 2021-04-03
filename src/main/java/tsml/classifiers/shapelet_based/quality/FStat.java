package tsml.classifiers.shapelet_based.quality;

import tsml.classifiers.shapelet_based.distances.ShapeletDistanceFunction;
import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

public class FStat extends ShapeletQualityFunction {

    public FStat(TimeSeriesInstances instances,
                 ShapeletDistanceFunction distance){
        super(instances,distance);
    }

    @Override
    public double calculate(ShapeletMV candidate){
        return 0;
    }
}
