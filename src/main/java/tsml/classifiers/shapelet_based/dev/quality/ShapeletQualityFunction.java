package tsml.classifiers.shapelet_based.dev.quality;

import tsml.classifiers.shapelet_based.dev.distances.ShapeletDistanceFunction;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

public abstract class ShapeletQualityFunction {

    protected TimeSeriesInstances trainInstances;


    public ShapeletQualityFunction(TimeSeriesInstances instances){
        this.trainInstances = instances;

    }



    public abstract double calculate(ShapeletDistanceFunction distance, ShapeletMV candidate);



}
