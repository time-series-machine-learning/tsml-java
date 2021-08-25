package tsml.classifiers.shapelet_based.dev.quality;

import tsml.classifiers.shapelet_based.dev.functions.ShapeletFunctions;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

public abstract class ShapeletQualityFunction {

    protected TimeSeriesInstances trainInstances;


    public ShapeletQualityFunction(TimeSeriesInstances instances){
        this.trainInstances = instances;

    }



    public abstract double calculate(ShapeletFunctions fun, ShapeletMV candidate);



}
