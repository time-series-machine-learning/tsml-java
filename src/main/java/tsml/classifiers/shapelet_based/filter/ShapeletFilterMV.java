package tsml.classifiers.shapelet_based.filter;

import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.shapelet_based.classifiers.MultivariateShapelet;
import tsml.classifiers.shapelet_based.type.ShapeletFunctions;
import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;

public interface ShapeletFilterMV extends TrainTimeContractable {


    public ArrayList<ShapeletMV> findShapelets(MultivariateShapelet.ShapeletParams params,
                                               TimeSeriesInstances instnces);
    default boolean isSimilar(ArrayList<ShapeletMV> shapelets, ShapeletMV candidate, ShapeletFunctions shapeletFunctions, double minDist){

        for (ShapeletMV shapelet: shapelets){
            if (shapeletFunctions.getDistanceFunction(shapelet, candidate)<minDist)
                return true;
        }
        return false;
    }
}
