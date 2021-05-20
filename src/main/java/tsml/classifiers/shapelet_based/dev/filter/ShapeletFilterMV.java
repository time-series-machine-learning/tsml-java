package tsml.classifiers.shapelet_based.dev.filter;

import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.classifiers.shapelet_based.dev.functions.ShapeletFunctions;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;

public interface ShapeletFilterMV extends TrainTimeContractable {


    public ArrayList<ShapeletMV> findShapelets(MSTC.ShapeletParams params,
                                               TimeSeriesInstances instnces);
    default boolean isSimilar(ArrayList<ShapeletMV> shapelets, ShapeletMV candidate, ShapeletFunctions shapeletFunctions, double minDist){

        for (ShapeletMV shapelet: shapelets){
            if (shapeletFunctions.getDistanceFunction(shapelet, candidate)<minDist)
                return true;
        }
        return false;
    }
}
