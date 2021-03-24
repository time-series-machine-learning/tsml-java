package tsml.classifiers.shapelet_based.filter;

import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.shapelet_based.classifiers.MultivariateShapelet;
import tsml.classifiers.shapelet_based.distances.ShapeletDistanceMV;
import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;

public interface ShapeletFilterMV extends TrainTimeContractable {


    public ArrayList<ShapeletMV> findShapelets(MultivariateShapelet.ShapeletParams params, TimeSeriesInstances instances);
    default boolean isSimilar(ArrayList<ShapeletMV> shapelets, ShapeletMV candidate, ShapeletDistanceMV distance, double minDist){

        for (ShapeletMV shapelet: shapelets){
            if (distance.calculate(shapelet.toDoubleArray(), candidate.toDoubleArray(),Math.min(candidate.getLength(),shapelet.getLength()))<minDist)
                return true;
        }
        return false;
    }
}
