package tsml.classifiers.shapelet_based.dev.filter;

import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.classifiers.shapelet_based.dev.functions.ShapeletFunctions;
import tsml.classifiers.shapelet_based.dev.quality.ShapeletQualityFunction;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.List;

public abstract class  ShapeletFilterMV  implements TrainTimeContractable {


    public abstract List<ShapeletMV> findShapelets(MSTC.ShapeletParams params,
                                               TimeSeriesInstances instances);

    public abstract List<ShapeletMV> findShapelets(MSTC.ShapeletParams params,
                                                        ShapeletQualityFunction quality,
                                                        TimeSeriesInstances instances);

    protected boolean isSimilar(List<ShapeletMV> shapelets, ShapeletMV candidate, ShapeletFunctions shapeletFunctions, double minDist){

        for (ShapeletMV shapelet: shapelets){
            if (shapeletFunctions.selfSimilarity(shapelet, candidate)){
                    return true;
            }
        }
        return false;
    }



    protected   long time,start;
    @Override
    public void setTrainTimeLimit(long time) {
        this.time = time;
    }

    @Override
    public boolean withinTrainContract(long start) {
        return start+time < System.nanoTime();
    }


}
