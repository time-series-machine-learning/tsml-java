package tsml.classifiers.shapelet_based.filter;

import tsml.classifiers.shapelet_based.classifiers.MSTC;
import tsml.classifiers.shapelet_based.quality.ShapeletQualityFunction;
import tsml.classifiers.shapelet_based.functions.ShapeletFunctions;
import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;
import java.util.Collections;

public class ExhaustiveFilter implements ShapeletFilterMV {


    public ArrayList<ShapeletMV> findShapelets(MSTC.ShapeletParams params,
                                               TimeSeriesInstances instances) {

        ArrayList<ShapeletMV> shapelets = new ArrayList<ShapeletMV>();

        int[] classesArray  = instances.getClassIndexes();

        ShapeletFunctions type = params.type.createShapeletType();
        ShapeletQualityFunction quality = params.quality.createShapeletQuality(instances,
                params.distance.createShapeletDistance());


        for (int index=0;index<instances.numInstances();index++){ // For each instance
            for (int shapeletSize=params.min;shapeletSize<=params.max;shapeletSize++) {  // For each shapelet size

                ShapeletMV[] candidates = type.getShapeletsOverInstance(shapeletSize,index,classesArray[index],instances.get(index));

                for (int candidate = 0 ; candidate < candidates.length; candidate++){

                     if (isSimilar(shapelets, candidates[candidate],type,params.minDist)) continue;
                    double q = quality.calculate (candidates[candidate]);
                    candidates[candidate].setQuality(q);
                    shapelets.add(candidates[candidate]);
                }
            }
            Collections.sort(shapelets);
            if (shapelets.size()>params.k) shapelets.subList(params.k,shapelets.size()).clear();

        }
        return shapelets;
    }

    private  long time;
    @Override
    public void setTrainTimeLimit(long time) {
        this.time = time;
    }

    @Override
    public boolean withinTrainContract(long start) {
        return false;
    }


}
