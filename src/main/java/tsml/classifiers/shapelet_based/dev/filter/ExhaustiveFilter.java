package tsml.classifiers.shapelet_based.dev.filter;

import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.classifiers.shapelet_based.dev.distances.ShapeletDistanceFunction;
import tsml.classifiers.shapelet_based.dev.functions.ShapeletFunctions;
import tsml.classifiers.shapelet_based.dev.quality.ShapeletQualityFunction;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;
import java.util.Collections;

public class ExhaustiveFilter extends ShapeletFilterMV {


    public ArrayList<ShapeletMV> findShapelets(MSTC.ShapeletParams params,
                                               TimeSeriesInstances instances) {


        ShapeletQualityFunction quality = params.quality.createShapeletQuality(instances);

        return findShapelets(params,quality,instances);

    }

    @Override
    public ArrayList<ShapeletMV> findShapelets(MSTC.ShapeletParams params, ShapeletQualityFunction quality,
                                               TimeSeriesInstances instances) {

        start = System.nanoTime();

        ShapeletFunctions fun = params.type.createShapeletType();
        ShapeletDistanceFunction distFunction = params.distance.createShapeletDistance(fun);
        int[] classesArray  = instances.getClassIndexes();


        ArrayList<ShapeletMV> shapelets = new ArrayList<ShapeletMV>();

        for (int shapeletSize=params.min;shapeletSize<=params.max;shapeletSize++) {  // For each shapelet size

            for (int index=0;index<instances.numInstances();index++){ // For each instance

                ShapeletMV[] candidates = fun.getShapeletsOverInstance(shapeletSize,index,classesArray[index],instances.get(index));

                for (int candidate = 0 ; candidate < candidates.length; candidate++){

                    //     if (isSimilar(shapelets, candidates[candidate],fun,params.minDist)) continue;
                    double q = quality.calculate (distFunction, candidates[candidate]);
                    candidates[candidate].setQuality(q);
                    shapelets.add(candidates[candidate]);
                }
            }
            Collections.sort(shapelets);
            if (shapelets.size()>params.k) shapelets.subList(params.k,shapelets.size()).clear();
            double avg = shapelets.stream().mapToDouble(ShapeletMV::getQuality).average().orElse(0);
            System.out.println(shapelets);
            if (withinTrainContract(start)){
                System.out.println("Contract time reached");
                return shapelets;
            }

        }
        return shapelets;
    }


}
