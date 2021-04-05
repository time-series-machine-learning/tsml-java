package tsml.classifiers.shapelet_based.filter;

import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.shapelet_based.classifiers.MultivariateShapelet;
import tsml.classifiers.shapelet_based.quality.ShapeletQualityFunction;
import tsml.classifiers.shapelet_based.type.ShapeletFunctions;
import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;
import java.util.Collections;

public class RandomFilter implements ShapeletFilterMV, TrainTimeContractable {

    @Override
    public ArrayList<ShapeletMV> findShapelets(MultivariateShapelet.ShapeletParams params,
                                               TimeSeriesInstances instances) {

        long start = System.nanoTime();

        ShapeletFunctions type = params.type.createShapeletType();
        ShapeletQualityFunction quality = params.quality.createShapeletQuality(instances,
                params.distance.createShapeletDistance());


        return randomShapelets(params,instances, start, type, quality,params.maxIterations,params.k);


    }

    protected ArrayList<ShapeletMV> randomShapelets(MultivariateShapelet.ShapeletParams params,
                                                    TimeSeriesInstances instances, long start,
                                                    ShapeletFunctions type, ShapeletQualityFunction quality,
                                                    int max_iterations, int k){

        ArrayList<ShapeletMV> shapelets = new ArrayList<ShapeletMV>();

        int[] classesArray  = instances.getClassIndexes();

        for (int r=0;r<max_iterations  ;r++){ // Iterate

            if (r % (params.maxIterations/10) == 0){
                System.out.println("Iteration: "+ r);
            }

            //Get random shapelet
            int shapeletSize = params.min + MultivariateShapelet.RAND.nextInt(params.max-params.min);
            int instanceIndex =  MultivariateShapelet.RAND.nextInt(instances.numInstances());
            ShapeletMV candidate = type.getRandomShapelet(
                    shapeletSize,
                    instanceIndex,
                    classesArray[instanceIndex],
                    instances.get(instanceIndex));

            //    if (!isSimilar(shapelets, candidate,similarFunction,params.minDist)) {
            double q = quality.calculate (candidate);
            candidate.setQuality(q);
            shapelets.add(candidate);
            //    }
            if (r % 1000 == 0){
                Collections.sort(shapelets);
                if ( shapelets.size()>k) shapelets.subList(k,shapelets.size()).clear();
                //  System.out.println(shapelets);
                if (withinTrainContract(start)){
                    System.out.println("Contract time reached");
                    return shapelets;
                }

            }

        }
        Collections.sort(shapelets);
        if ( shapelets.size()>k) shapelets.subList(k,shapelets.size()).clear();
        return shapelets;
    }

    protected   long time;
    @Override
    public void setTrainTimeLimit(long time) {
        this.time = time;
    }

    @Override
    public boolean withinTrainContract(long start) {
        return System.nanoTime()>time+start;
    }
}
