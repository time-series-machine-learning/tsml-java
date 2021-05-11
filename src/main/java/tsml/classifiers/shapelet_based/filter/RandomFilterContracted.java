package tsml.classifiers.shapelet_based.filter;

import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.shapelet_based.classifiers.MSTC;
import tsml.classifiers.shapelet_based.distances.ShapeletDistanceFunction;
import tsml.classifiers.shapelet_based.functions.ShapeletFunctions;
import tsml.classifiers.shapelet_based.quality.ShapeletQualityFunction;
import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.PriorityQueue;

public class RandomFilterContracted implements ShapeletFilterMV, TrainTimeContractable {

    @Override
    public ArrayList<ShapeletMV> findShapelets(MSTC.ShapeletParams params,
                                               TimeSeriesInstances instances) {

        long start = System.nanoTime();

        ShapeletFunctions type = params.type.createShapeletType();

        ShapeletDistanceFunction distFunction = params.distance.createShapeletDistance();
        ShapeletQualityFunction quality = params.quality.createShapeletQuality(instances,
                distFunction);

        ArrayList<ShapeletMV> shapelets = new ArrayList<ShapeletMV>();

        int[] classesArray  = instances.getClassIndexes();
        int r=0;


        while (!withinTrainContract(start)){ // Iterate

            if (r%1000000 == 0){
                System.out.println("Iteration: "+ r);

            }

            //Get random shapelet
            int shapeletSize = params.min + MSTC.RAND.nextInt(params.max-params.min);
            int instanceIndex =  MSTC.RAND.nextInt(instances.numInstances());
            ShapeletMV candidate = type.getRandomShapelet(
                    shapeletSize,
                    instanceIndex,
                    classesArray[instanceIndex],
                    instances.get(instanceIndex));

            if (!isSimilar(shapelets, candidate,type,params.minDist)) {
                double q = quality.calculate (candidate);
                candidate.setQuality(q);
                shapelets.add(candidate);
            }
            if (r % 10000 == 0){
                Collections.sort(shapelets);
                if ( shapelets.size()>params.k) shapelets.subList(params.k,shapelets.size()).clear();
                //  System.out.println(shapelets);
                if (withinTrainContract(start)){
                    System.out.println("Contract time reached");
                    return shapelets;
                }

            }
            r++;
        }
        Collections.sort(shapelets);
        if ( shapelets.size()>params.k) shapelets.subList(params.k,shapelets.size()).clear();
        return shapelets;


    }

    protected ArrayList<ShapeletMV> randomShapelets(MSTC.ShapeletParams params,
                                                    TimeSeriesInstances instances,
                                                    ArrayList<Integer> options, long start,
                                                    ShapeletFunctions type, ShapeletQualityFunction quality,
                                                    int max_iterations, int k){

        ArrayList<ShapeletMV> shapelets = new ArrayList<ShapeletMV>();

        int[] classesArray  = instances.getClassIndexes();

        for (int r=0;r<max_iterations  ;r++){ // Iterate

            if (r % (params.maxIterations/10) == 0){
                System.out.println("Iteration: "+ r);
            }

            //Get random shapelet
            int shapeletSize = params.min + MSTC.RAND.nextInt(params.max-params.min);
            int instanceIndex =  options.get(MSTC.RAND.nextInt(options.size()));
            ShapeletMV candidate = type.getRandomShapelet(
                    shapeletSize,
                    instanceIndex,
                    classesArray[instanceIndex],
                    instances.get(instanceIndex));

            double q = quality.calculate (candidate);
            if (q>0){
                if (!isSimilar(shapelets, candidate,type,params.minDist)) {
                    candidate.setQuality(q);
                    shapelets.add(candidate);
                }

            }
            if (r % 1000 == 0){
                Collections.sort(shapelets);
                if ( shapelets.size()>params.k) shapelets.subList(params.k,shapelets.size()).clear();
                //  System.out.println(shapelets);
                if (withinTrainContract(start)){
                    System.out.println("Contract time reached");
                    return shapelets;
                }

            }

        }
        Collections.sort(shapelets);
        if ( shapelets.size()>params.k) shapelets.subList(params.k,shapelets.size()).clear();
        return shapelets;

    }

    protected ArrayList<ShapeletMV> getResults(PriorityQueue<ShapeletMV> s){
        ArrayList<ShapeletMV> result = new ArrayList<ShapeletMV>();
        while (s.size() > 0) {
            result.add(s.poll()); //obj5, obj1, obj3
        }
        return result;
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
