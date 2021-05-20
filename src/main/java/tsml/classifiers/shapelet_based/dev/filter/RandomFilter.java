package tsml.classifiers.shapelet_based.dev.filter;

import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.classifiers.shapelet_based.dev.distances.ShapeletDistanceFunction;
import tsml.classifiers.shapelet_based.dev.functions.ShapeletFunctions;
import tsml.classifiers.shapelet_based.dev.quality.ShapeletQualityFunction;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;
import java.util.Collections;

public class RandomFilter implements ShapeletFilterMV, TrainTimeContractable {

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

        for (int r=0;r<params.maxIterations  ;r++){ // Iterate

            if (r % (params.maxIterations/10) == 0){
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

            if (!params.compareSimilar || !isSimilar(shapelets, candidate,type,params.minDist)) {
                double q = quality.calculate (candidate);
                candidate.setQuality(q);
                shapelets.add(candidate);
            }
            if (r % 1000 == 0){
             //   shapelets = removeSelfSimilar(type, shapelets);
                Collections.sort(shapelets);
                if ( shapelets.size()>params.k) shapelets.subList(params.k,shapelets.size()).clear();
                //  System.out.println(shapelets);
                if (withinTrainContract(start)){
                    System.out.println("Contract time reached");
                    return shapelets;
                }

            }

        }
      //  shapelets = removeSelfSimilar(type, shapelets);
        Collections.sort(shapelets);
        if ( shapelets.size()>params.k) shapelets.subList(params.k,shapelets.size()).clear();
        return shapelets;


    }

    protected ArrayList<ShapeletMV> randomShapelets(MSTC.ShapeletParams params,
                                                    TimeSeriesInstances instances,
                                                    long start,
                                                    ShapeletFunctions fun, ShapeletQualityFunction quality,
                                                    int max_iterations, int k){




        ArrayList<ShapeletMV> shapelets = new ArrayList<ShapeletMV>();
        double oldAvg = 0;
        int noChange = 0;
        int[] classesArray  = instances.getClassIndexes();

        for (int r=0;r<max_iterations  ;r++){ // Iterate

            if (r % (max_iterations/10) == 0){
                System.out.println("Iteration: "+ r);
            }

            //Get random shapelet
            int shapeletSize = params.min + MSTC.RAND.nextInt(params.max-params.min);
            int instanceIndex =  MSTC.RAND.nextInt(instances.numInstances());
            ShapeletMV candidate = fun.getRandomShapelet(
                    shapeletSize,
                    instanceIndex,
                    classesArray[instanceIndex],
                    instances.get(instanceIndex));


            if (!params.compareSimilar || !isSimilar(shapelets, candidate,fun,params.minDist)) {
                double q = quality.calculate (candidate);
                candidate.setQuality(q);
                shapelets.add(candidate);
            }

            if (r % 1000 == 0){
                Collections.sort(shapelets);
                if ( shapelets.size()>k) shapelets.subList(k,shapelets.size()).clear();

                //  System.out.println(shapelets);
                if (withinTrainContract(start)){
                    System.out.println("Contract time reached");
                    return shapelets;
                }

                double avg = shapelets.stream().mapToDouble(ShapeletMV::getQuality).average().orElse(0);
                System.out.println("Size " + shapelets.size() +
                        " Avg: " + avg);

                if (shapelets.size() == k && avg > 0.999){
                    System.out.println("Perfect shapelets found");
                    return shapelets;
                }
                if (oldAvg == avg){
                    noChange++;
                }else{
                    oldAvg = avg;
                    noChange=1;
                }

                if (noChange>=50 && avg >0.1){
                    System.out.println("No improvement after 50 checks, ending");
                    return shapelets;
                }

                if (avg < 0.3){
                    r=0;
                    System.out.println("Restart iterations");
                }

            }

        }

        Collections.sort(shapelets);
        if ( shapelets.size()>k) shapelets.subList(k,shapelets.size()).clear();
        return shapelets;

    }

    protected  ArrayList<ShapeletMV> removeSelfSimilar(ShapeletFunctions fun, ArrayList<ShapeletMV> shapelets) {
        // return a new pruned array list - more efficient than removing
        // self-similar entries on the fly and constantly reindexing
        ArrayList<ShapeletMV> outputShapelets = new ArrayList<>();
        int size = shapelets.size();
        boolean[] selfSimilar = new boolean[size];

        for (int i = 0; i < size; i++) {
            if (selfSimilar[i]) {
                continue;
            }

            outputShapelets.add(shapelets.get(i));

            for (int j = i + 1; j < size; j++) {
                // no point recalc'ing if already self similar to something
                if ((!selfSimilar[j]) && fun.selfSimilarity(shapelets.get(i), shapelets.get(j))) {
                    selfSimilar[j] = true;
                }
            }
        }
        return outputShapelets;
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
