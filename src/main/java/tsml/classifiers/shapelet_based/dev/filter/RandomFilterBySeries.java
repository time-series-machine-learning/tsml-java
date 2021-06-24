package tsml.classifiers.shapelet_based.dev.filter;

import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.classifiers.shapelet_based.dev.distances.ShapeletDistanceFunction;
import tsml.classifiers.shapelet_based.dev.functions.ShapeletFunctions;
import tsml.classifiers.shapelet_based.dev.quality.ShapeletQualityFunction;
import tsml.classifiers.shapelet_based.dev.type.ShapeletIndependentMV;
import tsml.classifiers.shapelet_based.dev.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;
import java.util.Collections;

public class RandomFilterBySeries extends RandomFilter {


    @Override
    public ArrayList<ShapeletMV> findShapelets(MSTC.ShapeletParams params,
                                               TimeSeriesInstances instances) {

        ArrayList[][] queues = new ArrayList[instances.getMaxNumDimensions()][instances.numClasses()];

        for (int i=0;i<queues.length;i++){
            for (int j=0;j<queues[i].length;j++)
                queues[i][j] = new ArrayList<ShapeletIndependentMV>();
        }
        int k_local = params.k / (instances.getMaxNumDimensions()*instances.numClasses());

        long start = System.nanoTime();
        int numClasses = instances.getClassCounts().length;
        ShapeletFunctions type = params.type.createShapeletType();
        ShapeletDistanceFunction distanceFunction= params.distance.createShapeletDistance(type);
        ShapeletQualityFunction quality = params.quality.createShapeletQuality(instances);

        int[] classesArray  = instances.getClassIndexes();

        for (int r=0;r<params.maxIterations  ;r++){ // Iterate

            if (r % (params.maxIterations/10) == 0){
                System.out.println("Iteration: "+ r);
            }

            //Get random shapelet
            int shapeletSize = params.min + MSTC.RAND.nextInt(params.max-params.min);
            int instanceIndex = MSTC.RAND.nextInt(instances.numInstances());
            ShapeletIndependentMV candidate = (ShapeletIndependentMV) type.getRandomShapelet(
                    shapeletSize,
                    instanceIndex,
                    classesArray[instanceIndex],
                    instances.get(instanceIndex));

            int curSeries = candidate.getSeriesIndex();
            int curClass = (int)candidate.getClassIndex();

            double q = quality.calculate (distanceFunction, candidate);
             if (q>0 ){
                candidate.setQuality(q);
                queues[curSeries][curClass].add(candidate);
             }
            if (r % 10000 == 0){
                for (int i=0;i<queues.length;i++){
                    for (int j=0;j<queues[i].length;j++){
                        Collections.sort(queues[i][j]);
                        if ( queues[i][j].size()>k_local) queues[i][j].subList(k_local,queues[i][j].size()).clear();
                    }
                }
                if (withinTrainContract(start)){
                    System.out.println("Contract time reached");
                    return getResults(queues);
                }

            }
        }
        for (int i=0;i<queues.length;i++){
            for (int j=0;j<queues[i].length;j++){
                Collections.sort(queues[i][j]);
                if ( queues[i][j].size()>k_local) queues[i][j].subList(k_local,queues[i][j].size()).clear();
            }
        }
        return getResults(queues);

    }

    protected ArrayList<ShapeletMV> getResults(ArrayList<ShapeletMV>[][] s){
        ArrayList<ShapeletMV> result = new ArrayList<ShapeletMV>();
        for (int i=0;i<s.length;i++){
            for (int j=0;j<s[i].length;j++){

                result.addAll(s[i][j]);
            }
        }

        return result;
    }


}
