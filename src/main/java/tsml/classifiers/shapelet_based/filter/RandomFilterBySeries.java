package tsml.classifiers.shapelet_based.filter;

import tsml.classifiers.shapelet_based.classifiers.MSTC;
import tsml.classifiers.shapelet_based.quality.ShapeletQualityFunction;
import tsml.classifiers.shapelet_based.functions.ShapeletFunctions;
import tsml.classifiers.shapelet_based.type.ShapeletIndependentMV;
import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;
import java.util.PriorityQueue;

public class RandomFilterBySeries extends RandomFilter {


    @Override
    public ArrayList<ShapeletMV> findShapelets(MSTC.ShapeletParams params,
                                               TimeSeriesInstances instances) {

        PriorityQueue[][] queues = new PriorityQueue[instances.getMaxNumDimensions()][instances.numClasses()];

        for (int i=0;i<queues.length;i++){
            for (int j=0;j<queues[i].length;j++)
                queues[i][j] = new PriorityQueue<ShapeletIndependentMV>(ShapeletIndependentMV::compareTo);
        }
        int k_local = params.k / (instances.getMaxNumDimensions()*instances.numClasses());

        long start = System.nanoTime();
        int numClasses = instances.getClassCounts().length;
        ShapeletFunctions type = params.type.createShapeletType();
        ShapeletQualityFunction quality = params.quality.createShapeletQuality(instances,
                params.distance.createShapeletDistance());

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

            double q = quality.calculate (candidate);
             if (q>0 ){
                candidate.setQuality(q);
                queues[curSeries][curClass].offer(candidate);
                if (queues[curSeries][curClass].size() > k_local)
                   queues[curSeries][curClass].poll();
             }
            if (r % 1000 == 0){
                //    Collections.sort(shapelets);
                //    if ( shapelets.size()>k) shapelets.subList(k,shapelets.size()).clear();
                //    System.out.println(shapelets);
                if (withinTrainContract(start)){
                    System.out.println("Contract time reached");
                    return getResults(queues);
                }

            }

        }
        //Collections.sort(shapelets);
        // if ( shapelets.size()>k) shapelets.subList(k,shapelets.size()).clear();
        return getResults(queues);

    }

    protected ArrayList<ShapeletMV> getResults(PriorityQueue<ShapeletMV>[][] s){
        ArrayList<ShapeletMV> result = new ArrayList<ShapeletMV>();
        for (int i=0;i<s.length;i++){
            for (int j=0;j<s[i].length;j++){
                System.out.println("Series: " + i + " Class " + j + " Size " + s[i][j].size());
                while (s[i][j].size() > 0) {
                    result.add(s[i][j].poll());
                }
            }
        }

        return result;
    }


}
