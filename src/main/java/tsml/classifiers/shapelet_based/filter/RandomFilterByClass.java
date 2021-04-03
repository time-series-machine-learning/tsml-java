package tsml.classifiers.shapelet_based.filter;

import tsml.classifiers.shapelet_based.classifiers.MultivariateShapelet;
import tsml.classifiers.shapelet_based.quality.ShapeletQualityFunction;
import tsml.classifiers.shapelet_based.type.ShapeletFunctions;
import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class RandomFilterByClass implements ShapeletFilterMV {


    @Override
    public ArrayList<ShapeletMV> findShapelets(MultivariateShapelet.ShapeletParams params,
                                               TimeSeriesInstances instances) {
        int numClasses = instances.getClassCounts().length;

        ArrayList<ShapeletMV> shapelets = new ArrayList<ShapeletMV>();

        for (int i=0;i<numClasses;i++){
            System.out.println("Class "  + i);
            ArrayList<ShapeletMV> classShapelets = findShapeletsInClass(params,instances,i,50000, params.k/numClasses);
            shapelets.addAll(classShapelets);
        }


        return shapelets;
    }


    private ArrayList<ShapeletMV> findShapeletsInClass(MultivariateShapelet.ShapeletParams params,
                                                       TimeSeriesInstances instances, double classIndex, int max_iterations, int k) {
        ArrayList<ShapeletMV> shapelets = new ArrayList<ShapeletMV>();


        List<TimeSeriesInstance> ci = instances.stream().filter(instance -> instance.getTargetValue()==classIndex).collect(Collectors.toList());
        TimeSeriesInstances classInstances = new TimeSeriesInstances(ci);
        int[] classesArrayForClass  = classInstances.getClassIndexes();

        ShapeletFunctions type = params.type.createShapeletType();
        ShapeletQualityFunction quality = params.quality.createShapeletQuality(instances,
                params.distance.createShapeletDistance());

        double maxQ = 0;
        for (int r=0;r<max_iterations;r++){ // Iterate
            if (r % (max_iterations/10) == 0){
                System.out.println(r);
            }

            int shapeletSize = params.min + MultivariateShapelet.RAND.nextInt(params.max-params.min);

            int instanceIndex =  MultivariateShapelet.RAND.nextInt(classInstances.numInstances());

            ShapeletMV candidate = type.getRandomShapelet(shapeletSize,instanceIndex,classesArrayForClass[instanceIndex], classInstances.get(instanceIndex));

        //    if (!isSimilar(shapelets, candidate,params.distance.createShapeletDistance())) {
                double q = quality.calculate (candidate);

                candidate.setQuality(q);
                shapelets.add(candidate);
                if (q>maxQ){
                    maxQ = q;
                 //   System.out.println(maxQ + " " + classesArray[instanceIndex]);
                }
          //  }
            if (r % 1000 == 0){
                Collections.sort(shapelets);
                if ( shapelets.size()>k) shapelets.subList(k,shapelets.size()).clear();


            }

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
