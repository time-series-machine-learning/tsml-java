package tsml.classifiers.shapelet_based.filter;

import tsml.classifiers.TrainTimeContractable;
import tsml.classifiers.shapelet_based.classifiers.MultivariateShapelet;
import tsml.classifiers.shapelet_based.quality.ShapeletQualityMV;
import tsml.classifiers.shapelet_based.type.ShapeletFactoryMV;
import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import tsml.classifiers.shapelet_based.distances.ShapeletDistanceMV;
import utilities.ClusteringUtilities;

public class RandomFilter implements ShapeletFilterMV, TrainTimeContractable {

    @Override
    public ArrayList<ShapeletMV> findShapelets(MultivariateShapelet.ShapeletParams params, TimeSeriesInstances instances) {
        long start = System.nanoTime();
        ArrayList<ShapeletMV> shapelets = new ArrayList<ShapeletMV>();
        double[][][] instancesArray = instances.toValueArray();
        int[] classesArray  = instances.getClassIndexes();
        ShapeletFactoryMV type = params.type.createShapeletType();
        ShapeletQualityMV quality = params.quality.createShapeletQuality(instancesArray,
                instances.getClassIndexes(),instances.getClassLabels(),instances.getClassCounts(),
                params.distance.createShapeletDistance());

        for (int index=0;index<instancesArray.length;index++) { // For each instance
            for (int channel = 0; channel < instancesArray[index].length; channel++) { // For each channel
                ClusteringUtilities.zNormalise(instancesArray[index][channel]);
            }
        }
        for (int r=0;r<params.maxIterations  ;r++){ // Iterate

            if (r % 10000 == 0){
                System.out.println(r);
            //    System.out.println(shapelets);
            }
            int shapeletSize = params.min + MultivariateShapelet.RAND.nextInt(params.max-params.min);
            int instanceIndex =  MultivariateShapelet.RAND.nextInt(instancesArray.length);

            ShapeletMV candidate = type.getRandomShapelet(shapeletSize,instanceIndex,classesArray[instanceIndex], instancesArray[instanceIndex]);

            if (!isSimilar(shapelets, candidate,params.distance.createShapeletDistance(),params.minDist)) {
                double q = quality.calculate (candidate);

                candidate.setQuality(q);
                shapelets.add(candidate);
            }
            if (r % 10000 == 0){
                Collections.sort(shapelets);
                if ( shapelets.size()>params.k) shapelets.subList(params.k,shapelets.size()).clear();
                if (withinTrainContract(start)){
                    System.out.println("Contract time reached");
                    return shapelets;
                }

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
        return System.nanoTime()>time+start;
    }
}
