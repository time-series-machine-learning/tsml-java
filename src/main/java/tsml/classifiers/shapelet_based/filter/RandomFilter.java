package tsml.classifiers.shapelet_based.filter;

import tsml.classifiers.shapelet_based.classifiers.MultivariateShapelet;
import tsml.classifiers.shapelet_based.quality.ShapeletQualityMV;
import tsml.classifiers.shapelet_based.type.ShapeletFactoryMV;
import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.data_containers.TimeSeries;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import utilities.ClusteringUtilities;

import java.util.ArrayList;
import java.util.Collections;
import tsml.classifiers.shapelet_based.distances.ShapeletDistanceMV;

public class RandomFilter implements ShapeletFilterMV {

    private double MIN_DIST = 0.01;
    private int MAX_ITERATTIONS = 100000;
    @Override
    public ArrayList<ShapeletMV> findShapelets(MultivariateShapelet.ShapeletParams params, TimeSeriesInstances instances) {
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

        for (int r=0;r<MAX_ITERATTIONS;r++){ // Iterate
            if (r % 10000 == 0) System.out.println(r);
            int shapeletSize = params.min + MultivariateShapelet.RAND.nextInt(params.max-params.min);
            int instanceIndex =  MultivariateShapelet.RAND.nextInt(instancesArray.length);

            ShapeletMV candidate = type.getRandomShapelet(shapeletSize,instanceIndex,instancesArray[instanceIndex]);

          //  if (isSimilar(shapelets, candidate,params.distance.createShapeletDistance())) continue;
            double q = quality.calculate (candidate);
            candidate.setQuality(q);
            shapelets.add(candidate);
            if (r % 1000 == 0){
                Collections.sort(shapelets);
                if ( shapelets.size()>params.k) shapelets.subList(params.k,shapelets.size()).clear();
            }

        }
        return shapelets;
    }
    private boolean isSimilar(ArrayList<ShapeletMV> shapelets, ShapeletMV candidate, ShapeletDistanceMV distance){

        for (ShapeletMV shapelet: shapelets){
            if (distance.calculate(shapelet.toDoubleArray(), candidate.toDoubleArray(),candidate.getLength())<MIN_DIST)
                return true;
        }
        return false;
    }
}
