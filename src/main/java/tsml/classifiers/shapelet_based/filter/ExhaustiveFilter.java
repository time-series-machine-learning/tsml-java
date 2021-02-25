package tsml.classifiers.shapelet_based.filter;

import tsml.classifiers.shapelet_based.classifiers.MultivariateShapelet;
import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.classifiers.shapelet_based.distances.ShapeletDistanceMV;
import tsml.classifiers.shapelet_based.quality.ShapeletQualityMV;
import tsml.classifiers.shapelet_based.type.ShapeletFactoryMV;
import tsml.data_containers.TimeSeriesInstances;
import utilities.ClusteringUtilities;

import java.util.ArrayList;
import java.util.Collections;

public class ExhaustiveFilter implements ShapeletFilterMV {


    private double MIN_DIST = 0.01;
    public ArrayList<ShapeletMV> findShapelets(MultivariateShapelet.ShapeletParams params, TimeSeriesInstances instances) {
        ArrayList<ShapeletMV> shapelets = new ArrayList<ShapeletMV>();
        double[][][] instancesArray = instances.toValueArray();
        int[] classesArray  = instances.getClassIndexes();
        ShapeletFactoryMV type = params.type.createShapeletType();
        ShapeletQualityMV quality = params.quality.createShapeletQuality(instancesArray,
                instances.getClassIndexes(),instances.getClassLabels(),instances.getClassCounts(),
                params.distance.createShapeletDistance());

        for (int index=0;index<instancesArray.length;index++){ // For each instance
         //   System.out.println("instance " + index);
            for (int channel=0;channel<instancesArray[index].length;channel++){
                ClusteringUtilities.zNormalise(instancesArray[index][channel]);
            }
            for (int shapeletSize=params.min;shapeletSize<=params.max;shapeletSize++) {  // For each shapelet size
           //     System.out.println("shapelet " + shapeletSize);

                ShapeletMV[] candidates = type.getShapeletsOverInstance(shapeletSize,index,instancesArray[index]);

                for (int candidate = 0 ; candidate < candidates.length; candidate++){
                   // System.out.println("candidate " + candidate + " of " + candidates.length);

                     if (isSimilar(shapelets, candidates[candidate],params.distance.createShapeletDistance())) continue;
                    double q = quality.calculate (candidates[candidate]);
                    candidates[candidate].setQuality(q);
                    shapelets.add(candidates[candidate]);
                }
            }
            Collections.sort(shapelets);
            if (shapelets.size()>params.k) shapelets.subList(params.k,shapelets.size()).clear();
          ///  System.out.println(Arrays.toString(shapelets.toArray()));

        }
        return shapelets;
    }

    private boolean isSimilar(ArrayList<ShapeletMV> shapelets, ShapeletMV candidate, ShapeletDistanceMV distance){

        for (ShapeletMV shapelet: shapelets){
            // System.out.println(this.distance.distance(shapelet.getData(), candidate.getData(),candidate.getLength()));
            if (distance.calculate(shapelet.toDoubleArray(), candidate.toDoubleArray(),candidate.getLength())<MIN_DIST)
                return true;
        }
        return false;
    }




}
