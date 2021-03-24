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


    public ArrayList<ShapeletMV> findShapelets(MultivariateShapelet.ShapeletParams params, TimeSeriesInstances instances) {
        ArrayList<ShapeletMV> shapelets = new ArrayList<ShapeletMV>();
        double[][][] instancesArray = instances.toValueArray();
        int[] classesArray  = instances.getClassIndexes();
        ShapeletFactoryMV type = params.type.createShapeletType();
        ShapeletQualityMV quality = params.quality.createShapeletQuality(instancesArray,
                instances.getClassIndexes(),instances.getClassLabels(),instances.getClassCounts(),
                params.distance.createShapeletDistance());

        for (int index=0;index<instancesArray.length;index++){ // For each instance
            for (int channel=0;channel<instancesArray[index].length;channel++){
                ClusteringUtilities.zNormalise(instancesArray[index][channel]);
            }
            for (int shapeletSize=params.min;shapeletSize<=params.max;shapeletSize++) {  // For each shapelet size

                ShapeletMV[] candidates = type.getShapeletsOverInstance(shapeletSize,index,classesArray[index],instancesArray[index]);

                for (int candidate = 0 ; candidate < candidates.length; candidate++){

                     if (isSimilar(shapelets, candidates[candidate],params.distance.createShapeletDistance(),params.minDist)) continue;
                    double q = quality.calculate (candidates[candidate]);
                    candidates[candidate].setQuality(q);
                    shapelets.add(candidates[candidate]);
                }
            }
            Collections.sort(shapelets);
            if (shapelets.size()>params.k) shapelets.subList(params.k,shapelets.size()).clear();

        }
        return shapelets;
    }

    private  long time;
    @Override
    public void setTrainTimeLimit(long time) {
        this.time = time;
    }




}
