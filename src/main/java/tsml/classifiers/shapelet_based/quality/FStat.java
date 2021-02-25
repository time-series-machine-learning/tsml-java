package tsml.classifiers.shapelet_based.quality;

import tsml.classifiers.shapelet_based.type.ShapeletMV;
import tsml.classifiers.shapelet_based.distances.ShapeletDistanceMV;

public class FStat extends ShapeletQualityMV{

    public FStat(double[][][] instancesArray, int[] classIndexes, String[] classNames, int[] classCounts, ShapeletDistanceMV distance) {
        super(instancesArray, classIndexes, classNames, classCounts, distance);
    }

    @Override
    public double calculate(ShapeletMV candidate){
        return 0;
    }
}
