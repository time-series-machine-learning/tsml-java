package tsml.classifiers.shapelet_based.dev.functions;

import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.classifiers.shapelet_based.dev.distances.ShapeletDistanceImprovedOnline;
import tsml.classifiers.shapelet_based.dev.type.ShapeletDimensionDependentMV;
import tsml.data_containers.TimeSeriesInstance;

public class ShapeletFunctionsDimDependant implements ShapeletFunctions<ShapeletDimensionDependentMV> {


    ShapeletDistanceImprovedOnline distance = new ShapeletDistanceImprovedOnline();

    @Override
    public ShapeletDimensionDependentMV[] getShapeletsOverInstance(int shapeletSize, int instanceIndex, double classIndex, TimeSeriesInstance instance) {
        int minLength = instance.getMinLength();
        int minDimension = 1;
        int maxDimensions = instance.getNumDimensions();
        int numDimension = minDimension + MSTC.RAND.nextInt(maxDimensions-minDimension);
        int[] dimensionIndex = new int[numDimension];
        for (int di=0;di<numDimension;di++){
            dimensionIndex[di] = MSTC.RAND.nextInt(maxDimensions);
        }
        ShapeletDimensionDependentMV[] candidates = new ShapeletDimensionDependentMV[minLength-shapeletSize];
        for (int seriesIndex=0;seriesIndex<minLength-shapeletSize;seriesIndex++){

            candidates[seriesIndex] = new ShapeletDimensionDependentMV(seriesIndex, shapeletSize, instanceIndex, classIndex, dimensionIndex, instance);

        }
        return candidates;
    }

    @Override
    public ShapeletDimensionDependentMV getRandomShapelet(int shapeletSize, int instanceIndex, double classIndex, TimeSeriesInstance instance) {
        int minLength = instance.getMinLength();
        int minDimension = 1;
        int maxDimensions = instance.getNumDimensions();
        int numDimension = minDimension + MSTC.RAND.nextInt(maxDimensions-minDimension);
        int[] dimensionIndex = new int[numDimension];
        for (int di=0;di<numDimension;di++){
            dimensionIndex[di] = MSTC.RAND.nextInt(maxDimensions);
        }

        return new ShapeletDimensionDependentMV(MSTC.RAND.nextInt(minLength-shapeletSize), shapeletSize ,instanceIndex, classIndex, dimensionIndex, instance);
    }


    public boolean selfSimilarity(ShapeletDimensionDependentMV shapelet, ShapeletDimensionDependentMV candidate) {
        // check whether they're the same dimension or not.
        if ( candidate.getInstanceIndex() == shapelet.getInstanceIndex()) {
            if (candidate.getStart() >= shapelet.getStart()
                    && candidate.getStart() < shapelet.getStart() + shapelet.getLength()) { // candidate starts within
                // exisiting shapelet
                return true;
            }
            if (shapelet.getStart() >= candidate.getStart()
                    && shapelet.getStart() < candidate.getStart() + candidate.getLength()) {
                return true;
            }
        }
        return false;
    }

    public double sDist(ShapeletDimensionDependentMV shapelet, TimeSeriesInstance instance) {

        double[][] timeSeries = instance.toValueArray();
        double[][] shapeletData = shapelet.getData();
        double dist = 0;

        for (int i=0;i<shapeletData.length;i++){
            dist += distance.calculate(shapeletData[i],timeSeries[shapelet.getDimension(i)]);
        }
        return dist;
    }
}
