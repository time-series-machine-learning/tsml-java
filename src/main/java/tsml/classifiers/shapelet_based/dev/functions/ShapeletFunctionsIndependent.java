package tsml.classifiers.shapelet_based.dev.functions;

import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.classifiers.shapelet_based.dev.distances.ShapeletDistanceImprovedOnline;
import tsml.classifiers.shapelet_based.dev.type.ShapeletIndependentMV;
import tsml.data_containers.TimeSeriesInstance;

import java.util.ArrayList;

public class ShapeletFunctionsIndependent implements ShapeletFunctions<ShapeletIndependentMV> {

    ShapeletDistanceImprovedOnline distance = new ShapeletDistanceImprovedOnline();

    @Override
    public ShapeletIndependentMV[] getShapeletsOverInstance(int shapeletSize, int instanceIndex, double classIndex, TimeSeriesInstance instance) {

        ArrayList<ShapeletIndependentMV> candidates = new ArrayList<ShapeletIndependentMV>();
        for(int channelIndex=0;channelIndex<instance.getNumDimensions();channelIndex++) {
            for (int seriesIndex = 0; seriesIndex < instance.get(channelIndex).getSeriesLength() - shapeletSize; seriesIndex++) {
                candidates.add(new ShapeletIndependentMV(seriesIndex, shapeletSize, instanceIndex, classIndex, channelIndex, instance));
            }
        }
        return candidates.toArray(new ShapeletIndependentMV[0]);

    }

    @Override
    public ShapeletIndependentMV getRandomShapelet(int shapeletSize, int instanceIndex, double classIndex, TimeSeriesInstance instance) {
        int channelIndex = MSTC.RAND.nextInt(instance.getNumDimensions());
        return new ShapeletIndependentMV(MSTC.RAND.nextInt(instance.get(channelIndex).getSeriesLength()-shapeletSize),
                shapeletSize, instanceIndex, classIndex, channelIndex, instance);
    }



    public boolean selfSimilarity(ShapeletIndependentMV shapelet, ShapeletIndependentMV candidate) {
        // check whether they're the same dimension or not.
        if (candidate.getSeriesIndex() == shapelet.getSeriesIndex() && candidate.getInstanceIndex() == shapelet.getInstanceIndex()) {
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

    public double sDist(ShapeletIndependentMV shapelet, TimeSeriesInstance instance) {

        double[] timeSeries = instance.getHSliceArray(shapelet.getSeriesIndex());
        double[] shapeletData = shapelet.getData();
        return distance.calculate(shapeletData,timeSeries);
    }

}
