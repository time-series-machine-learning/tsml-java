package tsml.classifiers.shapelet_based.type;

import java.util.ArrayList;

public class ShapeletFactoryIndependent implements ShapeletFactoryMV {
    @Override
    public ShapeletMV[] getShapeletsOverInstance(int shapeletSize, int instanceIndex, double[][] instance) {

        ArrayList<ShapeletMV> candidates = new ArrayList<ShapeletMV>();
        for(int channelIndex=0;channelIndex<instance.length;channelIndex++) {
            for (int seriesIndex = 0; seriesIndex < instance[channelIndex].length - shapeletSize; seriesIndex++) {
                candidates.add(new ShapeletIndependentMV(seriesIndex, shapeletSize, instanceIndex, channelIndex, instance));
            }
        }
        return candidates.toArray(new ShapeletMV[0]);

    }
}
