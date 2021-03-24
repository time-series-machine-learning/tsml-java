package tsml.classifiers.shapelet_based.type;

import tsml.classifiers.shapelet_based.classifiers.MultivariateShapelet;

import java.util.ArrayList;

public class ShapeletFactoryIndependent implements ShapeletFactoryMV {
    @Override
    public ShapeletMV[] getShapeletsOverInstance(int shapeletSize, int instanceIndex, double classIndex, double[][] instance) {

        ArrayList<ShapeletMV> candidates = new ArrayList<ShapeletMV>();
        for(int channelIndex=0;channelIndex<instance.length;channelIndex++) {
            for (int seriesIndex = 0; seriesIndex < instance[channelIndex].length - shapeletSize; seriesIndex++) {
                candidates.add(new ShapeletIndependentMV(seriesIndex, shapeletSize, instanceIndex, classIndex, channelIndex, instance));
            }
        }
        return candidates.toArray(new ShapeletMV[0]);

    }

    @Override
    public ShapeletMV getRandomShapelet(int shapeletSize, int instanceIndex, double classIndex, double[][] instance) {
        int channelIndex = MultivariateShapelet.RAND.nextInt(instance.length);
//        System.out.println(instance[channelIndex].length + " " + shapeletSize);
        return new ShapeletIndependentMV(MultivariateShapelet.RAND.nextInt(instance[channelIndex].length-shapeletSize), shapeletSize, instanceIndex, classIndex, channelIndex, instance);
    }
}
