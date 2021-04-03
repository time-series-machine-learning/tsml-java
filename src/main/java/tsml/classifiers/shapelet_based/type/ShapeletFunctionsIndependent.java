package tsml.classifiers.shapelet_based.type;

import tsml.classifiers.shapelet_based.classifiers.MultivariateShapelet;
import tsml.data_containers.TimeSeriesInstance;

import java.util.ArrayList;

public class ShapeletFunctionsIndependent implements ShapeletFunctions<ShapeletIndependentMV> {
    @Override
    public ShapeletIndependentMV[] getShapeletsOverInstance(int shapeletSize, int instanceIndex, double classIndex, TimeSeriesInstance instance) {

        ArrayList<ShapeletIndependentMV> candidates = new ArrayList<ShapeletIndependentMV>();
        for(int channelIndex=0;channelIndex<instance.getNumDimensions();channelIndex++) {
            for (int seriesIndex = 0; seriesIndex < instance.get(channelIndex).getSeriesLength() - shapeletSize; seriesIndex++) {
                candidates.add(new ShapeletIndependentMV(seriesIndex, shapeletSize, instanceIndex, classIndex, channelIndex, instance.toValueArray()));
            }
        }
        return candidates.toArray(new ShapeletIndependentMV[0]);

    }

    @Override
    public ShapeletIndependentMV getRandomShapelet(int shapeletSize, int instanceIndex, double classIndex, TimeSeriesInstance instance) {
        int channelIndex = MultivariateShapelet.RAND.nextInt(instance.getNumDimensions());
        return new ShapeletIndependentMV(MultivariateShapelet.RAND.nextInt(instance.get(channelIndex).getSeriesLength()-shapeletSize),
                shapeletSize, instanceIndex, classIndex, channelIndex, instance.toValueArray());
    }

    @Override
    public double getDistanceFunction(ShapeletIndependentMV shapelet1, ShapeletIndependentMV shapelet2) {
        ShapeletIndependentMV small,big;
        double sum = 0, min = Double.MAX_VALUE;
        if (shapelet1.length>shapelet2.length){
            small = shapelet2;
            big = shapelet1;
        }else{
            small = shapelet1;
            big = shapelet2;

        }

        for (int i=0;i<big.length-small.length;i++){
            sum = 0;
            for (int j=0;j<small.length;j++){
                sum =+ (small.data[j]-big.data[j+i])*(small.data[j]-big.data[j+i]);
            }
            if (min<sum){
                min = sum;
            }
        }


        return Math.sqrt(min);
    }
}
