package tsml.classifiers.shapelet_based.functions;

import tsml.classifiers.shapelet_based.classifiers.MSTC;
import tsml.classifiers.shapelet_based.type.ShapeletDependentMV;
import tsml.data_containers.TimeSeriesInstance;

public class ShapeletFunctionsDependant implements ShapeletFunctions<ShapeletDependentMV> {

    @Override
    public ShapeletDependentMV[] getShapeletsOverInstance(int shapeletSize, int instanceIndex, double classIndex,TimeSeriesInstance instance) {
        int minLength = instance.getMinLength();
        ShapeletDependentMV[] candidates = new ShapeletDependentMV[minLength-shapeletSize];
        for (int seriesIndex=0;seriesIndex<minLength-shapeletSize;seriesIndex++){

            candidates[seriesIndex] = new ShapeletDependentMV(seriesIndex, shapeletSize, instanceIndex, classIndex, instance);

        }
        return candidates;
    }

    @Override
    public ShapeletDependentMV getRandomShapelet(int shapeletSize, int instanceIndex, double classIndex, TimeSeriesInstance instance) {
        int minLength = instance.getMinLength();
        return new ShapeletDependentMV(MSTC.RAND.nextInt(minLength-shapeletSize), shapeletSize ,instanceIndex, classIndex, instance);
    }

    @Override
    public double getDistanceFunction(ShapeletDependentMV shapelet1, ShapeletDependentMV shapelet2) {
        ShapeletDependentMV small,big;
        double sum = 0, min = Double.MAX_VALUE;
        if (shapelet1.getLength()>shapelet2.getLength()){
            small = shapelet2;
            big = shapelet1;
        }else{
            small = shapelet1;
            big = shapelet2;

        }

        for (int i=0;i<big.getLength()-small.getLength();i++){
            sum = 0;
            for (int j=0;j<small.getData().length;j++){
                for (int k=0;k<small.getData()[j].length;k++){
                    sum =+ (small.getData()[j][k]-big.getData()[j][k+i])*(small.getData()[j][k]-big.getData()[j][k+i]);
                }
            }
            if (min<sum){
                min = sum;
            }
        }


        return Math.sqrt(min);
    }





}
