package tsml.classifiers.shapelet_based.dev.functions;

import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.classifiers.shapelet_based.dev.distances.ShapeletDistanceImprovedOnline;
import tsml.classifiers.shapelet_based.dev.type.ShapeletDependentMV;
import tsml.data_containers.TimeSeriesInstance;

public class ShapeletFunctionsDependant implements ShapeletFunctions<ShapeletDependentMV> {


    ShapeletDistanceImprovedOnline distance = new ShapeletDistanceImprovedOnline();

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
    public double distance(ShapeletDependentMV shapelet1, ShapeletDependentMV shapelet2) {
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

    public boolean selfSimilarity(ShapeletDependentMV shapelet, ShapeletDependentMV candidate) {
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
/*
    public double sDist(ShapeletDependentMV shapelet,  TimeSeriesInstance instance) {
        double sum = 0;
        double temp = 0;
        for(int channel=0;channel< instance.getNumDimensions(); channel++){

            for (int index = 0; index < shapelet.getLength(); index++)
            {
                temp = shapelet.getData()[channel][index] - instance.get(channel).get(index);
                sum = sum + (temp * temp);
            }
        }
        return sum/shapelet.getData().length;
    }*/
    public double sDist(ShapeletDependentMV shapelet, TimeSeriesInstance instance) {

        double[][] timeSeries = instance.toValueArray();
        double[][] shapeletData = shapelet.getData();
        double dist = 0;

        for (int i=0;i<shapeletData.length;i++){
            dist += distance.calculate(shapeletData[i],timeSeries[i]);
        }
        return dist;
    }
}
