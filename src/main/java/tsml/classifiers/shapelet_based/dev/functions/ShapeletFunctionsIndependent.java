package tsml.classifiers.shapelet_based.dev.functions;

import tsml.classifiers.shapelet_based.dev.classifiers.MSTC;
import tsml.classifiers.shapelet_based.dev.type.ShapeletIndependentMV;
import tsml.data_containers.TimeSeriesInstance;

import java.util.ArrayList;

public class ShapeletFunctionsIndependent implements ShapeletFunctions<ShapeletIndependentMV> {
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

    @Override
    public double getDistanceFunction(ShapeletIndependentMV shapelet1, ShapeletIndependentMV shapelet2) {
        ShapeletIndependentMV small,big;
        double sum = 0, min = Double.MAX_VALUE;
        if (shapelet1.getLength()>shapelet2.getLength()){
            small = shapelet2;
            big = shapelet1;
        }else{
            small = shapelet1;
            big = shapelet2;

        }

        for (int i=0;i<big.getLength()-small.getLength()+1;i++){
            sum = 0;
            for (int j=0;j<small.getLength();j++){
                sum =+ (small.getData()[j]-big.getData()[j+i])*(small.getData()[j]-big.getData()[j+i]);
            }
            if (sum<min){
                min = sum;
            }
        }


        return Math.sqrt(min);
  //  return distance(shapelet1.getData(),shapelet2.getData(),1);
    }

    public final double distance(double[] first, double[] second, double cutoff) {


        double minDist;
        boolean tooBig;

        int n = first.length - 1;
        int m = second.length - 1;
        /*  Parameter 0<=r<=1. 0 == no warp, 1 == full warp
         generalised for variable window size
         * */
        int windowSize = 1;
//Extra memory than required, could limit to windowsize,
//        but avoids having to recreate during CV
//for varying window sizes
        double[][] matrixD = new double[n][m];

        /*
         //Set boundary elements to max.
         */
        int start, end;
        for (int i = 0; i < n; i++) {
            start = windowSize < i ? i - windowSize : 0;
            end = i + windowSize + 1 < m ? i + windowSize + 1 : m;
            for (int j = start; j < end; j++) {
                matrixD[i][j] = Double.MAX_VALUE;
            }
        }
        matrixD[0][0] = (first[0] - second[0]) * (first[0] - second[0]);
//a is the longer series.
//Base cases for warping 0 to all with max interval	r
//Warp first.value(0] onto all second.value(1]...second.value(r+1]
        for (int j = 1; j < windowSize && j < m; j++) {
            matrixD[0][j] = matrixD[0][j - 1] + (first[0] - second[j]) * (first[0] - second[j]);
        }

//	Warp second.value(0] onto all first.value(1]...first.value(r+1]
        for (int i = 1; i < windowSize && i < n; i++) {
            matrixD[i][0] = matrixD[i - 1][0] + (first[i] - second[0]) * (first[i] - second[0]);
        }
//Warp the rest,
        for (int i = 1; i < n; i++) {
            tooBig = true;
            start = windowSize < i ? i - windowSize + 1 : 1;
            end = i + windowSize < m ? i + windowSize : m;
            for (int j = start; j < end; j++) {
                minDist = matrixD[i][j - 1];
                if (matrixD[i - 1][j] < minDist) {
                    minDist = matrixD[i - 1][j];
                }
                if (matrixD[i - 1][j - 1] < minDist) {
                    minDist = matrixD[i - 1][j - 1];
                }
                matrixD[i][j] = minDist + (first[i] - second[j]) * (first[i] - second[j]);
                if (tooBig && matrixD[i][j] < cutoff) {
                    tooBig = false;
                }
            }
            //Early abandon
            if (tooBig) {
                return Double.MAX_VALUE;
            }
        }
//Find the minimum distance at the end points, within the warping window.
        return matrixD[n - 1][m - 1];
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

    public double sDist(int start, ShapeletIndependentMV shapelet, TimeSeriesInstance instance) {
        double sum = 0;
        double temp = 0;
        double a,b,ab;
        for (int i = 0; i < shapelet.getLength(); i++)
        {
            temp = shapelet.getData()[i] - instance.get(shapelet.getSeriesIndex()).get(start+i);

            sum = sum + (temp * temp);
  //          sum = sum + (shapelet.getData()[i] * instance.get(shapelet.getSeriesIndex()).get(start+i));
        }
        return Math.sqrt(sum);
//        return distance(shapelet.getData(),instance.get(0).getVSliceArray(start,start+shapelet.getLength()-1),1);
    }

}
