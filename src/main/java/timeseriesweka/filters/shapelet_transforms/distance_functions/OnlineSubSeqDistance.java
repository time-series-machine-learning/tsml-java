/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.distance_functions;

import java.util.Arrays;
import timeseriesweka.filters.shapelet_transforms.Shapelet;
import weka.core.Instance;

/**
 *
 * @author raj09hxu
 */
public class OnlineSubSeqDistance extends SubSeqDistance {

    protected double[][] sortedIndices;

    @Override
    public void setShapelet(Shapelet shp)
    {
        super.setShapelet(shp);
        sortedIndices = sortIndexes(cand.getShapeletContent());
    }
    
    @Override
    public void setCandidate(Instance inst, int start, int length, int dim) {
        super.setCandidate(inst, start, length, dim);
        sortedIndices = sortIndexes(cand.getShapeletContent());
    }
    
    //we take in a start pos, but we also start from 0.
    @Override
    public double calculate(double[] timeSeries, int timeSeriesId) {
        DoubleWrapper sumPointer = new DoubleWrapper();
        DoubleWrapper sumsqPointer = new DoubleWrapper();

        //Generate initial subsequence 
        double[] subseq = new double[length];
        System.arraycopy(timeSeries, 0, subseq, 0, subseq.length);
        subseq = zNormalise(subseq, false, sumPointer, sumsqPointer);
        //Keep count of fundamental ops for experiment

        double sum = sumPointer.get();
        double sumsq = sumsqPointer.get();

        double bestDist = 0.0;

        double temp;
        
        int bestPos =0;
        //Compute initial distance
        for (int i = 0; i < length; i++) {
            temp = cand.getShapeletContent()[i] - subseq[i];
            bestDist = bestDist + (temp * temp);
        }

        double currentDist, start, end;
        // Scan through all possible subsequences of two
        for (int i = 1; i < timeSeries.length - length; i++) {
            //Update the running sums
            start = timeSeries[i - 1];
            end = timeSeries[i - 1 + length];
            
            //get rid of the start and add on the ends.
            sum = sum + end - start;
            sumsq = sumsq + (end * end) - (start * start);
            currentDist = calculateBestDistance(i, timeSeries, bestDist, sum, sumsq);  
            
            if (currentDist < bestDist) {
                bestDist = currentDist;
                bestPos = i;
            }
        }
        bestDist = (bestDist == 0.0) ? 0.0 : (1.0 / length * bestDist);
        
        return bestDist;
    }
    
        
    protected double calculateBestDistance(int i, double[] timeSeries, double bestDist, double sum, double sum2)
    {
        //Compute the stats for new series
        double mean = sum / length;

        //Get rid of rounding errors
        double stdv2 = (sum2 - (mean * mean * length)) / length;

        double stdv = (stdv2 < ROUNDING_ERROR_CORRECTION) ? 0.0 : Math.sqrt(stdv2);
                  
        
        //calculate the normalised distance between the series
        int j = 0;
        double currentDist = 0.0;
        double toAdd;
        int reordedIndex;
        double normalisedVal = 0.0;
        boolean dontStdv = (stdv == 0.0);

        while (j < length  && currentDist < bestDist)
        {
            //count ops
            incrementCount();
            
            reordedIndex = (int) sortedIndices[j][0];
            //if our stdv isn't done then make it 0.
            normalisedVal = dontStdv ? 0.0 : ((timeSeries[i + reordedIndex] - mean) / stdv);
            toAdd = cand.getShapeletContent()[reordedIndex] - normalisedVal;
            currentDist = currentDist + (toAdd * toAdd);
            j++;
        }
        

        return currentDist;
    }

    /**
     * Z-Normalise a time series
     *
     * @param input the input time series to be z-normalised
     * @param classValOn specify whether the time series includes a class value
     * (e.g. an full instance might, a candidate shapelet wouldn't)
     * @param sum
     * @param sum2
     * @return a z-normalised version of input
     */
    final double[] zNormalise(double[] input, boolean classValOn, DoubleWrapper sum, DoubleWrapper sum2) {
        double mean;
        double stdv;

        double classValPenalty = classValOn ? 1 : 0;

        double[] output = new double[input.length];
        double seriesTotal = 0;
        double seriesTotal2 = 0;

        for (int i = 0; i < input.length - classValPenalty; i++) {
            seriesTotal += input[i];
            seriesTotal2 += (input[i] * input[i]);
        }

        if (sum != null && sum2 != null) {
            sum.set(seriesTotal);
            sum2.set(seriesTotal2);
        }

        mean = seriesTotal / (input.length - classValPenalty);
        double num = (seriesTotal2 - (mean * mean * (input.length - classValPenalty))) / (input.length - classValPenalty);
        stdv = (num <= ROUNDING_ERROR_CORRECTION) ? 0.0 : Math.sqrt(num);

        for (int i = 0; i < input.length - classValPenalty; i++) {
            output[i] = (stdv == 0.0) ? 0.0 : (input[i] - mean) / stdv;
        }

        if (classValOn) {
            output[output.length - 1] = input[input.length - 1];
        }
        
        return output;
    }
    
    protected static class DoubleWrapper {

        private double d;

        public DoubleWrapper() {
            d = 0.0;
        }

        public DoubleWrapper(double d) {
            this.d = d;
        }

        public void set(double d) {
            this.d = d;
        }

        public double get() {
            return d;
        }
    }
        
    /**
     * A method to sort the array indeces according to their corresponding
     * values
     *
     * @param series a time series, which indeces need to be sorted
     * @return
     */
    public static double[][] sortIndexes(double[] series)
    {
        //Create an boxed array of values with corresponding indexes
        double[][] sortedSeries = new double[series.length][2];
        for (int i = 0; i < series.length; i++)
        {
            sortedSeries[i][0] = i;
            sortedSeries[i][1] = Math.abs(series[i]);
        }

        
        //this is the lamda expression.
        //Arrays.sort(sortedSeries, (double[] o1, double[] o2) -> Double.compare(o1[1],o2[1]));
        Arrays.sort(sortedSeries, (double[] o1, double[] o2) -> Double.compare(o2[1], o1[1]));

        return sortedSeries;
    }
}
