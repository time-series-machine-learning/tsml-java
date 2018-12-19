/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.distance_functions;

import java.io.Serializable;
import java.util.Arrays;
import weka.core.Instances;
import timeseriesweka.filters.shapelet_transforms.Shapelet;
import timeseriesweka.filters.shapelet_transforms.ShapeletCandidate;
import weka.core.Instance;

/**
 *
 * @author Aaron
 */
public class SubSeqDistance implements Serializable{
       
    public enum DistanceType{NORMAL, ONLINE, IMP_ONLINE, CACHED, ONLINE_CACHED, DEPENDENT, INDEPENDENT, DIMENSION};
    
    public static final double ROUNDING_ERROR_CORRECTION = 0.000000000000001;
    
    protected Instance candidateInst;
    protected double[] candidateArray;
    
    protected Shapelet shapelet;
    protected ShapeletCandidate cand;
    protected int      seriesId;
    protected int      startPos;
    protected int      length;
    protected int      dimension;
    
    protected long count;
    
    public void init(Instances data)
    {
        count =0;
    }
    
    final void incrementCount(){ count++;}
    
    public long getCount() {return count;}
    
    public ShapeletCandidate getCandidate(){
        return cand;
    }
    
    public void setShapelet(Shapelet shp) {
        shapelet = shp;
        startPos = shp.startPos;
        cand = shp.getContent();
        length = shp.getLength();
        dimension = shp.getDimension();
    }
    
    public void setCandidate(Instance inst, int start, int len, int dim) {
        //extract shapelet and nomrliase.
        cand = new ShapeletCandidate();
        startPos = start;
        length = len;
        dimension =  dim;

        //only call to double array when we've changed series.
        if(candidateInst==null || candidateInst != inst){
            candidateArray = inst.toDoubleArray();
            candidateInst = inst;
        }
        
        double[] temp = new double[length];
        //copy the data from the whole series into a candidate.
        System.arraycopy(candidateArray, start, temp, 0, length);
        cand.setShapeletContent(temp);
        
        // znorm candidate here so it's only done once, rather than in each distance calculation
        cand.setShapeletContent(zNormalise(cand.getShapeletContent(), false));
    }
    
    public void setSeries(int srsId) {
        seriesId = srsId;
    }
    
    public double calculate(Instance timeSeries, int timeSeriesId){
        return calculate(timeSeries.toDoubleArray(), timeSeriesId);
    }
           
    //we take in a start pos, but we also start from 0.
    public double calculate(double[] timeSeries, int timeSeriesId) 
    {
        double bestSum = Double.MAX_VALUE;
        double sum;
        double[] subseq;
        double temp;
        
        for (int i = 0; i < timeSeries.length - length; i++)
        {
            sum = 0;
            // get subsequence of two that is the same lengh as one
            subseq = new double[length];
            System.arraycopy(timeSeries, i, subseq, 0, length);

            subseq = zNormalise(subseq, false); // Z-NORM HERE

            for (int j = 0; j < length; j++)
            {
                //count ops
                count++;
                temp = (cand.getShapeletContent()[j] - subseq[j]);
                sum = sum + (temp * temp);
            }
            
            if (sum < bestSum)
            {
                bestSum = sum;
            }
        }

        double dist = (bestSum == 0.0) ? 0.0 : (1.0 / length * bestSum);
        return dist;
    }

     /**
     * Z-Normalise a time series
     *
     * @param input the input time series to be z-normalised
     * @param classValOn specify whether the time series includes a class value
     * (e.g. an full instance might, a candidate shapelet wouldn't)
     * @return a z-normalised version of input
     */
    final public double[] zNormalise(double[] input, boolean classValOn)
    {
        double mean;
        double stdv;

        int classValPenalty = classValOn ? 1 : 0;
        int inputLength = input.length - classValPenalty;

        double[] output = new double[input.length];
        double seriesTotal = 0;
        for (int i = 0; i < inputLength; i++)
        {
            seriesTotal += input[i];
        }

        mean = seriesTotal / (double) inputLength;
        stdv = 0;
        double temp;
        for (int i = 0; i < inputLength; i++)
        {
            temp = (input[i] - mean);
            stdv += temp * temp;
        }

        stdv /= (double) inputLength;

        // if the variance is less than the error correction, just set it to 0, else calc stdv.
        stdv = (stdv < ROUNDING_ERROR_CORRECTION) ? 0.0 : Math.sqrt(stdv);
        
        //System.out.println("mean "+ mean);
        //System.out.println("stdv "+stdv);
        
        for (int i = 0; i < inputLength; i++)
        {
            //if the stdv is 0 then set to 0, else normalise.
            output[i] = (stdv == 0.0) ? 0.0 : ((input[i] - mean) / stdv);
        }

        if (classValOn)
        {
            output[output.length - 1] = input[input.length - 1];
        }

        return output;
    }
}
