/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.distance_functions;

import java.io.Serializable;
import java.util.Arrays;
import weka.core.Instance;

/**
 *
 * @author raj09hxu
 */
public class MultivariateIndependentDistance extends MultivariateDistance implements Serializable{   
    
    //calculate the minimum distance for each channel, and then average.
    @Override
    public double calculate(Instance timeSeries, int timeSeriesId){
        Instance[] channel = utilities.multivariate_tools.MultivariateInstanceTools.splitMultivariateInstance(timeSeries);
        double cumulative_distance=0;
        for(int i=0; i< channel.length; i++){
            cumulative_distance += calculate(cand.getShapeletContent(i), channel[i].toDoubleArray());
        }
        
        //return candidate back into the holder and the instance it comes from.
        return cumulative_distance;
    }
    
    //we take in a start pos, but we also start from 0.
    public double calculate(double[] shape, double[] timeSeries) 
    {
        double bestSum = Double.MAX_VALUE;
        double sum;
        double[] subseq;
        double temp;
        
        //m-l+1
        //multivariate instances that are split dont have a class value on them.
        for (int i = 0; i < timeSeries.length - length + 1; i++)
        {
            sum = 0;
            // get subsequence of two that is the same lengh as one
            subseq = new double[length];
            System.arraycopy(timeSeries, i, subseq, 0, length);
            
            subseq = zNormalise(subseq, false); // Z-NORM HERE
            for (int j = 0; j < length; j++)
            {
                //count ops
                incrementCount();
                temp = (shape[j] - subseq[j]);
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
}
