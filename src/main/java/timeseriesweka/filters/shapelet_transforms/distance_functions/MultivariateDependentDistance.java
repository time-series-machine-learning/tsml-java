/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.distance_functions;

import java.io.Serializable;
import java.util.Arrays;
import timeseriesweka.filters.shapelet_transforms.ShapeletCandidate;
import static utilities.multivariate_tools.MultivariateInstanceTools.convertMultiInstanceToArrays;
import static utilities.multivariate_tools.MultivariateInstanceTools.splitMultivariateInstance;
import weka.core.Instance;

/**
 *
 * @author raj09hxu
 */
public class MultivariateDependentDistance extends MultivariateDistance implements Serializable{
    
    @Override
    public double calculate(Instance inst, int timeSeriesId){
        return calculate(convertMultiInstanceToArrays(splitMultivariateInstance(inst)), timeSeriesId);
    }
    
    //we take in a start pos, but we also start from 0.
    public double calculate(double[][] timeSeries, int timeSeriesId) 
    {
        double bestSum = Double.MAX_VALUE;
        double sum;
        double[] subseq;
        double temp;
        
        //m-l+1
        //multivariate instances that are split dont have a class value on them.
        for (int i = 0; i < seriesLength - length + 1; i++)
        {
            sum = 0;
            //loop through all the channels.
            for(int j=0; j< numChannels; j++){
                //copy a section of one of the series from the channel.
                subseq = new double[length];
                System.arraycopy(timeSeries[j], i, subseq, 0, length);
                subseq = zNormalise(subseq, false); // Z-NORM HERE
                for (int k = 0; k < length; k++)
                {
                    //count ops
                    incrementCount();
                    temp = (cand.getShapeletContent(j)[k] - subseq[k]);
                    sum = sum + (temp * temp);
                }
            }

            if (sum < bestSum)
            {
                bestSum = sum;
                //System.out.println(i);
            }
        }

        double dist = (bestSum == 0.0) ? 0.0 : (1.0 / length * bestSum);
        return dist;
    }
    
}
