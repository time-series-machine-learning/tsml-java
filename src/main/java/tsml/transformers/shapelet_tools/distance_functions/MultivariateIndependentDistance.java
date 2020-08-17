/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package tsml.transformers.shapelet_tools.distance_functions;

import java.io.Serializable;

import tsml.data_containers.TimeSeriesInstance;
import tsml.transformers.shapelet_tools.Shapelet;
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

        //calculate the minimum distance for each channel, and then average.
    @Override
    public double calculate(TimeSeriesInstance timeSeries, int timeSeriesId){
        double[][] data = timeSeries.toValueArray();
        double cumulative_distance=0;
        for(int i=0; i< data.length; i++){
            cumulative_distance += calculate(cand.getShapeletContent(i), data[i]);
        }
        
        //return candidate back into the holder and the instance it comes from.
        return cumulative_distance;
    }
    
    @Override
    public double distanceToShapelet(Shapelet otherShapelet){
        double sum = 0;
        //loop through all the channels.
        for(int j=0; j< numChannels; j++){
            sum += super.distanceToShapelet(otherShapelet);
        }
        
        double dist = (sum == 0.0) ? 0.0 : (1.0 / length * sum);
        return dist;
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
            
            subseq = seriesRescaler.rescaleSeries(subseq, false); // Z-NORM HERE
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
