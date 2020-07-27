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
    
    @Override
    public double calculate(TimeSeriesInstance timeSeries, int timeSeriesId){
        return calculate(timeSeries.toValueArray(), timeSeriesId);
    }

    @Override
    public double distanceToShapelet(Shapelet otherShapelet){
        double sum = 0;
        double temp;
        //loop through all the channels.
        for(int j=0; j< numChannels; j++){
            for (int k = 0; k < length; k++)
            {
                temp = (cand.getShapeletContent(j)[k] - otherShapelet.getContent().getShapeletContent(j)[k]);
                sum = sum + (temp * temp);
            }
        }
        
        double dist = (sum == 0.0) ? 0.0 : (1.0 / length * sum);
        return dist;
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
                subseq = seriesRescaler.rescaleSeries(subseq, false); // Z-NORM HERE
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
