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
import tsml.transformers.shapelet_tools.ShapeletCandidate;
import static utilities.multivariate_tools.MultivariateInstanceTools.convertMultiInstanceToArrays;
import static utilities.multivariate_tools.MultivariateInstanceTools.splitMultivariateInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author raj09hxu
 */
public class MultivariateDistance extends ShapeletDistance implements Serializable{
    protected int numChannels;
    protected int seriesLength;
    
    @Override
    public void init(Instances data)
    {
        count =0;
        numChannels = utilities.multivariate_tools.MultivariateInstanceTools.numDimensions(data);
        seriesLength = utilities.multivariate_tools.MultivariateInstanceTools.channelLength(data);
        
    }
    
    protected double[][] candidateArray2;
    
    @Override
    public void setCandidate(Instance inst, int start, int len, int dim) {
        //extract shapelet and nomrliase.
        cand = new ShapeletCandidate(numChannels);
        startPos = start;
        length = len;
        
        //only call to double array when we've changed series.
        if(candidateInst==null || candidateInst != inst){
            candidateArray2 = convertMultiInstanceToArrays(splitMultivariateInstance(inst));
            candidateInst = inst;
        }
        
        for(int i=0; i< numChannels; i++){
            double[] temp = new double[length];
            //copy the data from the whole series into a candidate.
            System.arraycopy(candidateArray2[i], start, temp, 0, length);
            temp = seriesRescaler.rescaleSeries(temp, false); //normalise each series.
            cand.setShapeletContent(i, temp);
        } 
    } 
    
    @Override
    public void setCandidate(TimeSeriesInstance inst, int start, int len, int dim) {
        //extract shapelet and nomrliase.
        cand = new ShapeletCandidate(numChannels);
        startPos = start;
        length = len;
        dimension =  dim;

        if(candidateInst==null || candidateInst != inst){
            candidateArray2 = inst.toValueArray();
            candidateTSInst = inst;
        }
        
        for(int i=0; i< numChannels; i++){
            double[] temp = inst.get(dimension).getSlidingWindowArray(start, start+length);
            temp = seriesRescaler.rescaleSeries(temp, false); //normalise each series.
            cand.setShapeletContent(i, temp);
        } 
    }
}
