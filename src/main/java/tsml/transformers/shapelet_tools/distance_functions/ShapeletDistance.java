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

import weka.core.Instances;
import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.TimeSeriesInstances;
import tsml.transformers.shapelet_tools.Shapelet;
import tsml.transformers.shapelet_tools.ShapeletCandidate;
import utilities.rescalers.SeriesRescaler;
import utilities.rescalers.ZNormalisation;
import weka.core.Instance;

/**
 *
 * @author Aaron, who does not like commenting code
 *
 * this implements the basic functionality of sDist in various ways
 */
public class ShapeletDistance implements Serializable{

//Each enum represents a class in this package
    public enum DistanceType{
        NORMAL,         // Standard full scan (with early abandon) is this class
        ONLINE,         // Faster normalisation for extracted subsequences (avoids a call to RescalerType
        IMPROVED_ONLINE,     // online calculation with variable start and bespoke abandon SEE DAWAK PAPER
        CACHED,         // Mueen's pre-cached version see Logical Shapelets paper
        ONLINE_CACHED,  // Untested hybrid between online and caching, unpublished, TO REMOVE
    // These three are for multivariate
        DEPENDENT,      // Uses pointwise distance over dimensions
        INDEPENDENT,    // Uses the average over individual dimensions
        DIMENSION       // Aaron's weird one: slide single series over all dimensions
    };

//And this?
    public enum RescalerType{NONE, STANDARDISATION, NORMALISATION};
//Should the seriesRescaler not depend on RescalerType??
    public SeriesRescaler seriesRescaler = new ZNormalisation();
    
    protected TimeSeriesInstance candidateTSInst;
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

    public void init(TimeSeriesInstances data)
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
        cand.setShapeletContent(seriesRescaler.rescaleSeries(cand.getShapeletContent(), false));
    }

    public void setCandidate(TimeSeriesInstance inst, int start, int len, int dim) {
        //extract shapelet and nomrliase.
        cand = new ShapeletCandidate();
        startPos = start;
        length = len;
        dimension =  dim;

        //only call to double array when we've changed series.
        if(candidateTSInst==null || candidateTSInst != inst){
            candidateArray = inst.get(dimension).toValueArray();
            candidateTSInst = inst;
        }
        
        double[] temp = inst.get(dimension).getSlidingWindowArray(start, start+length);
        cand.setShapeletContent(temp);
        
        // znorm candidate here so it's only done once, rather than in each distance calculation
        cand.setShapeletContent(seriesRescaler.rescaleSeries(cand.getShapeletContent(), false));
    }
    
    public void setSeries(int srsId) {
        seriesId = srsId;
    }
    
    public double calculate(Instance timeSeries, int timeSeriesId){
        return calculate(timeSeries.toDoubleArray(), timeSeriesId);
    }

    public double calculate(TimeSeriesInstance timeSeriesInstance, int timeSeriesId) {
		return calculate(timeSeriesInstance.get(dimension).toValueArray(), timeSeriesId);
	}
         
    public double distanceToShapelet(Shapelet otherShapelet){
        
        double temp;
        double sum = 0;
        for (int j = 0; j < length; j++)
        {
            temp = (cand.getShapeletContent()[j] - otherShapelet.getContent().getShapeletContent()[j]);
            sum = sum + (temp * temp);
        }
        double dist = (sum == 0.0) ? 0.0 : (1.0 / length * sum);
        return dist;
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

            subseq = seriesRescaler.rescaleSeries(subseq, false); // Z-NORM HERE

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


}
