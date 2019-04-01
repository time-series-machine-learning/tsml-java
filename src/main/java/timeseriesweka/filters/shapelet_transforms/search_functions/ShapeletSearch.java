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
package timeseriesweka.filters.shapelet_transforms.search_functions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import weka.core.Instance;
import weka.core.Instances;
import timeseriesweka.filters.shapelet_transforms.Shapelet;
import static utilities.multivariate_tools.MultivariateInstanceTools.channelLength;
/**
 *
 * @author raj09hxu
 */
public class ShapeletSearch implements Serializable{
    
    public enum SearchType {FULL, FS, GENETIC, RANDOM, LOCAL, MAGNIFY, TIMED_RANDOM, SKIPPING, TABU, REFINED_RANDOM, IMP_RANDOM, SUBSAMPLE_RANDOM, SKEWED};
    

    public interface ProcessCandidate{
        public default Shapelet process(Instance candidate, int start, int length) {return process(candidate, start, length, 0);}
        public Shapelet process(Instance candidate, int start, int length, int dimension);
    }
    
    ArrayList<String> shapeletsVisited = new ArrayList<>();
    int seriesCount;
    
    public ArrayList<String> getShapeletsVisited() {
        return shapeletsVisited;
    }
    
    protected Comparator<Shapelet> comparator;
    
    public void setComparator(Comparator<Shapelet> comp){
        comparator = comp;
    }
    
    protected int seriesLength;
    protected int minShapeletLength;
    protected int maxShapeletLength;
    
    protected int numDimensions;
    
    protected int lengthIncrement = 1;
    protected int positionIncrement = 1;
    
    protected Instances inputData;
    
    transient protected ShapeletSearchOptions options;
    
    protected ShapeletSearch(ShapeletSearchOptions ops){
        options = ops;
        
        minShapeletLength = ops.getMin();
        maxShapeletLength = ops.getMax();
        lengthIncrement = ops.getLengthInc();
        positionIncrement = ops.getPosInc();      
        numDimensions = ops.getNumDimensions();
    }
    
    public void setMinAndMax(int min, int max){
        minShapeletLength = min;
        maxShapeletLength = max;
    }
    
    public int getMin(){
        return minShapeletLength;
    }
    
    public int getMax(){
        return maxShapeletLength;
    }
    
    public void init(Instances input){
        inputData = input;
        
        //we need to detect whether it's multivariate or univariate.
        //this feels like a hack. BOO.
        //one relational and a class att.
        seriesLength = getSeriesLength();
    }
    
    public int getSeriesLength(){
        return inputData.numAttributes() >= maxShapeletLength ? inputData.numAttributes() : channelLength(inputData) + 1; //we add one here, because lots of code assumes it has a class value on the end/ 
    }
    
    //given a series and a function to find a shapelet 
    public ArrayList<Shapelet> SearchForShapeletsInSeries(Instance timeSeries, ProcessCandidate checkCandidate){
        ArrayList<Shapelet> seriesShapelets = new ArrayList<>();
        
        //for univariate this will just 
        for (int length = minShapeletLength; length <= maxShapeletLength; length+=lengthIncrement) {
            //for all possible starting positions of that length. -1 to remove classValue but would be +1 (m-l+1) so cancel.
            for (int start = 0; start < seriesLength - length; start+=positionIncrement) {
                //for univariate this will be just once.
                for(int dim = 0; dim < numDimensions; dim++)   {
                    Shapelet shapelet = checkCandidate.process(getTimeSeries(timeSeries,dim), start, length, dim);
                    if (shapelet != null) {
                        seriesShapelets.add(shapelet);
                        shapeletsVisited.add(seriesCount+","+length+","+start+","+shapelet.qualityValue);
                    }
                }
            }
        }
        
        seriesCount++;
        return seriesShapelets;
    }
    
    
    protected Instance getTimeSeries(Instance timeSeries, int dim){
        if(numDimensions > 1)
            return utilities.multivariate_tools.MultivariateInstanceTools.splitMultivariateInstanceWithClassVal(timeSeries)[dim];
        return timeSeries;
    }
}
