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
package tsml.filters.shapelet_filters.search_functions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import weka.core.Instance;
import weka.core.Instances;
import tsml.filters.shapelet_filters.Shapelet;
import static utilities.multivariate_tools.MultivariateInstanceTools.channelLength;
/**
 *
 * @author Aaron Bostrom
 * edited Tony Bagnall 25/11/19
 * Base class for ShapeletSearch that uses full enumeration.
 * Subclasses override SearchForShapeletsInSeries to define how to search a single series
 */
public class ShapeletSearch implements Serializable{

//Defines the search technique defined in this package.
    public enum SearchType {FULL, //Evaluate all shapelets using
                            RANDOM,
//ALL of the below are things Aaron tried in his thesis (package aaron_search)
//It is not commented and somewhat untested.
        GENETIC, FS, //Fast shapeletes
    //All of these are now in aaron_search and are largely untested. Recommend not using them.
        LOCAL, MAGNIFY, TIMED_RANDOM, SKIPPING, TABU, REFINED_RANDOM, IMPROVED_RANDOM, SUBSAMPLE_RANDOM, SKEWED, BO_SEARCH};
    
    
    //Immutable class to store search params. 
    //avoids using the Triple tuple, which is less clear.
    //TODO: could have a better name.
    protected static class CandidateSearchData{
        private final int startPosition;
        private final int length;
        private final int dimension; //this is optional, for use with multivariate data.
                                    // If included it relates to the dimension of the data the shapelet is associated with
        
        public CandidateSearchData(int pos,int len){
            startPosition = pos;
            length = len;
            dimension = 0;
        }

        public CandidateSearchData(int pos,int len,int dim){
            startPosition = pos;
            length = len;
            dimension = dim;
        }

        /**
         * @return the startPosition
         */
        public int getStartPosition() {
            return startPosition;
        }
        /**
         * @return the length
         */
        public int getLength() {
            return length;
        }
        /**
         * @return the dimension
         */
        public int getDimension() {
            return dimension;
        }
    }

    public interface ProcessCandidate{
        public default Shapelet process(Instance candidate, int start, int length) {return process(candidate, start, length, 0);}
        public Shapelet process(Instance candidate, int start, int length, int dimension);
    }
    protected ArrayList<String> shapeletsVisited = new ArrayList<>();
    protected int seriesCount;
    
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

    public long getNumShapeletsPerSeries(){ return options.getNumShapeletsToEvaluate();}
    protected ShapeletSearch(ShapeletSearchOptions ops){
        options = ops;
        
        minShapeletLength = ops.getMin();
        maxShapeletLength = ops.getMax();
        lengthIncrement = ops.getLengthIncrement();
        positionIncrement = ops.getPosIncrement();
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
    public String getSearchType(){
        return options.getSearchType().toString();
    }
    public void init(Instances input){
        inputData = input;
        
        //we need to detect whether it's multivariate or univariate.
        //this feels like a hack. BOO.
        //one relational and a class att.
        //TO DO: Tony says: this is both a hack and incorrect: it is counting the class value.
        seriesLength = setSeriesLength();
    }
    public int getSeriesLength(){
        return seriesLength; //we add one here, because lots of code assumes it has a class value on the end/ TO DO: CLARIFY THIS
    }

    public int setSeriesLength(){
        return inputData.numAttributes() >= maxShapeletLength ? inputData.numAttributes() : channelLength(inputData) + 1; //we add one here, because lots of code assumes it has a class value on the end/
    }
    
    //given a series and a function to find a shapelet 
    public ArrayList<Shapelet> searchForShapeletsInSeries(Instance timeSeries, ProcessCandidate checkCandidate){
        ArrayList<Shapelet> seriesShapelets = new ArrayList<>();
        
        //for univariate this will just evaluate all shapelets
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
