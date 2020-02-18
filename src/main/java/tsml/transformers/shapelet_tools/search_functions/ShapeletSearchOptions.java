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
package tsml.transformers.shapelet_tools.search_functions;

import tsml.transformers.shapelet_tools.search_functions.ShapeletSearch.SearchType;
import static tsml.transformers.shapelet_tools.search_functions.ShapeletSearch.SearchType.FULL;

/**
 *
 * @author Aaron Bostrom
 * ANY comments in this are added by Tony
 * This is a configuration utility for the class ShapeletSearch, used in the complex initialisation method
 * of Aaron's invention
 */
public class ShapeletSearchOptions {

    private final long timeLimit;//Is this in conjunction with numShapelets?
    private final SearchType searchType;
    private final int numDimensions;        //Number of dimensions in the data
    private final int min; //Min length of shapelets
    private final int max; //Max length of shapelets
    private final long seed;

    private final long numShapeletsToEvaluate;  //The number of shapelets to sample PER SERIES used in RandomSearch and subclasses

    private final int lengthIncrement;  //Defaults to 1 in ShapeletSearch, SkippingSearch will use this to avoid full search
    private final int posIncrement;     //Defaults to 1 in ShapeletSearch, SkippingSearch will use this to avoid full search
    private final float proportion;     //  Used in TabuSearch, RefinedRandomSearch, SubsampleRandomSearch, MagnifySearch
    private final int maxIterations;    // Used in LocalSearch to determine the number of search steps to take
    private final int[] lengthDistribution; // used in SkewedRandomSearch

    /**
     *     Why a protected constructor? So you have to go through ShapeletSearchOptions.Builder in
     *     order to configure
     *             ShapeletSearchOptions.Builder searchBuilder = new ShapeletSearchOptions.Builder();
     */
    protected ShapeletSearchOptions(Builder ops){
        min = ops.min;
        max = ops.max;
        seed = ops.seed;
        numShapeletsToEvaluate = ops.numShapeletsToEvaluate; //The number of shapelets to sample PER SERIES
        lengthIncrement = ops.lengthInc;
        posIncrement = ops.posInc;
        proportion = ops.proportion;
        maxIterations = ops.maxIterations;
        timeLimit = ops.timeLimit;
        searchType = ops.searchType;
        numDimensions = ops.numDimensions;
        lengthDistribution = ops.lengthDistribution;
    }

    public static class Builder{
        private int min;
        private int max;
        private long seed;
        private long numShapeletsToEvaluate; //Number of shapelets to evaluate PER SERIES
        private int lengthInc = 1;
        private int posInc = 1;
        private float proportion = 1.0f;
        private int maxIterations;
        private long timeLimit;
        private SearchType searchType;
        private int[] lengthDistribution;
        private int numDimensions = 1;

//Setters: why do they all return themselves?
        public Builder setNumDimensions(int dim){
            numDimensions = dim;
            return this;
        }
        public Builder setLengthDistribution(int[] lengthDist){
            lengthDistribution = lengthDist;
            return this;
        }
        public Builder setSearchType(SearchType st){
            searchType = st;
            return this;
        }
        
        public Builder setTimeLimit(long lim){
            timeLimit = lim;
            return this;
        }
        
        public Builder setMin(int min) {
            this.min = min;
            return this;
        }

        public Builder setMax(int max) {
            this.max = max;
             return this;
        }

        public Builder setSeed(long seed) {
            this.seed = seed;
             return this;
        }

        public Builder setNumShapeletsToEvaluate(long numShapeletsToEvaluate) {
            this.numShapeletsToEvaluate = numShapeletsToEvaluate;
             return this;
        }

        public Builder setLengthInc(int lengthInc) {
            this.lengthInc = lengthInc;
             return this;
        }

        public Builder setPosInc(int posInc) {
            this.posInc = posInc;
             return this;
        }

        public Builder setProportion(float proportion) {
            this.proportion = proportion;
             return this;
        }

        public Builder setMaxIterations(int maxIterations) {
            this.maxIterations = maxIterations;
             return this;
        }
        
        public ShapeletSearchOptions build(){
            setDefaults();
            return new ShapeletSearchOptions(this);
        }
        
        public void setDefaults(){
            if(searchType == null){
                searchType = FULL;
            }
        }
    }


    //Getters
    public int getMin() {
        return min;
    }
    public int getMax() {
        return max;
    }
    public long getSeed() {
        return seed;
    }
    public long getNumShapeletsToEvaluate() {
        return numShapeletsToEvaluate;
    }
    public int getLengthIncrement() {
        return lengthIncrement;
    }
    public int getPosIncrement() {
        return posIncrement;
    }
    public float getProportion() {
        return proportion;
    }
    public int getMaxIterations() {
        return maxIterations;
    }
    public long getTimeLimit() {
        return timeLimit;
    }
    public SearchType getSearchType(){
        return searchType;
    }
    public int[] getLengthDistribution() {
        return lengthDistribution;
    }
    public int getNumDimensions(){
        return numDimensions;
    }



}
