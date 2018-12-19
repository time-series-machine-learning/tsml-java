/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.search_functions;

import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearch.SearchType;
import static timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearch.SearchType.FULL;

/**
 *
 * @author raj09hxu
 */
public class ShapeletSearchOptions {

    public int getMin() {
        return min;
    }

    public int getMax() {
        return max;
    }

    public long getSeed() {
        return seed;
    }

    public long getNumShapelets() {
        return numShapelets;
    }

    public int getLengthInc() {
        return lengthInc;
    }

    public int getPosInc() {
        return posInc;
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
    
    private final int min;
    private final int max;
    private final long seed;
    private final long numShapelets;
    private final int lengthInc;
    private final int posInc;
    private final float proportion;
    private final int maxIterations;
    private final long timeLimit;
    private final SearchType searchType;
    private final int numDimensions;
    private final int[] lengthDistribution;
    
    protected ShapeletSearchOptions(Builder ops){
        min = ops.min;
        max = ops.max;
        seed = ops.seed;
        numShapelets= ops.numShapelets;
        lengthInc = ops.lengthInc;
        posInc = ops.posInc;
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
        private long numShapelets;
        private int lengthInc = 1;
        private int posInc = 1;
        private float proportion = 1.0f;
        private int maxIterations;
        private long timeLimit;
        private SearchType searchType;
        private int[] lengthDistribution;
        private int numDimensions = 1;

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

        public Builder setNumShapelets(long numShapelets) {
            this.numShapelets = numShapelets;
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
    
    
    
    
}
