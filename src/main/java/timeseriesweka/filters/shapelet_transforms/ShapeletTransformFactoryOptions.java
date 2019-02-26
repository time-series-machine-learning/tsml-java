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
package timeseriesweka.filters.shapelet_transforms;

import timeseriesweka.filters.shapelet_transforms.quality_measures.ShapeletQuality.ShapeletQualityChoice;
import static timeseriesweka.filters.shapelet_transforms.quality_measures.ShapeletQuality.ShapeletQualityChoice.INFORMATION_GAIN;
import timeseriesweka.filters.shapelet_transforms.quality_measures.ShapeletQualityMeasure;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearch;
import timeseriesweka.filters.shapelet_transforms.search_functions.ShapeletSearchOptions;
import timeseriesweka.filters.shapelet_transforms.distance_functions.SubSeqDistance.DistanceType;
import static timeseriesweka.filters.shapelet_transforms.distance_functions.SubSeqDistance.DistanceType.NORMAL;

/**
 *
 * @author Aaron
 */
public class ShapeletTransformFactoryOptions {
   
    private final int minLength;
    private final int maxLength; 
    private final int kShapelets;
    private final boolean balanceClasses;
    private final boolean binaryClassValue;
    private final boolean roundRobin;
    private final boolean candidatePruning;
    private final DistanceType distance;
    private final ShapeletQualityChoice qualityChoice;
    private final ShapeletSearchOptions searchOptions;
    
    
    private ShapeletTransformFactoryOptions(Builder options){
        minLength = options.minLength;
        maxLength = options.maxLength;
        kShapelets = options.kShapelets;
        balanceClasses = options.balanceClasses;
        binaryClassValue = options.binaryClassValue;
        distance = options.dist;
        qualityChoice = options.qualityChoice;
        searchOptions = options.searchOptions;
        roundRobin = options.roundRobin;
        candidatePruning = options.candidatePruning;
    }

    public boolean isBalanceClasses() {
        return balanceClasses;
    }

    public boolean isBinaryClassValue() {
        return binaryClassValue;
    }
    public int getMinLength() {
        return minLength;
    }

    public int getMaxLength() {
        return maxLength;
    }

    public int getkShapelets() {
        return kShapelets;
    }
    
    public DistanceType getDistance() {
        return distance;
    }
    
    public boolean useRoundRobin(){
        return roundRobin;
    }
    
        
    public boolean useCandidatePruning(){
        return candidatePruning;
    }
    
    public ShapeletSearchOptions getSearchOptions(){
        return searchOptions;
    }
    
    public ShapeletQualityChoice getQualityChoice(){
        return qualityChoice;
    }
    
    @Override
    public String toString(){
        return minLength + " " + maxLength + " " + kShapelets + " " + balanceClasses;
    }
    
    public static class Builder {
        
        private int minLength;
        private int maxLength; 
        private int kShapelets;
        private boolean balanceClasses;
        private boolean binaryClassValue;
        private boolean roundRobin;
        private boolean candidatePruning;
        private DistanceType dist;
        private ShapeletQualityChoice qualityChoice;
        private ShapeletSearchOptions searchOptions;
        
        
        public Builder useRoundRobin(){
            roundRobin = true;
            return this;
        }
        
        
        public Builder useCandidatePruning(){
            candidatePruning = true;
            return this;
        }
        
        
        public Builder setSearchOptions(ShapeletSearchOptions sOp){
            searchOptions = sOp;
            return this;
        }
        
        public Builder setQualityMeasure(ShapeletQualityChoice qm){
            qualityChoice = qm;
            return this;
        }
        
        public Builder setMinLength(int min){
            minLength = min;
            return this;
        }
        
        public Builder setMaxLength(int max){
            maxLength = max;
            return this;
        }
        
        public Builder setKShapelets(int k){
            kShapelets = k;
            return this;
        }
        
        public Builder useClassBalancing(){
            balanceClasses = true;
            return this;
        }
        
        public Builder useBinaryClassValue(){
            balanceClasses = true;
            return this;
        }
        
        public Builder setDistanceType(DistanceType dis){
            dist = dis;
            return this;
        }
        
        public ShapeletTransformFactoryOptions build(){
            setDefaults();
            return new ShapeletTransformFactoryOptions(this);
        }
        
        private void setDefaults(){            
            //if there is no search options. use default params;
            if(searchOptions == null){
                searchOptions = new ShapeletSearchOptions.Builder()
                                    .setMin(minLength)
                                    .setMax(maxLength)
                                    .setSearchType(ShapeletSearch.SearchType.FULL)
                                    .build();
            }
            
            if(qualityChoice == null){
                qualityChoice = INFORMATION_GAIN;
            }
            
            
            if(dist == null){
                dist =  NORMAL;
            }
        }
    }

}
