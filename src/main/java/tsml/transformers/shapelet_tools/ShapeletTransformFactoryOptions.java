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
package tsml.transformers.shapelet_tools;

import tsml.transformers.shapelet_tools.quality_measures.ShapeletQuality.ShapeletQualityChoice;
import static tsml.transformers.shapelet_tools.quality_measures.ShapeletQuality.ShapeletQualityChoice.INFORMATION_GAIN;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearch;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchOptions;
import tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance.DistanceType;
import static tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance.DistanceType.NORMAL;
import tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance.RescalerType;
import static tsml.transformers.shapelet_tools.distance_functions.ShapeletDistance.RescalerType.NORMALISATION;

/**
 *
 * @author Aaron
 * This is a holder for the shapelet options which are then used to build the Transform in ShapeletTransformFactory.
 * We have a Builder class, that clones the Options class, that sets up the Factory class, that builds the Transform
 *
 * I knew an old lady who swallowed a fly, perhaps she'll die?  This could usefully be restructured ....
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
    private final RescalerType rescalerType;

    private ShapeletTransformFactoryOptions(ShapeletTransformOptions options){
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
        rescalerType = options.rescalerType;
    }

    public RescalerType  getRescalerType(){
        return rescalerType;
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

    /** Funny inner class that serves no purpose but to confuse ... and why does everything return this?!?
     *
      */
    public static class ShapeletTransformOptions {
        
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
        private RescalerType rescalerType;


        public ShapeletTransformOptions useRoundRobin(){
            roundRobin = true;
            return this;
        }
        public ShapeletTransformOptions useCandidatePruning(){
            candidatePruning = true;
            return this;
        }

        public ShapeletTransformOptions setRoundRobin(boolean b){
            roundRobin = b;
            return this;
        }


        public ShapeletTransformOptions setCandidatePruning(boolean b){
            candidatePruning = b;
            return this;
        }
        
        public ShapeletTransformOptions setSearchOptions(ShapeletSearchOptions sOp){
            searchOptions = sOp;
            return this;
        }
        
        public ShapeletTransformOptions setQualityMeasure(ShapeletQualityChoice qm){
            qualityChoice = qm;
            return this;
        }
        
        public ShapeletTransformOptions setMinLength(int min){
            minLength = min;
            return this;
        }
        public ShapeletTransformOptions setMaxLength(int max){
            maxLength = max;
            return this;
        }
        public int getMaxLength() {
            return maxLength;
        }
        public int getMinLength() {
            return minLength;
        }
        public ShapeletTransformOptions setKShapelets(int k){
            kShapelets = k;
            return this;
        }
        public ShapeletTransformOptions setClassBalancing(boolean b){
            balanceClasses = b;
            return this;
        }

        public ShapeletTransformOptions setBinaryClassValue(boolean b){
            binaryClassValue = b;
            return this;
        }

        public ShapeletTransformOptions useClassBalancing(){
            balanceClasses = true;
            return this;
        }
        
        public ShapeletTransformOptions useBinaryClassValue(){
            binaryClassValue = true;
            return this;
        }
        
        public ShapeletTransformOptions setDistanceType(DistanceType dis){
            dist = dis;
            return this;
        }
        
                
        public ShapeletTransformOptions setRescalerType(RescalerType type){
            rescalerType = type;
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
            
            if(rescalerType == null){
                rescalerType = NORMALISATION;
            }
            
            if(dist == null){
                dist =  NORMAL;
            }
        }
        public String toString(){
            String str="DistType,"+dist+",QualityMeasure,"+qualityChoice+",RescaleType,"+rescalerType;
            str+=",UseRoundRobin,"+roundRobin+",useCandidatePruning,"+candidatePruning+",UseClassBalancing,"+balanceClasses;
            str+=",useBinaryClassValue,"+binaryClassValue+",minShapeletLength,"+minLength+",maxShapeletLength,"+maxLength;
            return str;
        }
    }

}
