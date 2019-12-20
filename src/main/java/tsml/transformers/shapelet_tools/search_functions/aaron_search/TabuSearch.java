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
package tsml.transformers.shapelet_tools.search_functions.aaron_search;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.LinkedList;
import java.util.Queue;

import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchOptions;
import weka.core.Instance;
import weka.core.Instances;
import tsml.transformers.shapelet_tools.Shapelet;
/**
 *
 * @author raj09hxu
 */
public class TabuSearch extends ImprovedRandomSearch {
    int neighbourhoodWidth = 3;     //3x3 
    
    int maxTabuSize = 50;

    int initialNumShapeletsPerSeries;
    
    Shapelet bsf_shapelet;  
    
    BitSet seriesToConsider;
    
    float proportion = 1.0f;
    
    public TabuSearch(ShapeletSearchOptions ops) {
        super(ops);
        
        proportion = ops.getProportion();
    }
    
    @Override
    public void init(Instances input){
        super.init(input);
        
        float subsampleSize = (float) inputData.numInstances() * proportion;
        initialNumShapeletsPerSeries = (int) ((float) numShapeletsPerSeries / subsampleSize);
        seriesToConsider = new BitSet(inputData.numInstances());
        
        
        System.out.println(initialNumShapeletsPerSeries);
        
        //if we're looking at less than root(m) shapelets per series. sample to root n.
        if(initialNumShapeletsPerSeries < Math.sqrt(inputData.numAttributes()-1)){
            //recalc prop and subsample size.
            proportion =  ((float) Math.sqrt(inputData.numInstances()) / (float)inputData.numInstances());
            subsampleSize = (float) inputData.numInstances() * proportion;
            initialNumShapeletsPerSeries = (int) ((float) numShapeletsPerSeries / subsampleSize);
            System.out.println("subsampleSize " + (int)subsampleSize);
        }
                    
        if(proportion >= 1.0){
            seriesToConsider.set(0, inputData.numInstances(), true); //enable all
        }
        else{
            //randomly select % of the series.
            for(int i=0; i< subsampleSize; i++){
                seriesToConsider.set(random.nextInt((int) inputData.numInstances()));
            }
            System.out.println(seriesToConsider);
        }
        
        System.out.println(initialNumShapeletsPerSeries);
        
        // we might need to reduce the number of series. could do 10% subsampling.
        if(initialNumShapeletsPerSeries < 1)
            System.err.println("Too Few Starting shapelets");
    }
    
    
    @Override
    public ArrayList<Shapelet> searchForShapeletsInSeries(Instance timeSeries, ProcessCandidate checkCandidate){
        
        ArrayList<Shapelet> seriesShapelets = new ArrayList<>();
        
        if(!seriesToConsider.get(currentSeries++)) return seriesShapelets;
        
        Queue<CandidateSearchData> tabuList = new LinkedList<>();
        
        CandidateSearchData shapelet;
        
        int numShapeletsEvaluated = 0;

        //Only consider a fixed amount of shapelets.
        while(initialNumShapeletsPerSeries > numShapeletsEvaluated){
            
            //create the random shapelet.
            //if it's the first iteration and we've got a previous best shapelet.
            if(numShapeletsEvaluated == 0 && bsf_shapelet != null){
               shapelet = new CandidateSearchData(bsf_shapelet.startPos,bsf_shapelet.length) ;
               bsf_shapelet = null; //reset the best one for this series.
            }else{
                shapelet = createRandomShapelet(timeSeries);
            }
            
            ArrayList<CandidateSearchData> candidateList = new ArrayList<>();
            candidateList.add(shapelet);
            candidateList.addAll(createNeighbourhood(shapelet, timeSeries.numAttributes()));
            boolean inList = false;
            for(CandidateSearchData neighbour : candidateList){
                //i think if we collide with the tabuList we should abandon the neighbourhood.
                if(tabuList.contains(neighbour)){
                    inList = true;
                    break;
                }
            }
            //if inList is true we want to abandon this whole search area.
            if(inList){
                continue;
            }

            //find the best local candidate
            CandidateSearchData bestLocal = null;
            Shapelet local_bsf_shapelet = null;
            for(CandidateSearchData shape : candidateList ){
                Shapelet sh = checkCandidate.process(timeSeries, shape.getStartPosition(), shape.getLength());
                numShapeletsEvaluated++;

                //we've abandoned this shapelet, and therefore it is null.
                if(sh == null) continue;
                
                if(local_bsf_shapelet == null){
                    bestLocal = shape;
                    local_bsf_shapelet = sh;
                }

                //if the comparator says it's better.
                if(comparator.compare(local_bsf_shapelet, sh) > 0){
                    bestLocal = shape;
                    local_bsf_shapelet = sh;
                }
            }
            
            if(local_bsf_shapelet == null) continue;
            
            
            if(bsf_shapelet == null){
                bsf_shapelet = local_bsf_shapelet;
                seriesShapelets.add(local_bsf_shapelet); //stick the local best ones in the list.
            }
                
            //update bsf shapelet if the local one is better.
            if(comparator.compare(bsf_shapelet, local_bsf_shapelet) > 0){
                bsf_shapelet = local_bsf_shapelet;
                seriesShapelets.add(local_bsf_shapelet); //stick the local best ones in the list.
            }          
            
            //add the best local to the TabuList
            tabuList.add(bestLocal);
            
            if(tabuList.size() > maxTabuSize){
                tabuList.remove();
            }
        }
        
        return seriesShapelets;
    }
    
    ArrayList<CandidateSearchData> createNeighbourhood(CandidateSearchData shapelet){
        return createNeighbourhood(shapelet, inputData.numAttributes()-1);
    }
    
    ArrayList<CandidateSearchData> createNeighbourhood(CandidateSearchData shapelet, int m){
        ArrayList<CandidateSearchData> neighbourhood = new ArrayList<>();
        neighbourhood.add(shapelet); //add the shapelet to the neighbourhood.
        
        int halfWidth = (int)((double)neighbourhoodWidth / 2.0);
                
        for(int pos= -halfWidth; pos <= halfWidth; pos++){
            for(int len= -halfWidth; len <= halfWidth; len++){
                if(len == 0 && pos == 0) continue;
                
                //need to prune impossible shapelets.
                int newLen = shapelet.getLength() + len;
                int newPos = shapelet.getStartPosition() + pos;
                
                if(newLen < minShapeletLength || //don't allow length to be less than minShapeletLength. 
                   newLen > maxShapeletLength || //don't allow length to be more than maxShapeletLength.
                   newPos < 0                 || //don't allow position to be less than 0.               
                   newPos >= (m-newLen))       //don't allow position to be greater than m-l+1.
                    continue;
                
                neighbourhood.add(new CandidateSearchData(newPos,newLen));
            } 
        }
        
        return neighbourhood;
    }
    
    private CandidateSearchData createRandomShapelet(Instance series){
        int numLengths = maxShapeletLength - minShapeletLength; //want max value to be inclusive.
        int length = random.nextInt(numLengths) + minShapeletLength; //offset the index by the min value.
        int position  = random.nextInt(series.numAttributes() - length); // can only have valid start positions based on the length. (numAtts-1)-l+1
        return new CandidateSearchData(position,length);   
    }
    
    
    public static void main(String[] args){
        //3 -> 100 series. 100 shapelets.
        
        //will aim to make a searchFactory so you dont hand build a searchFunction.
        ShapeletSearchOptions tabuOptions = new ShapeletSearchOptions.Builder().setMin(3).setMax(100).setNumShapeletsToEvaluate(1000).setSeed(0).build();
        
        TabuSearch tb = new  TabuSearch(tabuOptions);
        //edge case neighbour hood testing
        //length of m, and 
        for (int len = 3; len < 100; len++) {
            for(int pos=0; pos< 100-len+1; pos++){
                ArrayList<CandidateSearchData> createNeighbourhood = tb.createNeighbourhood(new CandidateSearchData(pos,len), 100);
                System.out.println(createNeighbourhood);
            }
        }

    }
}
