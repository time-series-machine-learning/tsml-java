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
import java.util.HashMap;
import java.util.Map;
import tsml.transformers.shapelet_tools.Shapelet;
import tsml.transformers.shapelet_tools.search_functions.RandomSearch;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchOptions;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Aaron
 *
random search of shapelet locations with replacement seems
to be the improvement.

 */
public class ImprovedRandomSearch extends RandomSearch {
    
     protected Map<Integer, ArrayList<CandidateSearchData>> shapeletsToFind = new HashMap<>();
    
    int currentSeries =0;
    
    public  Map<Integer, ArrayList<CandidateSearchData>> getShapeletsToFind(){
        return shapeletsToFind;
    }
        
    public ImprovedRandomSearch(ShapeletSearchOptions ops) {
        super(ops);
    }

    
    @Override
    public void init(Instances input){
        super.init(input);
        int numLengths = maxShapeletLength - minShapeletLength; //want max value to be inclusive.
        
        //generate the random shapelets we're going to visit.
        for(int i = 0; i< numShapeletsPerSeries; i++){
            //randomly generate values.
            int series = random.nextInt(input.numInstances());
            int length = random.nextInt(numLengths) + minShapeletLength; //offset the index by the min value.
            int position  = random.nextInt(seriesLength - length); // can only have valid start positions based on the length. (numAtts-1)-l+1
            int dimension = random.nextInt(numDimensions);
            //find the shapelets for that series.
            ArrayList<CandidateSearchData> shapeletList = shapeletsToFind.get(series);
            if(shapeletList == null)
                shapeletList = new ArrayList<>();
            
            //add the random shapelet to the length
            shapeletList.add(new CandidateSearchData(position,length,dimension));
            //put back the updated version.
            
            shapeletsToFind.put(series, shapeletList);
        }          
    }
    
    
    @Override
    public ArrayList<Shapelet> searchForShapeletsInSeries(Instance timeSeries, ProcessCandidate checkCandidate){
        
        ArrayList<Shapelet> seriesShapelets = new ArrayList<>();
        ArrayList<CandidateSearchData> shapeletList = shapeletsToFind.get(currentSeries);
        currentSeries++;
        
        //no shapelets to consider.
        if(shapeletList == null){
            return seriesShapelets;
        }
        
        //Only consider a fixed amount of shapelets.
        for(CandidateSearchData shapelet : shapeletList){
            //position is in var2, and length is in var1
            Shapelet shape = checkCandidate.process(getTimeSeries(timeSeries,shapelet.getDimension()), shapelet.getStartPosition(), shapelet.getLength(), shapelet.getDimension());
            if(shape != null)
                seriesShapelets.add(shape);           
        }

        return seriesShapelets;
    }
}
