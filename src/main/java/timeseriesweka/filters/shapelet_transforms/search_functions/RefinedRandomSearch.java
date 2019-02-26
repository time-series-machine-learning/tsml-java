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

import java.util.ArrayList;
import timeseriesweka.filters.shapelet_transforms.ShapeletTransformTimingUtilities;
import utilities.generic_storage.Pair;
import utilities.generic_storage.Triple;
import weka.core.Instances;
/**
 *
 * @author raj09hxu
 */
public class RefinedRandomSearch extends ImpRandomSearch{

    float shapeletToSeriesRatio;
    
    protected RefinedRandomSearch(ShapeletSearchOptions ops) {
        super(ops);
        
        shapeletToSeriesRatio = ops.getProportion();
    }
       
    @Override
    public void init(Instances input){
        super.init(input); 
        int numInstances = input.numInstances();
        int numAttributes = seriesLength - 1;
        
         float currentRatio;
         do{
            long totalShapelets = ShapeletTransformTimingUtilities.calculateNumberOfShapelets(--numInstances, numAttributes, minShapeletLength, maxShapeletLength);
            currentRatio = (float) numShapelets / (float) totalShapelets;
            
            if(numInstances == 25) break; // any less than 25 and we've sampled too far (Subject to change and discussion).
            
        }while(currentRatio < shapeletToSeriesRatio);
         

        inputData = input;
        int numLengths = maxShapeletLength - minShapeletLength; //want max value to be inclusive.
        
        
        //generate the random shapelets we're going to visit.
        for(int i=0; i<numShapelets; i++){
            //randomly generate values.
            int series = random.nextInt(numInstances);
            int length = random.nextInt(numLengths) + minShapeletLength; //offset the index by the min value.
            int position  = random.nextInt(numAttributes - length + 1); // can only have valid start positions based on the length. the upper bound is exclusive. 
            int dimension = random.nextInt(numDimensions);
            //so for the m-m+1 case it always resolves to 0.
            
            //find the shapelets for that series.
            ArrayList<Triple<Integer,Integer,Integer>> shapeletList = shapeletsToFind.get(series);
            if(shapeletList == null)
                shapeletList = new ArrayList<>();
            
            //add the random shapelet to the length
            shapeletList.add(new Triple(length, position, dimension));
            //put back the updated version.
            shapeletsToFind.put(series, shapeletList);
        }          
    }
    
    
}
