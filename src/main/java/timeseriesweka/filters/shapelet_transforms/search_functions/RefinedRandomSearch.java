/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
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
