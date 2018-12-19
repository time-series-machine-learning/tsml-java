/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.search_functions;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import timeseriesweka.filters.shapelet_transforms.Shapelet;
import utilities.generic_storage.Pair;
import utilities.generic_storage.Triple;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Aaron
 */
public class ImpRandomSearch extends RandomSearch{
     protected Map<Integer, ArrayList<Triple<Integer,Integer, Integer>>> shapeletsToFind = new HashMap<>();
    
    int currentSeries =0;
    
    public  Map<Integer, ArrayList<Triple<Integer,Integer, Integer>>> getShapeletsToFind(){
        return shapeletsToFind;
    }
        
    protected ImpRandomSearch(ShapeletSearchOptions ops) {
        super(ops);
    }

    
    @Override
    public void init(Instances input){
        super.init(input);
        int numLengths = maxShapeletLength - minShapeletLength; //want max value to be inclusive.
        
        //generate the random shapelets we're going to visit.
        for(int i=0; i<numShapelets; i++){
            //randomly generate values.
            int series = random.nextInt(input.numInstances());
            int length = random.nextInt(numLengths) + minShapeletLength; //offset the index by the min value.
            int position  = random.nextInt(seriesLength - length); // can only have valid start positions based on the length. (numAtts-1)-l+1
            int dimension = random.nextInt(numDimensions);
            //find the shapelets for that series.
            ArrayList<Triple<Integer,Integer, Integer>> shapeletList = shapeletsToFind.get(series);
            if(shapeletList == null)
                shapeletList = new ArrayList<>();
            
            //add the random shapelet to the length
            shapeletList.add(new Triple(length, position, dimension));
            //put back the updated version.
            
            shapeletsToFind.put(series, shapeletList);
        }          
    }
    
    
    @Override
    public ArrayList<Shapelet> SearchForShapeletsInSeries(Instance timeSeries, ShapeletSearch.ProcessCandidate checkCandidate){
        
        ArrayList<Shapelet> seriesShapelets = new ArrayList<>();
        ArrayList<Triple<Integer,Integer, Integer>> shapeletList = shapeletsToFind.get(currentSeries);
        currentSeries++;
        
        //no shapelets to consider.
        if(shapeletList == null){
            return seriesShapelets;
        }
        
        //Only consider a fixed amount of shapelets.
        for(Triple<Integer,Integer, Integer> shapelet : shapeletList){
            //position is in var2, and length is in var1
            Shapelet shape = checkCandidate.process(getTimeSeries(timeSeries,shapelet.var3), shapelet.var1, shapelet.var2, shapelet.var3);
            if(shape != null)
                seriesShapelets.add(shape);           
        }

        return seriesShapelets;
    }
}
