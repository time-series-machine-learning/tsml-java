/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.search_functions;

import java.util.ArrayList;
import java.util.Random;
import weka.core.Instance;
import timeseriesweka.filters.shapelet_transforms.Shapelet;
/**
 *
 * @author raj09hxu
 */
public class LocalSearch extends RandomTimedSearch{

    
    int maxIterations;
    
    protected LocalSearch(ShapeletSearchOptions ops) {
        super(ops);
        
        maxIterations = ops.getMaxIterations();
    }
    
    @Override
    public ArrayList<Shapelet> SearchForShapeletsInSeries(Instance timeSeries, ProcessCandidate checkCandidate){
        ArrayList<Shapelet> seriesShapelets = new ArrayList<>();
        
        int numLengths = maxShapeletLength - minShapeletLength /*+ 1*/; //want max value to be inclusive.
        
        visited = new boolean[numLengths][];

        //maxIterations is the same as K. 
        for(int currentIterations = 0; currentIterations < maxIterations; currentIterations++){
            int lengthIndex = random.nextInt(numLengths);
            int length = lengthIndex + minShapeletLength; //offset the index by the min value.
            
            int maxPositions = seriesLength - length ;
            int start  = random.nextInt(maxPositions); // can only have valid start positions based on the length.

            //we haven't constructed the memory for this length yet.
            initVisitedMemory(seriesLength, length);


            if(!visited[lengthIndex][start]){
                Shapelet shape = evaluateShapelet(timeSeries, start, length, checkCandidate);
                //if the shapelet is null, it means it was poor quality. so we've abandoned this branch. 
                if(shape != null)
                    seriesShapelets.add(shape);
            }
        }
        return seriesShapelets;
    }

    private static final int START_DEC = 0, START_INC = 1, LENGTH_DEC = 2, LENGTH_INC = 3;
    
    private Shapelet evaluateShapelet(Instance series, int start, int length, ProcessCandidate checkCandidate) {
        //we've not eval'd this shapelet; consider and put in list.
        Shapelet shapelet = visitCandidate(series, start, length, checkCandidate);

        if(shapelet == null)
            return shapelet;
        
        Shapelet[] shapelets;

        int index;
        Shapelet bsf_shapelet = shapelet;    
        
        do{ 
            //need to reset directions after each loop.
            shapelets = new Shapelet[4];
            
            //calculate the best four directions.
            int startDec = start - 1;
            int startInc = start + 1;
            int lengthDec = length - 1;
            int lengthInc = length + 1;

            //as long as our start position doesn't go below 0. 
            if(startDec >= 0){
                shapelets[START_DEC] = visitCandidate(series, startDec, length, checkCandidate);
            }

            //our start position won't make us over flow on length. the start position is in the last pos.
            if(startInc < seriesLength - length){
                shapelets[START_INC] = visitCandidate(series, startInc, length, checkCandidate);
            }

            //don't want to be shorter than the min length && does our length reduction invalidate our start pos?
            if(lengthDec > minShapeletLength && start < seriesLength - lengthDec){
                shapelets[LENGTH_DEC] = visitCandidate(series, start, lengthDec, checkCandidate);
            }

            //dont want to be longer than the max length && does our start position invalidate our new length. 
            if(lengthInc < maxShapeletLength && start < seriesLength - lengthInc){

                shapelets[LENGTH_INC] = visitCandidate(series, start, lengthInc, checkCandidate);
            }
            
            //find the best shaplet direction and record which the best direction is. if it's -1 we don't want to move. 
            //we only want to move when the shapelet is better, or equal but longer. 
            index = -1;
            for(int i=0; i < shapelets.length; i++){
                Shapelet shape = shapelets[i];
                
                //if we're greater than the quality value then we want it, or if we're the same as the quality value but we are increasing length.
                if(shape != null && ((shape.qualityValue > bsf_shapelet.qualityValue) || 
                  (shape.qualityValue == bsf_shapelet.qualityValue && i == LENGTH_INC))){
                    index = i;
                    bsf_shapelet = shape;
                }
            }
            
                  
            //find the direction thats best, if it's same as current but longer keep searching, if it's shorter and same stop, if it's 
            start = bsf_shapelet.startPos;
            length = bsf_shapelet.length;
            
            
        }while(index != -1); //no directions are better.
        
        
        return bsf_shapelet;
    }
}
