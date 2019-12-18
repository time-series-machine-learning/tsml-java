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

import tsml.transformers.shapelet_tools.Shapelet;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearch;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchOptions;
import weka.core.Instance;

import java.util.ArrayList;
import java.util.Random;

/**
 *
 * @author raj09hxu
 *
 * random search of shapelet locations, does not visit the same shapelet twice
 */
public class RandomSearch extends ShapeletSearch{
        
    protected Random random;
    protected long numPerSeries;    //Number of shapelets to sample per series
    protected boolean[][] visited;  //
    
    protected RandomSearch(ShapeletSearchOptions ops) {
        super(ops);    
        numPerSeries = ops.getNumShapeletsToEvaluate();
        random = new Random(ops.getSeed());
    }
    
    @Override
    public ArrayList<Shapelet> searchForShapeletsInSeries(Instance timeSeries, ProcessCandidate checkCandidate){
        
        ArrayList<Shapelet> seriesShapelets = new ArrayList<>();
        
        int numLengths = maxShapeletLength - minShapeletLength /*+ 1*/; //want max value to be inclusive.
        
        visited = new boolean[numLengths][];
        
        //Only consider a fixed number of shapelets per series.
        for(int i = 0; i< numPerSeries; i++ ){
            int lengthIndex = random.nextInt(numLengths);
            int length = lengthIndex + minShapeletLength; //offset the index by the min value.
            
            int maxPositions = seriesLength - length ;
            int start  = random.nextInt(maxPositions); // can only have valid start positions based on the length.

            //we haven't constructed the memory for this length yet.
            initVisitedMemory(seriesLength, length);
            
            Shapelet shape = visitCandidate(timeSeries, start, length, checkCandidate);
            if(shape != null)
                seriesShapelets.add(shape);           
        }

        for(int i=0; i<visited.length; i++){
            if(visited[i] == null) continue;
            for(int j=0; j<visited[i].length; j++){
                if(visited[i][j])
                    shapeletsVisited.add(seriesCount+","+(i+minShapeletLength)+","+j);
            }
        }
        
        seriesCount++; //keep track of the series.
        
        
        return seriesShapelets;
    }
    
        
    protected void initVisitedMemory(int seriesLength, int length){
        int lengthIndex = getLengthIndex(length);
        if(visited[lengthIndex] == null){
            int maxPositions = seriesLength - length;
            visited[lengthIndex] = new boolean[maxPositions];
        }  
    }
    
        
    protected int getLengthIndex(int length){
        return length - minShapeletLength;
    }
      
    public long getNumPerSeries(){ return numPerSeries;}
    protected Shapelet visitCandidate(Instance series, int start, int length, ProcessCandidate checkCandidate){
        initVisitedMemory(series.numAttributes(), length);
        int lengthIndex = getLengthIndex(length);
        Shapelet shape = null;     
        if(!visited[lengthIndex][start]){
            shape = checkCandidate.process(series, start, length);
            visited[lengthIndex][start] = true;
        }
        return shape;
    }

}
