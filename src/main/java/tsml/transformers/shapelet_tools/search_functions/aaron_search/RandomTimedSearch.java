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
import java.util.Random;

import tsml.transformers.shapelet_tools.search_functions.RandomSearch;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchOptions;
import weka.core.Instance;
import tsml.transformers.shapelet_tools.Shapelet;
/**
 *
 * @author raj09hxu
 */
public class RandomTimedSearch extends RandomSearch {
        
    protected long timeLimit;

    public RandomTimedSearch(ShapeletSearchOptions ops) {
        super(ops);    
        
        timeLimit = ops.getTimeLimit();
        
        random = new Random(ops.getSeed());
    }
    
    @Override
    public ArrayList<Shapelet> searchForShapeletsInSeries(Instance timeSeries, ProcessCandidate checkCandidate){
        
        long currentTime =0;
        
        ArrayList<Shapelet> seriesShapelets = new ArrayList<>();
        
        int numLengths = maxShapeletLength - minShapeletLength /*+ 1*/; //want max value to be inclusive.
        
        visited = new boolean[numLengths][];
        
        //you only get a 1/nth of the time.
        while((timeLimit/inputData.numInstances()) > currentTime){
            int lengthIndex = random.nextInt(numLengths);
            int length = lengthIndex + minShapeletLength; //offset the index by the min value.
            
            int maxPositions = seriesLength - length ;
            int start  = random.nextInt(maxPositions); // can only have valid start positions based on the length.

            //we haven't constructed the memory for this length yet.
            initVisitedMemory(seriesLength, length);
            
            Shapelet shape = visitCandidate(timeSeries, start, length, checkCandidate);
            if(shape != null)
                seriesShapelets.add(shape);

            
            //we add time, even if we've visited it, this is just incase we end up stuck in some
            // improbable recursive loop.
            currentTime += calculateTimeToRun(inputData.numInstances(), seriesLength-1, length); //n,m,l            
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
    
            
    protected long calculateTimeToRun(int n, int m, int length){
        long time = (m - length + 1) * length; //number of subsequeneces in the seuquenece, and we do euclidean comparison length times for each.
        return time * (n-1); //we calculate this for n-1 series.
    }
    

}
