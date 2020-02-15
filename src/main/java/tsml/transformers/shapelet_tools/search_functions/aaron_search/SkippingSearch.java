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

import tsml.transformers.shapelet_tools.search_functions.ShapeletSearch;
import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchOptions;
import weka.core.Instance;
import weka.core.Instances;
import tsml.transformers.shapelet_tools.Shapelet;
/**
 *
 * @author Aaron
 *
 * Skipping search. Assume it just jumps through a range of
 */
public class SkippingSearch extends ShapeletSearch {
    
    int[] positions;
    int[] lengths;
    
    public SkippingSearch(ShapeletSearchOptions sops){
        super(sops);
        
    }
    
    @Override
    public void init(Instances input){
        super.init(input);
        
        //create array of classValues.
        positions = new int[input.numClasses()];
        lengths = new int[input.numClasses()];
    }
    
    
    @Override
    public ArrayList<Shapelet> searchForShapeletsInSeries(Instance timeSeries, ProcessCandidate checkCandidate){
        //we want to store a startLength and startPos for each class and cycle them when we're skipping.
        int index = (int)timeSeries.classValue();
        int start = positions[index];
        int length = lengths[index] + minShapeletLength;
        ArrayList<Shapelet> seriesShapelets = new ArrayList<>();

        for (; length <= maxShapeletLength; length+=lengthIncrement) {
            //for all possible starting positions of that length. -1 to remove classValue
            for (; start <= seriesLength - length - 1; start+=positionIncrement) {
                Shapelet shapelet = checkCandidate.process(timeSeries, start, length);
                if (shapelet != null) {
                    seriesShapelets.add(shapelet);
                }
            }
        }
        
        //IE if we're skipping 2positions. we want to cycle between starting a series at 0,1
        positions[index] = ++positions[index] % positionIncrement;
        lengths[index] = ++lengths[index] % lengthIncrement;
        return seriesShapelets;
    }
    
}
