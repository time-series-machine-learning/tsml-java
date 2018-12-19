/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.search_functions;

import java.util.ArrayList;
import weka.core.Instance;
import weka.core.Instances;
import timeseriesweka.filters.shapelet_transforms.Shapelet;
/**
 *
 * @author Aaron
 */
public class SkippingSearch extends ShapeletSearch{
    
    int[] positions;
    int[] lengths;
    
    protected SkippingSearch(ShapeletSearchOptions sops){
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
    public ArrayList<Shapelet> SearchForShapeletsInSeries(Instance timeSeries, ProcessCandidate checkCandidate){
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
