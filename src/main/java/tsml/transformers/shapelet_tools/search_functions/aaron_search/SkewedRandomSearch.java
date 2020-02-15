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

import tsml.transformers.shapelet_tools.search_functions.ShapeletSearchOptions;
import weka.core.Instances;

/**
 *
 * @author Aaron
 */
public class SkewedRandomSearch extends ImprovedRandomSearch {
    
    int[] lengthDistribution;
    int[] cumulativeDistribution;
    
    public SkewedRandomSearch(ShapeletSearchOptions sops){
        super(sops);
        
        lengthDistribution = sops.getLengthDistribution();
    }
    
    @Override
    public void init(Instances input){
        super.init(input);

        cumulativeDistribution = findCumulativeCounts(lengthDistribution);
        //generate the random shapelets we're going to visit.
        for(int i = 0; i< numShapeletsPerSeries; i++){
            //randomly generate values.
            int series = random.nextInt(input.numInstances());
            
            //this gives an index, we assume the length dsitribution is from min-max. so a value of 0 is == minShapeletLength
            int length = sampleCounts(cumulativeDistribution, random) + minShapeletLength; //select the random length from the distribution of lengths.
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
    
    /**
    * 
    * @param counts count of number of items at each level i
    * @return cumulative count of items at level <=i
    */    
    public static int[] findCumulativeCounts(int[] counts){
        int[] c=new int[counts.length];
        c[0]=counts[0];
        int i=1;
        while(i<counts.length){
            c[i]=c[i-1]+counts[i];
            i++;
        }
        return c;
    }
    
    /**
    * 
    * @param cumulativeCounts: cumulativeCounts[i] is the number of items <=i
    * as found by findCumulativeCounts 
    * cumulativeCounts[length-1] is the total number of objects
     * @param rand
    * @return a randomly selected level i based on sample of cumulativeCounts
    */
    public static int sampleCounts(int[] cumulativeCounts, Random rand){
        int c=rand.nextInt(cumulativeCounts[cumulativeCounts.length-1]);
        int pos=0;
        while(cumulativeCounts[pos]<= c)
            pos++;
        return pos;
    }


    public static void main(String[] args) {
        int[] histogram = {4,4,0,1};
        int[] cum = findCumulativeCounts(histogram);
        
        Random rand = new Random(0);
        //i histogrammed this for a bunch of different distributions.
        for (int i = 0; i < 1000; i++) {
            //System.out.print(sampleCounts(cum, rand) + ",");
        }

    }


}
