/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesweka.filters.shapelet_transforms.search_functions;

import java.util.ArrayList;
import java.util.Random;
import utilities.generic_storage.Pair;
import utilities.generic_storage.Triple;
import weka.core.Instances;

/**
 *
 * @author Aaron
 */
public class SkewedRandomSearch extends ImpRandomSearch{
    
    int[] lengthDistribution;
    int[] cumulativeDistribution;
    
    protected SkewedRandomSearch(ShapeletSearchOptions sops){
        super(sops);
        
        lengthDistribution = sops.getLengthDistribution();
    }
    
    @Override
    public void init(Instances input){
        super.init(input);

        cumulativeDistribution = findCumulativeCounts(lengthDistribution);
        //generate the random shapelets we're going to visit.
        for(int i=0; i<numShapelets; i++){
            //randomly generate values.
            int series = random.nextInt(input.numInstances());
            
            //this gives an index, we assume the length dsitribution is from min-max. so a value of 0 is == minShapeletLength
            int length = sampleCounts(cumulativeDistribution, random) + minShapeletLength; //select the random length from the distribution of lengths.
            int position  = random.nextInt(seriesLength - length); // can only have valid start positions based on the length. (numAtts-1)-l+1
            int dimension = random.nextInt(numDimensions);
            
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
