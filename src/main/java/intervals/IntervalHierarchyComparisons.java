/*
 * Copyright (C) 2019 xmw13bzu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package intervals;

import intervals.IntervalHierarchy.Interval;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 *
 * Class for comparing two interval heirarchies, e.g. how do the train estimates for each interval
 * align with the test observations, either for the same classifier or for proxies against targets etc
 * 
 * Computes various metrics
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class IntervalHierarchyComparisons {

    IntervalHierarchy hierTrain;
    IntervalHierarchy hierTest;
    
    ArrayList<Interval> orderedIntervalsTrain;
    ArrayList<Interval> orderedIntervalsTest;
    
    Interval fullSeriesTrain; 
    Interval fullSeriesTest;
    
    int Rtrain;     // data driven para, how many intervals improve over full series, i.e. how many are 'relevant' for a/the proxy
    int Rtest;     // and for b/the target
    int k = 10; // user para, how many intervals at the top do we care about? informed by e.g. what if we ensemble top k
    
    double nDCGatK; //normed discounted cumulative gain at K, default K = 10 
            //https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    double top1; //is the best interval estimated in a the same interval observed to be best in b? 
    double rPrecision; //aka P@R where R = number of intervals better than full series, P = prop intervals better than full series
            //https://cs.stackexchange.com/questions/67736/what-is-the-difference-between-r-precision-and-precision-at-k/74744
    
    public IntervalHierarchyComparisons(IntervalHierarchy train, IntervalHierarchy test) {
        init(train,test);
    }
    
    public void init(IntervalHierarchy train, IntervalHierarchy test) {
        hierTrain = train;
        hierTest = test;
        
        orderedIntervalsTrain = hierTrain.getOrderedIntervals(); //descending order of 'quality', biggest acc/shortest length
        orderedIntervalsTest = hierTest.getOrderedIntervals();
        
        fullSeriesTrain = hierTrain.getFullSeriesInterval();
        fullSeriesTest = hierTest.getFullSeriesInterval();
        
        Rtrain = hierTrain.getR();
        Rtest = hierTest.getR();
    }
    
    
    public double computeTop1() {
        if (orderedIntervalsTrain.get(0).equals(orderedIntervalsTest.get(0))) 
            top1 = 1.0;
        else 
            top1 = 0.0;
            
        return top1;
    }
    
    public double computeNDCG() throws Exception {
        nDCGatK = computeNDCGatK(orderedIntervalsTest.size());
        return nDCGatK;
    }
    
    public double computeNDCGatK(int k) throws Exception {
        nDCGatK = DCGatK(k)/iDCGatK(k);
        return nDCGatK;
    }
    
    private double DCG() throws Exception {
        return DCGatK(orderedIntervalsTest.size());
    }
    
    private double DCGatK(int k) throws Exception {
        //discounted cumulative gain. for the ordering given by the estimates on train data, 
        //sum the corresponding test/true relevances of the intervals
        
        double DCG = 0.0;
        
        for (int i = 0; i < k; i++) {
            Interval trainInterval = orderedIntervalsTrain.get(i);
            Interval testInterval = getMatchingIntervalFrom(trainInterval, orderedIntervalsTest);
            
            DCG += discountedGain(i+1, testInterval.relevance); //i+1 to one-index
        }
        
        return DCG;
    }
    
    private double iDCG() { 
        //idealDCG, the cumulative discounted relevance of all relevant test intervals only (orderd by relevance), 
        //i.e. only those intervals that actually do improve performance in reality (on test)
        //can get these directly from the orderedintervals
        
        return iDCGatK(orderedIntervalsTest.size());
    }
    
    private double iDCGatK(int k) { 
        //idealDCG, the cumulative discounted relevance of all relevant test intervals only (orderd by relevance), 
        //i.e. only those intervals that actually do improve performance in reality (on test)
        //can get these directly from the orderedintervals
        
        //will sum up to all releva*NT* documents or the k with highest relevance, if k < number relevant documents (i.e.: R)
        
        double iDCG = 0.0;
               
        for (int i = 0; i < k; i++) {
            Interval interval = orderedIntervalsTest.get(i);
            if (interval.equals(fullSeriesTest))
                break;
                  
            iDCG += discountedGain(i+1, interval.relevance);  //i+1 to one-index
        }
        
        return iDCG;
    }
    
    private double discountedGain(int position, double relevance) {
        double numerator = relevance;
        double denominator = (Math.log(position+1) / LOG2);
        return numerator / denominator;
//        return (relevance) / (Math.log(position+1) / LOG2);
//        return ((2*relevance) - 1) / (Math.log(position+1) / LOG2);
    }
    
    static final double LOG2 = Math.log(2);
        
    public Interval getMatchingIntervalFrom(Interval find, List<Interval> intervals) throws Exception {
        for (Interval interval : intervals)
            if (interval.equals(find))
                return interval;
        throw new Exception("Could not find train interval " + find + " in test intervals " + intervals);
    }
    
    public double computeRPrecision() throws Exception { 
        //the proportion of truely relevant intervals within the top R estimated intervals,
        //where are R is the total number of truely relevant intervals
        
        double prec = 0.0;
        for (int i = 0; i < Rtest; i++) {
            Interval trainInterval = orderedIntervalsTrain.get(i);
            Interval testInterval = getMatchingIntervalFrom(trainInterval, orderedIntervalsTest);
            
            if (testInterval.relevance >= 0) //it's relevant
                prec++;
        }
        
        rPrecision = prec/Rtest;
        return rPrecision;
    }
    
    public void computeAllMetrics() throws Exception {
        computeTop1();
        computeRPrecision();
        computeNDCGatK(Rtest);
    }
    
    public static void main(String[] args) throws Exception {
        IntervalHierarchy trainHier = new IntervalHierarchy("train", "E:/Intervals/GunpointExampleAna/locallyCreatedResults/", "ED", "Gunpoint", 0);
        IntervalHierarchy testHier = new IntervalHierarchy("test", "E:/Intervals/GunpointExampleAna/locallyCreatedResults/", "ED", "Gunpoint", 0);
        
        IntervalHierarchyComparisons comps = new IntervalHierarchyComparisons(trainHier, testHier);
        comps.computeAllMetrics();
        
        System.out.println("comps.top1 = " + comps.top1);
        System.out.println("comps.rPrecision = " + comps.rPrecision);
        System.out.println("comps.nDCGatK = " + comps.nDCGatK);
    }
    
}
