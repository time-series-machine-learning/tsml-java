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
import java.util.Collections;

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

    IntervalHierarchy hierA;
    IntervalHierarchy hierB;
    
    ArrayList<Interval> orderedIntervalsA;
    ArrayList<Interval> orderedIntervalsB;
    
    Interval fullSeriesA; 
    Interval fullSeriesB;
    
    int Ra;     // data driven para, how many intervals improve over full series, i.e. how many are 'relevant' for a/the proxy
    int Rb;     // and for b/the target
    int k = 5; // user para, how many intervals at the top do we care about? informed by e.g. what if we ensemble top k
    
    double nDCG;
    double nDCGatK; //normed discounted cumulative gain at K, default K = 5
    double top1; //is the best interval estimated in a the same interval observed to be best in b? 
    double rPrecision; //aka P@R where R = number of intervals better than full series, P = prop intervals better than full series
    
    public IntervalHierarchyComparisons(IntervalHierarchy a, IntervalHierarchy b) {
        init(a,b);
    }
    
    public void init(IntervalHierarchy a, IntervalHierarchy b) {
        hierA = a;
        hierB = b;
        
        orderedIntervalsA = hierA.getOrderedIntervals(); //descending order of 'quality', biggest acc/shortest length
        orderedIntervalsB = hierB.getOrderedIntervals();
        
        fullSeriesA = hierA.getFullSeriesInterval();
        fullSeriesB = hierB.getFullSeriesInterval();
        
        Ra = hierA.getHierarchyEvaluation().numIntervalsBetterThanFullSeries;
        Rb = hierB.getHierarchyEvaluation().numIntervalsBetterThanFullSeries;
    }
    
    
    public double computeAtoB_Top1IntervalAccuracy() {
        if (orderedIntervalsA.get(0).intervalPercents.equals(orderedIntervalsB.get(0).intervalPercents)) 
            top1 = 1.0;
        else 
            top1 = 0.0;
            
        return top1;
    }
    
    public double computeAtoB_NDCG() {
        nDCG = DCG()/iDCG();
        return nDCG;
    }
    
    private double DCG() {
        double DCG = 0.0;
        
        return DCG;
    }
    
    private double iDCG() { 
        double iDCG = 0.0;
        
        return iDCG;
    }
    
    
    public static void main(String[] args) {
        
    }
    
}
