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

import evaluation.storage.ClassifierResults;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class IntervalHeirarchy {

    public static final int maxNumDifferentIntervals = 210;
    public static final int maxNumIntervalPoints = 20; //so 21 values really, 0 .. 20 corresponding to props 0 .. 1

    
    Interval[][] heirarchy;
    
    
    public class Interval {
        public int intervalID;
        
        public double[] intervalPercents;
        public double startPercent;
        public double endPercent;
        
        public String intervalStr;
        
        public int startInd;
        public int endInd;
        public int[] intervalInds;
        
        public ClassifierResults res;
        public double score;
        
        public Interval() { 
            
        }
        
        public Interval(double[] intervalPercents) { 
            this.intervalPercents = intervalPercents;
            
            startPercent = intervalPercents[0];
            endPercent = intervalPercents[1];
            
            intervalStr = buildIntervalStr(intervalPercents);
        }
        
        public Interval(int intervalID) throws Exception { 
            this.intervalID = intervalID;
            
            intervalPercents = defineInterval(intervalID);
            
            startPercent = intervalPercents[0];
            endPercent = intervalPercents[1];
            
            intervalStr = buildIntervalStr(intervalPercents);
            
        }
    }
    
    
    public IntervalHeirarchy() throws Exception {
        buildHeirarchy();
    }
    
    public IntervalHeirarchy(String split, String baseResPath, String baseClassifier, String dataset, int fold) throws Exception {
        buildHeirarchy(split, baseResPath, baseClassifier, dataset, fold);
    }
    
    public void buildHeirarchy() throws Exception {
        buildHeirarchy(null, null, null, null, 0);
    }
    
    public void buildHeirarchy(String split, String resultsPath, String baseClassifier, String dataset, int fold) throws Exception {
        
        heirarchy = new Interval[maxNumIntervalPoints][];
        
        int size = maxNumIntervalPoints;
        for (int i = 0; i < maxNumIntervalPoints; i++) {
            heirarchy[i] = new Interval[size--];
        }
        
        
        
        for (int i = 0; i < maxNumDifferentIntervals; i++) {
            Interval interval = new Interval(i);
            int[] inds = findHeirarchyIndices(interval.intervalPercents);
        
            if (resultsPath != null)
                interval.res = new ClassifierResults(resultsPath + buildIntervalClassifierName(baseClassifier, interval.intervalPercents) + "/Predictions/" + dataset + "/testFold"+ fold + ".csv");
            
            heirarchy[inds[0]][inds[i]] = interval;
        }
    }
    
    
    public static int[] findHeirarchyIndices(double[] interval) {
        double rawIntervalLength = interval[1] - interval[0];
        
        int normIntervalLength = (int)(rawIntervalLength / (1 / maxNumIntervalPoints));
        int intervalStart = (int)(interval[0] / (1 / maxNumIntervalPoints));
        
        return new int[] { normIntervalLength, intervalStart };
    }
    
    
    public String toString() { 
        StringBuilder sb = new StringBuilder();
        
        for (int i = maxNumIntervalPoints-1; i >= 0; i--) {
            for (int j = 0; j < heirarchy[i].length; j++)
                sb.append(heirarchy[i][j].intervalStr).append(",");
            sb.append("\n");
            
            for (int j = 0; j < heirarchy[i].length; j++)
                sb.append(heirarchy[i][j].res.getAcc()).append(",");
            sb.append("\n\n");
        }
        
        return sb.toString();
    }
    
    
    
    
    
    
    
    public static String buildIntervalStr(double[] interval) {
        return String.format("%.2f", interval[0]).split("\\.")[1] + "_" + String.format("%.2f", interval[1]).split("\\.")[1];
    }

    
    public static String buildIntervalClassifierName(String classifier, double[] interval) {
        return classifier + "_" + buildIntervalStr(interval);
    }

    public static double[] defineInterval(int intID) throws Exception {
        int startId = 0;
        int endId = 1;
        int c = 0;
        while (c != intID) {
            if (++endId > maxNumIntervalPoints) {
                startId++;
                endId = startId + 1;
                if (startId > maxNumIntervalPoints - 1) {
                    throw new Exception("something wrong in interval defintion, startId=" + startId + " endId=" + endId + " intId=" + intID);
                }
            }
            c++;
        }
        double startProp = (double) startId / maxNumIntervalPoints;
        double endProp = (double) endId / maxNumIntervalPoints;
        return new double[]{startProp, endProp};
    }

    
    
    
    
    
    
    public static void main(String[] args) throws Exception {
        IntervalHeirarchy ih = new IntervalHeirarchy("test", "E:/Intervals/InitialResults/", "ED", "Gunpoint", 0);
        
        System.out.println(ih);
    }
    
}
