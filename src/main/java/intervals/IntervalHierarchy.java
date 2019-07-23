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

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.Evaluator;
import evaluation.storage.ClassifierResults;
import intervals.IntervalHierarchy.Interval;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.function.Function;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class IntervalHierarchy implements Iterable<Interval> {

    public static final int maxNumDifferentIntervals = 210;
    public static final int maxNumIntervalPoints = 20; //so 21 values really, 0 .. 20 corresponding to props 0 .. 1

    
    public Interval[][] heirarchy;
    public ArrayList<Interval> orderedEvaluatedIntervals;
    
    public int numIntervalsBetterThanFullSeries; //aka R, in R-Precision etc
    public double propIntervalsBetterThanFullSeries;
    
    public static Function<ClassifierResults, Double> defaultMetric = ClassifierResults.GETTER_Accuracy;
    
    public static class Interval implements Comparable<Interval> {
        public int intervalID;
        
        public double[] intervalPercents;
        public double startPercent;
        public double endPercent;
        public double intervalLength;
        
        public String intervalStr;
        
        public ClassifierResults results;
        public double relevance;
        
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
            
            intervalLength = endPercent - startPercent;
            
            intervalStr = buildIntervalStr(intervalPercents);
            
        }
        
        public void computeRelevance(Interval fullSeriesInterval) {
            computeRelevance(fullSeriesInterval, defaultMetric);
        }
        
        public void computeRelevance(Interval fullSeriesInterval, Function<ClassifierResults, Double> metric) {
            relevance = metric.apply(this.results) - metric.apply(fullSeriesInterval.results);
        }

        public boolean equals(Interval o) {
            return Arrays.equals(this.intervalPercents, o.intervalPercents);
        }
        
        @Override
        public int compareTo(Interval o) {
            // if 'less than' means lower quality, then... 
            
            int c = Double.compare(defaultMetric.apply(this.results), defaultMetric.apply(o.results));
            if (c != 0) // smaller accuracy means less than
                return c;
            else        // same accuracy? then longer length means less than
                return Double.compare(o.intervalLength, this.intervalLength);
        }
    }

    
    /**
     * An iterator that starts at the highest resolution (smallest length) intervals,
     * and then moves up in length. i.e. [0][0], [0][1] .... [1][0], [1][1] ... [maxNumIntervalPoints-1][0]
     */
    public class HeirarchyIterator implements Iterator<Interval> {

        int intervalLengthInd = 0;
        int intervalStartInd = 0;
        
        @Override
        public boolean hasNext() {
            return intervalLengthInd < heirarchy.length && intervalStartInd < heirarchy[intervalLengthInd].length;
        }

        @Override
        public Interval next() {
            Interval res = heirarchy[intervalLengthInd][intervalStartInd];
            
            if (++intervalStartInd == heirarchy[intervalLengthInd].length) {
                intervalStartInd = 0;
                intervalLengthInd++;
            }
            
            return res;
        }
        
    }
    
    @Override
    public Iterator<Interval> iterator() {
        return new HeirarchyIterator();
    }




    
    public IntervalHierarchy() throws Exception {
        buildHeirarchy();
    }
    
    public IntervalHierarchy(String split, String baseResPath, String baseClassifier, String dataset, int fold) throws Exception {
        buildHeirarchy(split, baseResPath, baseClassifier, dataset, fold);
    }
    
    /**
     * Builds an empty hierarchy
     */
    public void buildHeirarchy() throws Exception {
        initHierarchy();
    }
    
    private void initHierarchy() {
        heirarchy = new Interval[maxNumIntervalPoints][];
        
        int size = maxNumIntervalPoints;
        for (int i = 0; i < maxNumIntervalPoints; i++) {
            heirarchy[i] = new Interval[size--];
        }
    }
    
    /**
     * Populates a hierarchy from results files
     */
    public void buildHeirarchy(String split, String resultsPath, String baseClassifier, String dataset, int fold) throws Exception {
        initHierarchy();
        
        for (int i = 0; i < maxNumDifferentIntervals; i++) {
            Interval interval = new Interval(i);
            int[] hierInds = findHeirarchyIndices(interval.intervalPercents);
        
            if (resultsPath != null)
                interval.results = new ClassifierResults(resultsPath + buildIntervalClassifierName(baseClassifier, interval.intervalPercents) + "/Predictions/" + dataset + "/"+split+"Fold"+ fold + ".csv");
            
            if (heirarchy[hierInds[0]][hierInds[1]] != null) {
                throw new Exception("Same heirarchy position already populate, likely double precision error still");
//                Interval lookatme = heirarchy[inds[0]][inds[1]];
//                inds = findHeirarchyIndices(interval.intervalPercents);
            }
            heirarchy[hierInds[0]][hierInds[1]] = interval;
        }
    }
    
    
    /**
     * Builds and populates a hierarchy by evaluating the classifier on each sub-interval of the train set. 
     */
    public void buildHeirarchy(Evaluator eval, Classifier classifier, Instances trainData, boolean normIntervals) throws Exception {
        initHierarchy();
        
        for (int i = 0; i < maxNumDifferentIntervals; i++) {
            Interval interval = new Interval(i);
            int[] hierInds = findHeirarchyIndices(interval.intervalPercents);
            
            Instances intervalData = IntervalCreation.crop_proportional(trainData, interval.startPercent, interval.endPercent, normIntervals);     
            
            ClassifierResults intervalRes = eval.evaluate(classifier, intervalData);
            interval.results = intervalRes;
            
            heirarchy[hierInds[0]][hierInds[1]] = interval;
        }
    }
    
    
    public static int[] findHeirarchyIndices(double[] interval) {
        double rawIntervalLength_precision = interval[1] - interval[0];
        //can encounter double precision problems, e.g 0.15 - 0.1 = 0.499999999
        //round to nearest 1/maxNumIntervalPoints, i.e. nearest 0.05 by default
        double rawIntervalLength = Math.round(rawIntervalLength_precision * (double)maxNumIntervalPoints) / (double)maxNumIntervalPoints;
        
        int normIntervalLength = (int)(Math.round(rawIntervalLength * maxNumIntervalPoints));
        int intervalStart = (int)(Math.round(interval[0] * maxNumIntervalPoints));
        
        return new int[] { normIntervalLength-1, intervalStart }; //-1 to zero-index
    }
    
    
    public String toString() { 
        StringBuilder sb = new StringBuilder();
        
        //headings vals interleaved
        for (int i = maxNumIntervalPoints-1; i >= 0; i--) {
            for (int j = 0; j < heirarchy[i].length; j++)
                sb.append(String.format("%8s, ", heirarchy[i][j].intervalStr));
            sb.append("\n");
            
            for (int j = 0; j < heirarchy[i].length; j++)
                sb.append(String.format("%.6f, ", heirarchy[i][j].results.getAcc()));
            sb.append("\n\n");
        }
        sb.append("\n");
        
        //headings only
        for (int i = maxNumIntervalPoints-1; i >= 0; i--) {
            for (int j = 0; j < heirarchy[i].length; j++)
                sb.append(String.format("%8s, ", heirarchy[i][j].intervalStr));
            sb.append("\n");
        }
        sb.append("\n");
        
        //vals only
        for (int i = maxNumIntervalPoints-1; i >= 0; i--) {
            for (int j = 0; j < heirarchy[i].length; j++)
                sb.append(String.format("%.6f, ", heirarchy[i][j].results.getAcc()));
            sb.append("\n");
        }
        
        return sb.toString();
    }
    
    public double[] getAvgImportances() { 
        return getAvgImportances(defaultMetric);
    }
    
    public double[] getMinImportances() { 
        return getMinImportances(defaultMetric);
    }
    
    public double[] getAvgImportances(Function<ClassifierResults, Double> metric) { 
        double[] importances = new double[maxNumIntervalPoints];
        int[] numTotalOccurences = new int[maxNumIntervalPoints]; //for dividing to find mean at end
        
        //init, highest resolution intervals align with the highest resolution importances we
        //can ascribe, naturally
        Interval[] smallestIntervals = heirarchy[0];
        for (int j = 0; j < maxNumIntervalPoints; j++) {
            importances[j] = metric.apply(smallestIntervals[j].results);
            numTotalOccurences[j] = 1;
        }
        
        for (int i = 1; i < maxNumIntervalPoints; i++) {
            for (int j = 0; j < heirarchy[i].length; j++) {
                Interval outer = heirarchy[i][j];
                
                for (int k = 0; k < maxNumIntervalPoints; k++) {
                    Interval inner = smallestIntervals[k];
                    
                    if (containedWithin(inner, outer)) {
                        importances[k] += metric.apply(outer.results);
                        numTotalOccurences[k]++;
                    }
                }
            }
        }
        
        for (int i = 0; i < maxNumIntervalPoints; i++)
            importances[i] /= numTotalOccurences[i];
        
        return importances;
    }
    
    public double[] getMinImportances(Function<ClassifierResults, Double> metric) { 
        double[] importances = new double[maxNumIntervalPoints];
        
        //init, highest resolution intervals align with the highest resolution importances we
        //can ascribe, naturally
        Interval[] smallestIntervals = heirarchy[0];
        for (int j = 0; j < maxNumIntervalPoints; j++)
            importances[j] = metric.apply(smallestIntervals[j].results);
        
        for (int i = 1; i < maxNumIntervalPoints; i++) {
            for (int j = 0; j < heirarchy[i].length; j++) {
                Interval outer = heirarchy[i][j];
                
                for (int k = 0; k < maxNumIntervalPoints; k++) {
                    Interval inner = smallestIntervals[k];
                    
                    if (containedWithin(inner, outer)) {
                        double score = metric.apply(outer.results);
                        if (score < importances[k])
                            importances[k] = score;
                    }
                }
            }
        }
        
        return importances;
    }
    
    /**
     * Intended for usage with the target's results. Does not include the time to estimate 
     * the error of each interval, only the build time when doing a full build on each
     * 
     * @return the total build time of all intervals, in nanoseconds
     */
    public long getTotalHeirarchyBuildTime() { 
        long totalTime = 0;
        
        for (Interval interval : this)
            totalTime += interval.results.getBuildTimeInNanos();
        
        return totalTime;
    }
    
    /**
     * Intended for usage with the proxy/surrogate's results. Does not include the time 
     * to rebuild fully on the intervals
     * 
     * @return the total time to estimate performance of all intervals, in nanoseconds
     */
    public long getTotalHeirarchyEstimateTime() { 
        long totalTime = 0;
        
        for (Interval interval : this)
            totalTime += interval.results.getErrorEstimateTime();
        
        return totalTime;
    }
    
    /**
     * @return build time of the full series 'interval', in nanoseconds
     */
    public long getFullSeriesBuildTime() {
        return getFullSeriesInterval().results.getBuildTimeInNanos();
    }
    
    /**
     * @return time to estimate performance on the full series 'interval', in nanoseconds
     */
    public long getFullSeriesEstimateTime() {
        return getFullSeriesInterval().results.getErrorEstimateTime();
    }
    
    public Interval getFullSeriesInterval() { 
        return heirarchy[maxNumIntervalPoints-1][0];
    }
    
    
    public ArrayList<Interval> getOrderedIntervals() { 
        if (orderedEvaluatedIntervals == null)
            computeHierarchyEval();
        
        return orderedEvaluatedIntervals;
    }
    
    public int getR() {
        if (orderedEvaluatedIntervals == null)
            computeHierarchyEval();
        
        return numIntervalsBetterThanFullSeries;
    }
    
    public Interval getBestInterval() {
        if (orderedEvaluatedIntervals == null)
            computeHierarchyEval();
        
        return orderedEvaluatedIntervals.get(0);
    }
    
    public void computeHierarchyEval() {
        Interval fullSeries = getFullSeriesInterval();
        
        orderedEvaluatedIntervals = new ArrayList<>();
        for (Interval interval : this)
            orderedEvaluatedIntervals.add(interval);
        
        Collections.sort(orderedEvaluatedIntervals);
        Collections.reverse(orderedEvaluatedIntervals); //todo fix to do descending in one
        
        numIntervalsBetterThanFullSeries = 0;
        for (Interval interval : orderedEvaluatedIntervals) {
            interval.computeRelevance(fullSeries);
            if (interval.relevance > 0)
                numIntervalsBetterThanFullSeries++;
        }
        propIntervalsBetterThanFullSeries = (double) numIntervalsBetterThanFullSeries / orderedEvaluatedIntervals.size();
    }
    
    private boolean containedWithin(Interval inner, Interval outter) { 
        return inner.startPercent >= outter.startPercent && inner.endPercent <= outter.endPercent;
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
        IntervalHierarchy ih = new IntervalHierarchy("train", "E:/Intervals/GunpointExampleAna/locallyCreatedResults/", "ED", "Gunpoint", 0);
        
        System.out.println(ih);
        
        System.out.println("\n\n");
        
        System.out.println("avg");
        System.out.println(Arrays.toString(ih.getAvgImportances()));
        System.out.println("min");
        System.out.println(Arrays.toString(ih.getMinImportances()));
        
        System.out.println("\n\n");
        
        System.out.println("Heirarchy build time: " + ih.getTotalHeirarchyBuildTime());
        System.out.println("Fullseries build time: " + ih.getFullSeriesBuildTime());
    }
    
}
