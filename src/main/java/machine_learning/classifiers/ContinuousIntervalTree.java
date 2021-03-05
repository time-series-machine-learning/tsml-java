/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
package machine_learning.classifiers;

import experiments.data.DatasetLoading;
import weka.classifiers.AbstractClassifier;
import weka.core.*;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;
import java.util.function.Function;

import static utilities.ArrayUtilities.normalise;
import static utilities.ArrayUtilities.sum;

/**
 * A tree for time series interval forests.
 * Based on the time series tree (TST) implementation from the time series forest (TSF) paper.
 *
 * @author Matthew Middlehurst
 **/
public class ContinuousIntervalTree extends AbstractClassifier implements Randomizable, Serializable {

    private static double log2 = Math.log(2);

    //Margin gain from TSF paper
    private boolean useMargin = true;
    //Number of thresholds to try for attribute splits
    private int k = 20;
    //Max tree depth
    private int maxDepth = Integer.MAX_VALUE;

    private int seed = 0;
    private Random rand;

    private TreeNode root;

    private int numAttributes;

    protected static final long serialVersionUID = 2L;

    public ContinuousIntervalTree(){}

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.setMinimumNumberInstances(2);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // attributes
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        return result;
    }

    @Override
    public void setSeed(int seed) {
        this.seed = seed;
    }

    public void setUseMargin(boolean b){
        this.useMargin = b;
    }

    public void setK(int i){
        this.k = i;
    }

    public void setMaxDepth(int i) { this.maxDepth = i; }

    @Override
    public int getSeed() {
        return seed;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        numAttributes = data.numAttributes()-1;
        if (data.classIndex() != numAttributes) throw new Exception("Class attribute must be the last index.");

        rand = new Random(seed);

        //thresholds for each attribute
        double[][] thresholds = findThresholds(data);

        //Initial tree node setup
        double[] dist = new double[data.numClasses()];
        for (Instance inst: data){
            dist[(int)inst.classValue()]++;
        }
        double rootEntropy = 0;
        for (int i = 0; i < data.numClasses(); i++){
            double p = dist[i]/data.numInstances();
            rootEntropy += p > 0 ? -(p*Math.log(p)/log2) : 0;
        }

        root = new TreeNode();
        root.buildTree(data, thresholds, rootEntropy, dist, -1, false);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] probs = distributionForInstance(instance);
        return tieBreak(probs);
    }

    public double classifyInstance(double[][] instance, Function<Interval, Double>[] functions,
                                   int[][] intervals, int[] attributes,
                                   int[] dimensions) throws Exception {
        double[] probs = distributionForInstance(instance, functions, intervals, attributes, dimensions);
        return tieBreak(probs);
    }

    public double classifyInstance(double[][][] instance, Function<Interval, Double>[] functions,
                                   int[][][] intervals, int[] attributes,
                                   int[][] dimensions) throws Exception {
        double[] probs = distributionForInstance(instance, functions, intervals, attributes, dimensions);
        return tieBreak(probs);
    }

    public double classifyInstance(double[][] instance, Function<Interval, Double>[] functions,
                                   int[][] intervals, int[] attributes,
                                   int[] dimensions, ArrayList<double[]> info) throws Exception {
        double[] probs = distributionForInstance(instance, functions, intervals, attributes, dimensions, info);
        return tieBreak(probs);
    }

    public double classifyInstance(double[][][] instance, Function<Interval, Double>[] functions,
                                   int[][][] intervals, int[] attributes,
                                   int[][] dimensions, ArrayList<double[]> info) throws Exception {
        double[] probs = distributionForInstance(instance, functions, intervals, attributes, dimensions, info);
        return tieBreak(probs);
    }

    private int tieBreak(double[] probs){
        int maxClass = 0;
        for (int n = 1; n < probs.length; n++){
            if (probs[n] > probs[maxClass] || (probs[n] == probs[maxClass] && rand.nextBoolean())){
                maxClass = n;
            }
        }
        return maxClass;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return root.distributionForInstance(instance);
    }

    //For interval forests, transforms the time series at the node level to save time on predictions (CIF)
    public double[] distributionForInstance(double[][] instance, Function<Interval, Double>[] functions,
                                            int[][] intervals, int[] attributes,
                                            int[] dimensions) throws Exception {
        return root.distributionForInstance(instance, functions, intervals, attributes, dimensions);
    }

    //For interval forests with multiple representations, transforms the time series at the node level to save time on
    //predictions (DrCIF)
    public double[] distributionForInstance(double[][][] instance, Function<Interval, Double>[] functions,
                                            int[][][] intervals, int[] attributes,
                                            int[][] dimensions) throws Exception {
        return root.distributionForInstance(instance, functions, intervals, attributes, dimensions);
    }

    //Fills the info List with the attribute, threshold and where it next moved for each node traversed
    public double[] distributionForInstance(double[][] instance, Function<Interval, Double>[] functions,
                                            int[][] intervals, int[] attributes,
                                            int[] dimensions, ArrayList<double[]> info) throws Exception {
        return root.distributionForInstance(instance, functions, intervals, attributes, dimensions, info);
    }

    public double[] distributionForInstance(double[][][] instance, Function<Interval, Double>[] functions,
                                            int[][][] intervals, int[] attributes,
                                            int[][] dimensions, ArrayList<double[]> info) throws Exception {
        return root.distributionForInstance(instance, functions, intervals, attributes, dimensions, info);
    }

    private double[][] findThresholds(Instances data){
        double[][] thresholds = new double[numAttributes][k];
        for (int i = 0; i < numAttributes; i++){
            double min = Double.MAX_VALUE;
            double max = -99999999;
            for (Instance inst: data){
                double v = inst.value(i);
                if (v < min){
                    min = v;
                }
                if (v > max){
                    max = v;
                }
            }

            double step = (max - min)/(k-1);
            for (int n = 0; n < k; n++){
                thresholds[i][n] = min + step*n;
            }
        }
        return thresholds;
    }

    //Returns the attribute used for each node and its information gain
    public ArrayList<Double>[] getTreeSplitsGain(){
        ArrayList<Double> splits = new ArrayList<>();
        ArrayList<Double> gain = new ArrayList<>();

        if (root.bestSplit > -1) findSplitsGain(root, splits, gain);

        ArrayList<Double>[] r = new ArrayList[2];
        r[0] = splits;
        r[1] = gain;
        return r;
    }

    private void findSplitsGain(TreeNode tree, ArrayList<Double> splits, ArrayList<Double> gain){
        splits.add((double)tree.bestSplit);
        gain.add(tree.bestGain);

        for (int i = 0; i < tree.children.length; i++){
            if (tree.children[i].bestSplit > -1){
                findSplitsGain(tree.children[i], splits, gain);
            }
        }
    }

    //Returns true for attributes which are used in tree nodes, false otherwise
    public boolean[] getAttributesUsed(){
        boolean[] attsUsed = new boolean[numAttributes];
        if (root.bestSplit > -1) findAttributesUsed(root, attsUsed);
        return attsUsed;
    }

    private void findAttributesUsed(TreeNode tree, boolean[] attsUsed){
        if (!attsUsed[tree.bestSplit]){
            attsUsed[tree.bestSplit] = true;
        }

        for (int i = 0; i < tree.children.length; i++){
            if (tree.children[i].bestSplit > -1){
                findAttributesUsed(tree.children[i], attsUsed);
            }
        }
    }

    private class TreeNode implements Serializable {
        int bestSplit = -1;
        double bestThreshold = 0;
        double bestGain = 0;
        double bestMargin = 0;
        TreeNode[] children;
        double[] leafDistribution;
        int depth;

        protected static final long serialVersionUID = 1L;

        TreeNode(){}

        void buildTree(Instances data, double[][] thresholds, double entropy, double[] distribution, int lastDepth,
                       boolean leaf){
            double[][] bestEntropies = new double[0][0];
            depth = lastDepth + 1;

            int remainingClasses = 0;
            for (double d: distribution){
                if (d > 0) remainingClasses++;
            }

            if (!leaf && remainingClasses > 1 && depth < maxDepth){
                //Loop through all attributes each using k threshold values looking the best split for this node
                for (int i = 0; i < numAttributes; i++){
                    for (int n = 0; n < k; n++){
                        //gain stored in [0][0]
                        double[][] entropies = entropyGain(data, i, thresholds[i][n], entropy);

                        if (entropies[0][0] > bestGain || (!useMargin && entropies[0][0] == bestGain && entropies[0][0]
                                > 0 && rand.nextBoolean())){
                            bestSplit = i;
                            bestThreshold = thresholds[i][n];
                            bestGain = entropies[0][0];
                            bestMargin = 0;
                            bestEntropies = entropies;
                        }
                        //Use margin gain if there is a tie
                        else if (useMargin && entropies[0][0] == bestGain && entropies[0][0] > 0){
                            double margin = findMargin(data, i, thresholds[i][n]);
                            if (bestMargin == 0) bestMargin = findMargin(data, bestSplit, bestThreshold);

                            //Select randomly if there is a tie again
                            if (margin > bestMargin || (margin == bestMargin && rand.nextBoolean())){
                                bestSplit = i;
                                bestThreshold = thresholds[i][n];
                                bestMargin = margin;
                                bestEntropies = entropies;
                            }
                        }
                    }
                }
            }

            if (bestSplit > -1){
                Instances[] split = splitData(data);
                children = new TreeNode[3];

                //Left node
                children[0] = new TreeNode();
                if (split[0].isEmpty()){
                    children[0].buildTree(split[0], thresholds, entropy, distribution, depth,true);
                }
                else{
                    children[0].buildTree(split[0], thresholds, bestEntropies[0][1], bestEntropies[1], depth,false);
                }

                //Right node
                children[1] = new TreeNode();
                if (split[1].isEmpty()){
                    children[1].buildTree(split[1], thresholds, entropy, distribution, depth,true);
                }
                else{
                    children[1].buildTree(split[1], thresholds, bestEntropies[0][2], bestEntropies[2], depth,false);
                }

                //Missing value node
                children[2] = new TreeNode();
                if (split[2].isEmpty()){
                    children[2].buildTree(split[2], thresholds, entropy, distribution, depth,true);
                }
                else{
                    children[2].buildTree(split[2], thresholds, bestEntropies[0][3], bestEntropies[3], depth,false);
                }
            }
            else{
                leafDistribution = distribution;
                leafDistribution = normalise(leafDistribution);
            }
        }

        double[][] entropyGain(Instances data, int att, double threshold, double parentEntropy){
            double[][] dists = new double[4][data.numClasses()];
            for (Instance inst: data){
                if (Double.isNaN(inst.value(att))){
                    dists[3][(int)inst.classValue()]++;
                }
                else if (inst.value(att) <= threshold){
                    dists[1][(int)inst.classValue()]++;
                }
                else{
                    dists[2][(int)inst.classValue()]++;
                }
            }

            double sumLeft = sum(dists[1]);
            double sumRight = sum(dists[2]);
            double sumMissing = sum(dists[3]);

            double[] entropies = new double[4];
            for (int i = 0; i < data.numClasses(); i++){
                double p1 = sumLeft > 0 ? dists[1][i]/sumLeft : 0;
                entropies[1] += p1 > 0 ? -(p1*Math.log(p1)/log2) : 0;
                double p2 = sumRight > 0 ? dists[2][i]/sumRight : 0;
                entropies[2] += p2 > 0 ? -(p2*Math.log(p2)/log2) : 0;
                double p3 = sumMissing > 0 ? dists[3][i]/sumMissing : 0;
                entropies[3] += p3 > 0 ? -(p3*Math.log(p3)/log2) : 0;
            }

            entropies[0] = parentEntropy
                    - sumLeft/data.numInstances() * entropies[1]
                    - sumRight/data.numInstances() * entropies[2]
                    - sumMissing/data.numInstances() * entropies[3];

            dists[0] = entropies;

            return dists;
        }

        double findMargin(Instances data, int att, double threshold){
            double min = Double.MAX_VALUE;

            for (Instance inst: data){
                double n = Math.abs(inst.value(att)-threshold);
                if (n < min){
                    min = n;
                }
            }

            return min;
        }

        Instances[] splitData(Instances data){
            Instances[] split = new Instances[3];
            split[0] = new Instances(data, data.numInstances());
            split[1] = new Instances(data, data.numInstances());
            split[2] = new Instances(data, data.numInstances());

            for (Instance inst: data){
                if (Double.isNaN(inst.value(bestSplit))){
                    split[2].add(inst);
                }
                else if (inst.value(bestSplit) <= bestThreshold){
                    split[0].add(inst);
                }
                else{
                    split[1].add(inst);
                }
            }

            return split;
        }

        double[] distributionForInstance(Instance inst){
            if (bestSplit > -1){
                if (Double.isNaN(inst.value(bestSplit))){
                    return children[2].distributionForInstance(inst);
                }
                else if (inst.value(bestSplit) <= bestThreshold){
                    return children[0].distributionForInstance(inst);
                }
                else{
                    return children[1].distributionForInstance(inst);
                }
            }
            else{
                return leafDistribution;
            }
        }

        double[] distributionForInstance(double[][] inst, Function<Interval, Double>[] functions,
                                         int[][] intervals, int[] attributes, int[] dimensions){
            if (bestSplit > -1){
                int interval = bestSplit/attributes.length;
                int att = bestSplit%attributes.length;
                int dim = dimensions[interval];

                double val = functions[attributes[att]].apply(new Interval(inst[dim], intervals[interval][0],
                        intervals[interval][1]));

                if (Double.isNaN(val)){
                    return children[2].distributionForInstance(inst, functions, intervals, attributes, dimensions);
                }
                else if (val <= bestThreshold){
                    return children[0].distributionForInstance(inst, functions, intervals, attributes, dimensions);
                }
                else{
                    return children[1].distributionForInstance(inst, functions, intervals, attributes, dimensions);
                }
            }
            else{
                return leafDistribution;
            }
        }

        double[] distributionForInstance(double[][][] inst, Function<Interval, Double>[] functions,
                                         int[][][] intervals, int[] attributes, int[][] dimensions){
            if (bestSplit > -1){
                int repSum = 0;
                int rep = -1;
                for (int i = 0; i < intervals.length; i++) {
                    if (bestSplit < repSum + attributes.length * intervals[i].length){
                        rep = i;
                        break;
                    }
                    repSum += attributes.length * intervals[i].length;
                }

                int att = bestSplit%attributes.length;
                int interval = (bestSplit-repSum)/attributes.length;
                int dim = dimensions[rep][interval];

                double val = functions[attributes[att]].apply(new Interval(inst[rep][dim], intervals[rep][interval][0],
                        intervals[rep][interval][1]));

                if (Double.isNaN(val)){
                    return children[2].distributionForInstance(inst, functions, intervals, attributes, dimensions);
                }
                else if (val <= bestThreshold){
                    return children[0].distributionForInstance(inst, functions, intervals, attributes, dimensions);
                }
                else{
                    return children[1].distributionForInstance(inst, functions, intervals, attributes, dimensions);
                }
            }
            else{
                return leafDistribution;
            }
        }

        double[] distributionForInstance(double[][] inst, Function<Interval, Double>[] functions,
                                         int[][] intervals, int[] attributes, int[] dimensions,
                                         ArrayList<double[]> info){
            if (bestSplit > -1){
                int interval = bestSplit/attributes.length;
                int att = bestSplit%attributes.length;
                int dim = dimensions[interval];

                double val = functions[attributes[att]].apply(new Interval(inst[dim], intervals[interval][0],
                        intervals[interval][1]));

                if (Double.isNaN(val)){
                    info.add(new double[]{bestSplit, bestThreshold, 2});
                    return children[2].distributionForInstance(inst, functions, intervals, attributes, dimensions,
                            info);
                }
                else if (val <= bestThreshold){
                    info.add(new double[]{bestSplit, bestThreshold, 0});
                    return children[0].distributionForInstance(inst, functions, intervals, attributes, dimensions,
                            info);
                }
                else{
                    info.add(new double[]{bestSplit, bestThreshold, 1});
                    return children[1].distributionForInstance(inst, functions, intervals, attributes, dimensions,
                            info);
                }
            }
            else{
                info.add(leafDistribution);
                return leafDistribution;
            }
        }

        double[] distributionForInstance(double[][][] inst, Function<Interval, Double>[] functions,
                                         int[][][] intervals, int[] attributes, int[][] dimensions,
                                         ArrayList<double[]> info){
            if (bestSplit > -1){
                int repSum = 0;
                int rep = -1;
                for (int i = 0; i < intervals.length; i++) {
                    if (bestSplit < repSum + attributes.length * intervals[i].length){
                        rep = i;
                        break;
                    }
                    repSum += attributes.length * intervals[i].length;
                }

                int att = bestSplit%attributes.length;
                int interval = (bestSplit-repSum)/attributes.length;
                int dim = dimensions[rep][interval];

                double val = functions[attributes[att]].apply(new Interval(inst[rep][dim], intervals[rep][interval][0],
                        intervals[rep][interval][1]));

                if (Double.isNaN(val)){
                    info.add(new double[]{bestSplit, bestThreshold, 2});
                    return children[2].distributionForInstance(inst, functions, intervals, attributes, dimensions,
                            info);
                }
                else if (val <= bestThreshold){
                    info.add(new double[]{bestSplit, bestThreshold, 0});
                    return children[0].distributionForInstance(inst, functions, intervals, attributes, dimensions,
                            info);
                }
                else{
                    info.add(new double[]{bestSplit, bestThreshold, 1});
                    return children[1].distributionForInstance(inst, functions, intervals, attributes, dimensions,
                            info);
                }
            }
            else{
                info.add(leafDistribution);
                return leafDistribution;
            }
        }

        @Override
        public String toString(){
            return "[" + bestSplit + "," + depth + "]";
        }
    }

    public static class Interval {
        public double[] series;
        public int start;
        public int end;

        public Interval(double[] series, int start, int end){
            this.series = series;
            this.start = start;
            this.end = end;
        }
    }

    public static void main(String[] args) throws Exception {
        int fold = 0;

        Instances[] data = DatasetLoading.sampleItalyPowerDemand(fold);
        Instances train = data[0];
        Instances test = data[1];

        ContinuousIntervalTree c = new ContinuousIntervalTree();
        c.buildClassifier(train);
    }
}