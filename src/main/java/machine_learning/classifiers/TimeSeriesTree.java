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
package machine_learning.classifiers;

import weka.classifiers.AbstractClassifier;
import weka.core.*;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Random;

import static utilities.ArrayUtilities.normalise;
import static utilities.ArrayUtilities.sum;

/**
 * Time series tree implementation from the time series forest (TSF) paper.
 *
 * Author: Matthew Middlehurst
 **/
public class TimeSeriesTree extends AbstractClassifier implements Randomizable, Serializable {

    private static double log2 = Math.log(2);

    private boolean useMargin = true;
    private boolean norm = true;
    private int k = 20;

    private int seed = 0;
    private Random rand;

    private TreeNode root;

    private double[] mean;
    private double[] stdev;
    private int numAttributes;

    protected static final long serialVersionUID = 1L;

    public TimeSeriesTree(){}

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.setMinimumNumberInstances(2);

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

    public void setNormalise(boolean b){
        this.norm = b;
    }

    public void setUseMargin(boolean b){
        this.useMargin = b;
    }

    public void setK(int i){
        this.k = i;
    }

    @Override
    public int getSeed() {
        return seed;
    }

    public boolean getNormalise() {
        return norm;
    }

    public double getNormStdev(int idx){
        return stdev[idx];
    }

    public double getNormMean(int idx){
        return mean[idx];
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        numAttributes = data.numAttributes()-1;
        if (data.classIndex() != numAttributes) throw new Exception("Class attribute must be the last index.");

        rand = new Random(seed);

        Instances newData;
        if (norm){
            newData = new Instances(data);
            mean = new double[numAttributes];
            stdev = new double[numAttributes];

            for (int i = 0; i < numAttributes; i++){
                for (Instance inst: newData){
                    mean[i] += inst.value(i);
                }
                mean[i] /= newData.numInstances();

                double squareSum = 0;
                for (Instance inst: newData){
                    double temp = inst.value(i) - mean[i];
                    squareSum += temp * temp;
                }
                stdev[i] = Math.sqrt(squareSum/(newData.numInstances()-1));

                if (stdev[i] == 0) stdev[i] = 1;

                for (Instance inst: newData){
                    inst.setValue(i, (inst.value(i) - mean[i]) / stdev[i]);
                }
            }
        }
        else{
            newData = data;
        }

        double[][] thresholds = findThresholds(newData);

        double[] dist = new double[data.numClasses()];
        for (Instance inst: data){
            dist[(int)inst.classValue()]++;
        }
        double rootEntropy = 0;
        for (int i = 0; i < data.numClasses(); i++) {
            double p1 = dist[i]/data.numInstances();
            rootEntropy += p1 > 0 ? -(p1*Math.log(p1)/log2) : 0;
        }

        root = new TreeNode();
        root.buildTree(newData, thresholds, rootEntropy, dist);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] probs = distributionForInstance(instance);

        int maxClass = 0;
        for (int n = 1; n < probs.length; n++) {
            if (probs[n] > probs[maxClass] || (probs[n] == probs[maxClass] && rand.nextBoolean())) {
                maxClass = n;
            }
        }

        return maxClass;
    }

    public double classifyInstance(Instance instance, ArrayList<double[]> info) throws Exception {
        double[] probs = distributionForInstance(instance, info);

        int maxClass = 0;
        for (int n = 1; n < probs.length; n++) {
            if (probs[n] > probs[maxClass] || (probs[n] == probs[maxClass] && rand.nextBoolean())) {
                maxClass = n;
            }
        }

        return maxClass;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        Instance newInst;
        if (norm){
            newInst = new DenseInstance(instance);
            for (int i = 0; i < numAttributes; i++){
                newInst.setValue(i, (newInst.value(i) - mean[i]) / stdev[i]);
            }
            newInst.setDataset(instance.dataset());
        }
        else{
            newInst = instance;
        }

        return root.distributionForInstance(newInst);
    }

    public double[] distributionForInstance(Instance instance, ArrayList<double[]> info) throws Exception {
        Instance newInst;
        if (norm){
            newInst = new DenseInstance(instance);
            for (int i = 0; i < numAttributes; i++){
                newInst.setValue(i, (newInst.value(i) - mean[i]) / stdev[i]);
            }
            newInst.setDataset(instance.dataset());
        }
        else{
            newInst = instance;
        }

        return root.distributionForInstance(newInst, info);
    }

    private double[][] findThresholds(Instances data){
        double[][] thresholds = new double[numAttributes][k];
        for (int i = 0; i < numAttributes; i++){
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            for (Instance inst: data){
                double v = inst.value(i);
                if (v < min){
                    min = v;
                }
                else if (v > max){
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
        double bestMargin = Double.MIN_VALUE;
        TreeNode[] children;
        double[] leafDistribution;

        protected static final long serialVersionUID = 1L;

        TreeNode(){}

        void buildTree(Instances data, double[][] thresholds, double entropy, double[] distribution){
            double[][] bestEntropies = new double[0][0];

            for (int i = 0; i < numAttributes; i++){
                for (int n = 0; n < k; n++){
                    //gain stored in [0][0]
                    double[][] entropies = entropyGain(data, i, thresholds[i][n], entropy);

                    if (entropies[0][0] > bestGain ||
                            (!useMargin && entropies[0][0] == bestGain && entropies[0][0] > 0 && rand.nextBoolean())){
                        bestSplit = i;
                        bestThreshold = thresholds[i][n];
                        bestGain = entropies[0][0];
                        bestMargin = Double.MIN_VALUE;
                        bestEntropies = entropies;
                    }
                    else if (useMargin && entropies[0][0] == bestGain && entropies[0][0] > 0){
                        double margin = findMargin(data, i, thresholds[i][n]);
                        if (bestMargin == Double.MIN_VALUE) bestMargin = findMargin(data, bestSplit, bestThreshold);

                        if (margin > bestMargin || (margin == bestMargin && rand.nextBoolean())){
                            bestSplit = i;
                            bestThreshold = thresholds[i][n];
                            bestMargin = margin;
                            bestEntropies = entropies;
                        }
                    }
                }
            }

            if (bestSplit > -1){
                Instances[] dataSplit = splitData(data);
                children = new TreeNode[2];
                children[0] = new TreeNode();
                children[0].buildTree(dataSplit[0], thresholds, bestEntropies[0][1], bestEntropies[1]);
                children[1] = new TreeNode();
                children[1].buildTree(dataSplit[1], thresholds, bestEntropies[0][2], bestEntropies[2]);
            }
            else{
                leafDistribution = distribution;
                leafDistribution = normalise(leafDistribution);
            }
        }

        double[][] entropyGain(Instances data, int att, double threshold, double parentEntropy){
            double[][] dists = new double[3][data.numClasses()];
            for (Instance inst: data){
                if (inst.value(att) <= threshold){
                    dists[1][(int)inst.classValue()]++;
                }
                else{
                    dists[2][(int)inst.classValue()]++;
                }
            }

            double sumLeft = sum(dists[1]);
            double sumRight = sum(dists[2]);

            double[] entropies = new double[3];
            for (int i = 0; i < data.numClasses(); i++) {
                double p1 = dists[1][i]/sumLeft;
                entropies[1] += p1 > 0 ? -(p1*Math.log(p1)/log2) : 0;
                double p2 = dists[2][i]/sumRight;
                entropies[2] += p2 > 0 ? -(p2*Math.log(p2)/log2) : 0;
            }

            entropies[0] = parentEntropy - sumLeft/data.numInstances() * entropies[1]
                    - sumRight/data.numInstances() * entropies[2];

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
            Instances[] split = new Instances[2];
            split[0] = new Instances(data, data.numInstances());
            split[1] = new Instances(data, data.numInstances());

            for (Instance inst: data){
                if (inst.value(bestSplit) <= bestThreshold){
                    split[0].add(inst);
                }
                else{
                    split[1].add(inst);
                }
            }

            return split;
        }

        double[] distributionForInstance(Instance inst){
            if (bestSplit > -1) {
                if (inst.value(bestSplit) <= bestThreshold) {
                    return children[0].distributionForInstance(inst);
                } else {
                    return children[1].distributionForInstance(inst);
                }
            }
            else{
                return leafDistribution;
            }
        }

        double[] distributionForInstance(Instance inst, ArrayList<double[]> info){
            if (bestSplit > -1) {
                if (inst.value(bestSplit) <= bestThreshold) {
                    info.add(new double[]{bestSplit, bestThreshold, 0});
                    return children[0].distributionForInstance(inst, info);
                } else {
                    info.add(new double[]{bestSplit, bestThreshold, 1});
                    return children[1].distributionForInstance(inst, info);
                }
            }
            else{
                info.add(leafDistribution);
                return leafDistribution;
            }
        }
    }
}