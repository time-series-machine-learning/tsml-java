package machine_learning.classifiers;

import weka.classifiers.AbstractClassifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;

import java.util.ArrayList;
import java.util.Random;

import static utilities.ArrayUtilities.normalise;
import static utilities.ArrayUtilities.sum;

public class TimeSeriesTree extends AbstractClassifier implements Randomizable {

    private static double log2 = Math.log(2);

    private boolean norm = true;
    private boolean useEntrance = true;
    private int k = 20;

    private int seed = 0;
    private Random rand;

    private TreeNode root;
    private double[] mean;
    private double[] stdev;

    public TimeSeriesTree(){}

    @Override
    public void setSeed(int seed) {
        this.seed = seed;
    }

    @Override
    public int getSeed() {
        return seed;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes()-1) throw new Exception("Class attribute must be the last index.");

        rand = new Random(seed);

        Instances newData;
        if (norm){
            newData = new Instances(data);
            mean = new double[newData.numAttributes()-1];
            stdev = new double[newData.numAttributes()-1];

            for (int i = 0; i < newData.numAttributes()-1; i++){
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
        for (int n = 1; n < probs.length; ++n) {
            if (probs[n] > probs[maxClass]) {
                maxClass = n;
            }
            else if (probs[n] == probs[maxClass]){
                if (rand.nextBoolean()){
                    maxClass = n;
                }
            }
        }

        return maxClass;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        Instance newInst;
        if (norm){
            newInst = new DenseInstance(instance);
            for (int i = 0; i < newInst.numAttributes()-1; i++){
                newInst.setValue(i, (newInst.value(i) - mean[i]) / stdev[i]);
            }
            newInst.setDataset(instance.dataset());
        }
        else{
            newInst = instance;
        }

        return root.distributionForInstance(newInst);
    }

    private double[][] findThresholds(Instances data){
        double[][] thresholds = new double[data.numAttributes()-1][k];
        for (int i = 0; i < data.numAttributes()-1; i++){
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

        findSplitsGain(root, splits, gain);

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

    private class TreeNode {
        int bestSplit = -1;
        double bestThreshold = 0;
        double bestGain = 0;
        double bestMargin = 0;
        TreeNode[] children;
        double[] leafDistribution;

        TreeNode(){}

        void buildTree(Instances data, double[][] thresholds, double entropy, double[] distribution){
            double[][] bestEntropies = new double[0][0];

            for (int i = 0; i < data.numAttributes()-1; i++){
                for (int n = 0; n < k; n++){
                    //gain stored in [0][0]
                    double[][] entropies = entropyGain(data, i, thresholds[i][n], entropy);

                    if (useEntrance){
                        if (entropies[0][0] > bestGain){
                            bestSplit = i;
                            bestThreshold = thresholds[i][n];
                            bestGain = entropies[0][0];
                            bestMargin = findMargin(data, i, thresholds[i][n]);
                            bestEntropies = entropies;
                        }
                        else if (entropies[0][0] == bestGain && entropies[0][0] > 0){
                            double margin = findMargin(data, i, thresholds[i][n]);

                            if (margin > bestMargin || (margin == bestMargin && rand.nextBoolean())){
                                bestSplit = i;
                                bestThreshold = thresholds[i][n];
                                bestMargin = margin;
                                bestEntropies = entropies;
                            }
                        }
                    }
                    else{
                        if (entropies[0][0] > bestGain ||
                                (entropies[0][0] == bestGain && entropies[0][0] > 0 && rand.nextBoolean())){
                            bestSplit = i;
                            bestThreshold = thresholds[i][n];
                            bestGain = entropies[0][0];
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
                normalise(leafDistribution);
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
    }
}
