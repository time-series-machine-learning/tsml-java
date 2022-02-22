package machine_learning.clusterers.consensus;

import evaluation.storage.ClustererResults;
import experiments.data.DatasetLoading;
import machine_learning.clusterers.KMeans;
import tsml.clusterers.EnhancedAbstractClusterer;
import utilities.GenericTools;
import weka.clusterers.NumberOfClustersRequestable;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import static utilities.ArrayUtilities.sum;
import static utilities.ArrayUtilities.unique;
import static utilities.ClusteringUtilities.randIndex;
import static utilities.GenericTools.max;
import static utilities.Utilities.argMax;

public class ACE extends ConsensusClusterer implements LoadableConsensusClusterer, NumberOfClustersRequestable {

    private double alpha = 0.8;
    private double alphaIncrement = 0.1;
    private double alphaMin = 0.6;
    private double alpha2 = 0.7;
    private int k = 2;

    private int[] newLabels;
    private Random rand;

    public ACE(EnhancedAbstractClusterer[] clusterers) {
        super(clusterers);
    }

    public ACE(ArrayList<EnhancedAbstractClusterer> clusterers) {
        super(clusterers);
    }

    @Override
    public int numberOfClusters() throws Exception {
        return k;
    }

    @Override
    public void setNumClusters(int numClusters) throws Exception {
        k = numClusters;
    }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        super.buildClusterer(data);

        if (buildClusterers){
            for (EnhancedAbstractClusterer clusterer: clusterers){
                clusterer.buildClusterer(data);
            }
        }

        ArrayList<Integer>[][] ensembleClusters = new ArrayList[clusterers.length][];
        for (int i = 0; i < ensembleClusters.length; i++) {
            ensembleClusters[i] = clusterers[i].getClusters();
        }

        buildEnsemble(ensembleClusters, data.numInstances());
    }

    @Override
    public void buildFromFile(String[] directoryPaths) throws Exception {
        ArrayList<Integer>[][] ensembleClusters = new ArrayList[directoryPaths.length][];
        int numInstances = -1;

        for (int i = 0; i < directoryPaths.length; i++) {
            ClustererResults r = new ClustererResults(directoryPaths[i] + "trainFold" + seed + ".csv");
            if (i == 0)
                numInstances = r.numInstances();

            ensembleClusters[i] = new ArrayList[r.getNumClusters()];

            for (int n = 0; n < ensembleClusters[i].length; n++) {
                ensembleClusters[i][n] = new ArrayList();
            }

            int[] fileAssignments = r.getClusterValuesAsIntArray();
            for (int n = 0; n < fileAssignments.length; n++) {
                ensembleClusters[i][fileAssignments[n]].add(n);
            }
        }

        buildEnsemble(ensembleClusters, numInstances);
    }

    @Override
    public int clusterInstance(Instance inst) throws Exception {
        double[] dist = distributionForInstance(inst);
        return argMax(dist, rand);
    }

    @Override
    public double[] distributionForInstance(Instance inst) throws Exception {
        Instance newInst = copyInstances ? new DenseInstance(inst) : inst;
        int clsIdx = inst.classIndex();
        if (clsIdx >= 0){
            newInst.setDataset(null);
            newInst.deleteAttributeAt(clsIdx);
        }

        //todo checkout with certainty stuff
        double[] dist = new double[k];
        int offset = 0;
        for (EnhancedAbstractClusterer clusterer : clusterers) {
            dist[offset + clusterer.clusterInstance(inst)]++;
            offset += clusterer.numberOfClusters();
        }

        for (int i = 0; i < dist.length; i++){
            dist[i] /= clusterers.length;
        }

        return dist;
    }

    @Override
    public int[] clusterFromFile(String[] directoryPaths) throws Exception {
        double[][] dists = distributionFromFile(directoryPaths);

        int[] arr = new int[dists.length];
        for (int i = 0; i < dists.length; i++) {
            arr[i] = argMax(dists[i], rand);
        }

        return arr;
    }

    @Override
    public double[][] distributionFromFile(String[] directoryPaths) throws Exception {
        int[][] ensembleAssignments = new int[directoryPaths.length][];

        int offset = 0;
        for (int i = 0; i < directoryPaths.length; i++) {
            ClustererResults r = new ClustererResults(directoryPaths[i] + "testFold" + seed + ".csv");
            ensembleAssignments[i] = r.getClusterValuesAsIntArray();

            for (int n = 0; n < ensembleAssignments[i].length; n++) {
                ensembleAssignments[i][n] = newLabels[offset + ensembleAssignments[i][n]];
            }

            offset += r.getNumClusters();
        }

        double[][] dists = new double[ensembleAssignments[0].length][k];
        for (int i = 0; i < dists.length; i++) {
            for (int[] clusterAssignments : ensembleAssignments) {
                dists[i][clusterAssignments[i]]++;
            }

            for (int n = 0; n < dists[n].length; n++){
                dists[i][n] /= ensembleAssignments.length;
            }
        }

        return dists;
    }

    private void buildEnsemble(ArrayList<Integer>[][] ensembleClusters, int numInstances) throws Exception {
        if (!seedClusterer) {
            rand = new Random();
        } else {
            rand = new Random(seed);
        }

        int clusterCount = 0;
        ArrayList<ArrayList<double[]>> binaryClusterMembership = new ArrayList<>(ensembleClusters.length);
        for (ArrayList<Integer>[] memberClusters: ensembleClusters){
            clusterCount += memberClusters.length;
            ArrayList<double[]> binaryClusters = new ArrayList<>(memberClusters.length);
            binaryClusterMembership.add(binaryClusters);

            for (ArrayList<Integer> memberCluster : memberClusters) {
                double[] binaryCluster = new double[numInstances];
                binaryClusters.add(binaryCluster);
                for (int n : memberCluster) {
                    binaryCluster[n] = 1;
                }
            }
        }

        if (k > clusterCount){
            throw new Exception("K is greater than the total number of clusters in the ensemble.");
        }

        double newAlpha = alpha;

        newLabels = new int[clusterCount];
        for (int i = 1; i < clusterCount; i++){
            newLabels[i] = i;
        }

        boolean stage1 = true;
        double[][] clusterSimilarities = null;
        while (true) {
            // find the similarity of each cluster from different ensemble members
            int countI = 0;
            clusterSimilarities = new double[clusterCount][];
            for (int i = 0; i < binaryClusterMembership.size(); i++) {
                for (int n = 0; n < binaryClusterMembership.get(i).size(); n++) {
                    clusterSimilarities[countI + n] = new double[countI];
                }

                int countN = 0;
                for (int n = 0; n < i; n++) {
                    for (int j = 0; j < binaryClusterMembership.get(i).size(); j++) {
                        for (int k = 0; k < binaryClusterMembership.get(n).size(); k++) {
                            clusterSimilarities[countI + j][countN + k] = setCorrelation(
                                    binaryClusterMembership.get(i).get(j), binaryClusterMembership.get(n).get(k),
                                    numInstances);
                        }
                    }
                    countN += binaryClusterMembership.get(n).size();
                }
                countI += binaryClusterMembership.get(i).size();
            }

            // update alpha to the max similarity value if not using initial clusters
            if (!stage1){
                newAlpha = maxSimilarity(clusterSimilarities);

                if (newAlpha < alphaMin){
                    break;
                }
            }

            int tempClusterCount = clusterCount;
            int[] tempNewLabels = Arrays.copyOf(newLabels, newLabels.length);

            // merge clusters with a similarity greater than alpha
            boolean[] newCluster = new boolean[clusterCount];
            boolean[] merged = new boolean[clusterCount];
            countI = 0;
            for (int i = 0; i < binaryClusterMembership.size(); i++) {
                int countN = 0;
                for (int n = 0; n < i; n++) {
                    for (int j = 0; j < binaryClusterMembership.get(i).size(); j++) {
                        for (int k = 0; k < binaryClusterMembership.get(n).size(); k++) {
                            if (!merged[countI + j] && !merged[countN + k] &&
                                    clusterSimilarities[countI + j][countN + k] >= newAlpha) {
                                for (int g = 0; g < tempNewLabels.length; g++){
                                    if (tempNewLabels[g] == countI + j){
                                        tempNewLabels[g] = countN + k;
                                    }
                                }
                                tempNewLabels[countI + j] = tempNewLabels[countN + k];
                                merged[countI + j] = true;
                                newCluster[countN + k] = true;
                                tempClusterCount--;

                                for (int v = 0; v < numInstances; v++) {
                                    binaryClusterMembership.get(n).get(k)[v] +=
                                            binaryClusterMembership.get(i).get(j)[v];
                                }
                            }
                        }
                    }
                    countN += binaryClusterMembership.get(n).size();
                }
                countI += binaryClusterMembership.get(i).size();
            }

            // using the initial clusters, keep going and incrementing alpha until the number of
            // clusters is greater than k
            if (stage1) {
                if (tempClusterCount >= k) {
                    clusterCount = tempClusterCount;
                    binaryClusterMembership = removeMerged(binaryClusterMembership, merged, newCluster);
                    newLabels = relabel(tempNewLabels);
                    stage1 = false;
                } else {
                    newAlpha += alphaIncrement;
                }
            }
            // no longer using the initial clusters, keep going and lowering alpha to the max similarity
            // until the number of clusters is less than or equal to k or less than the minimum alpha
            else{
                if (tempClusterCount == k){
                    clusterCount = tempClusterCount;
                    binaryClusterMembership = removeMerged(binaryClusterMembership, merged, newCluster);
                    newLabels = relabel(tempNewLabels);
                    break;
                }
                else if (tempClusterCount < k){
                    break;
                }
                else{
                    clusterCount = tempClusterCount;
                    binaryClusterMembership = removeMerged(binaryClusterMembership, merged, newCluster);
                    newLabels = relabel(tempNewLabels);
                }
            }
        }

        // calculate how certain each cluster is for each case
        double[][] membershipCounts = new double[clusterCount][];
        int clusterIdx = 0;
        for (ArrayList<double[]> clusterGroup : binaryClusterMembership) {
            for (double[] cluster : clusterGroup) {
                membershipCounts[clusterIdx++] = cluster;
            }
        }

        double[][] membershipSimilarities = new double[clusterCount][numInstances];
        for (int i = 0; i < numInstances; i++){
            double sum = 0;
            for (int n = 0; n < clusterCount; n++){
                sum += membershipCounts[n][i];
            }

            for (int n = 0; n < clusterCount; n++){
                membershipSimilarities[n][i] = membershipCounts[n][i] / sum;
            }
        }

        int[] certainClusters = new int[clusterCount];
        for (int i = 0; i < clusterCount; i++){
            for (int n = 0; n < numInstances; n++){
                if (membershipSimilarities[i][n] > alpha2){
                    certainClusters[i] = 1;
                    break;
                }
            }
        }

        // if we dont have k clusters with at least one certain (member similarity > alpha2) case, find the k most
        // certain clusters
        Integer[] clusterRanks = new Integer[clusterCount];
        double ncc = sum(certainClusters);
        double newAlpha2 = alpha2;
        if (ncc != k){
            double[] clusterCertainties = new double[clusterCount];
            for (int i = 0; i < clusterCount; i++) {
                int numObjects = 0;
                for (int n = 0; n < numInstances; n++) {
                    if (membershipSimilarities[i][n] > 0) {
                        clusterCertainties[i] += membershipSimilarities[i][n];
                        numObjects++;
                    }
                    clusterCertainties[i] /= numObjects;
                }
            }

            for (int i = 0; i < clusterCount; i++) {
                clusterRanks[i] = i;
            }

            GenericTools.SortIndexDescending sort = new GenericTools.SortIndexDescending(clusterCertainties);
            Arrays.sort(clusterRanks, sort);

            //todo check out, weird
            newAlpha2 = -1;
            if (ncc < 1) {
                newAlpha2 = clusterCertainties[clusterRanks[k - 1]];
            }
            else {
                for (int i = 1; i < clusterCount; i++) {
                    double m = max(clusterSimilarities[i]);
                    if (m > newAlpha2) {
                        newAlpha2 = m;
                    }
                }
            }
        }
        //todo checkout
        else{
            int n = 0;
            for (int i = 0; i < clusterCount; i++) {
                if (certainClusters[i] == 1) {
                    clusterRanks[n++] = i;
                }
            }

            for (int i = 0; i < clusterCount; i++) {
                if (certainClusters[i] == 0) {
                    clusterRanks[n++] = i;
                }
            }
        }

        for (int i = 0; i < newLabels.length; i++){
            for (int n = 0; n < clusterCount; n++){
                if (newLabels[i] == clusterRanks[n]){
                    newLabels[i] = n;
                    break;
                }
            }
        }

        // calculate similarities of remaining clusters to removed ones to determine labels for new cases
        clusterSimilarities = new double[k][clusterCount - k];
        for (int i = k; i < clusterCount; i++){
            double max = -2;
            int maxIdx = -1;

            for (int n = 0; n < k; n++){
                double similarity = setCorrelation(membershipSimilarities[clusterRanks[i]],
                        membershipSimilarities[clusterRanks[n]], numInstances);
                clusterSimilarities[n][i - k] = similarity;

                if (similarity > max){
                    max = similarity;
                    maxIdx = n;
                }
            }

            for (int n = 0; n < newLabels.length; n++){
                if (newLabels[n] == i){
                    newLabels[n] = maxIdx;
                }
            }

            for (int n = 0; n < numInstances; n++){
                membershipCounts[maxIdx][n] += membershipCounts[clusterRanks[i]][n];
            }
        }

        assignments = new double[numInstances];
        clusters = new ArrayList[k];
        for (int i = 0; i < k; i++) {
            clusters[i] = new ArrayList();
        }

        //todo checkout
        if (clusterCount > k) {
            for (int i = 0; i < numInstances; i++) {
                double sum = 0;
                for (int n = 0; n < clusterCount; n++) {
                    sum += membershipCounts[clusterRanks[n]][i];
                }

                for (int n = 0; n < clusterCount; n++) {
                    membershipSimilarities[clusterRanks[n]][i] = membershipCounts[clusterRanks[n]][i] / sum;
                }
            }
        }

        // assign similarities to any cases which have no similarity with any of the current clusters
        int numUnclustered = 0;
        for (int i = 0; i < numInstances; i++) {
            double max = 0;
            int maxIdx = -1;
            double sum = 0;

            for (int n = 0; n < k; n++) {
                sum += membershipSimilarities[clusterRanks[n]][i];

                if (membershipSimilarities[clusterRanks[n]][i] > max){
                    max = membershipSimilarities[clusterRanks[n]][i];
                    maxIdx = n;
                }
            }

            // todo checkout, weird
            if (sum == 0){
                for (int n = 0; n < k; n++) {
                    for (int j = k; j < clusterCount; j++) {
                        if (membershipSimilarities[clusterRanks[j]][i] > 0) {
                            membershipSimilarities[clusterRanks[n]][i] += clusterSimilarities[n][j - k] *
                                    membershipSimilarities[clusterRanks[j]][i];
                        }
                    }

                    if (membershipSimilarities[clusterRanks[n]][i] > max){
                        max =  membershipSimilarities[clusterRanks[n]][i];
                        maxIdx = n;
                    }
                }
            }

            // this skips stage 3.
            if (max > newAlpha2) {
                assignments[i] = maxIdx;
                clusters[maxIdx].add(i);
            }
            else {
                assignments[i] = -1;
                numUnclustered++;
            }
        }

        // assign clusters to uncertain cases
        if (numUnclustered > 0) {
            double[] clusterQualities = new double[k];
            double[] membershipSums = new double[k];
            for (int n = 0; n < k; n++){
                int numAboveZero = 0;
                for (int i = 0; i < numInstances; i++) {
                    if (membershipSimilarities[clusterRanks[n]][i] > 0) {
                        membershipSums[n] += membershipSimilarities[clusterRanks[n]][i];
                        numAboveZero++;
                    }
                }
                membershipSums[n] /= numAboveZero;

                double sum = 0;
                for (int c: clusters[n]){
                    sum += Math.pow(membershipSimilarities[clusterRanks[n]][c] - membershipSums[n], 2);
                }
                clusterQualities[n] = sum / clusters[n].size();
            }

            for (int i = 0; i < numInstances; i++){
                if (assignments[i] == -1){
                    double minQualityChange = Double.MAX_VALUE;
                    double newClusterQuality = -1;
                    int minIdx = -1;

                    for (int n = 0; n < k; n++){
                        double sum = 0;
                        for (int c: clusters[n]){
                            sum += Math.pow(membershipSimilarities[clusterRanks[n]][c] - membershipSums[n], 2);
                        }
                        sum += Math.pow(membershipSimilarities[clusterRanks[n]][i] - membershipSums[n], 2);
                        double quality = sum / (clusters[n].size() + 1);

                        double qualityChange = quality - clusterQualities[n];
                        if (qualityChange < minQualityChange){
                            minQualityChange = qualityChange;
                            newClusterQuality = quality;
                            minIdx = n;
                        }
                    }

                    clusterQualities[minIdx] = newClusterQuality;
                    assignments[i] = minIdx;
                    clusters[minIdx].add(i);
                }
            }
        }
    }

    private double setCorrelation(double[] c1, double[] c2, int n){
        double c1Size = 0;
        double c2Size = 0;
        double intersection = 0;

        for (int i = 0; i < n; i ++){
            if (c1[i] > 0) {
                c1Size++;

                if (c2[i] > 0) {
                    c2Size++;
                    intersection++;
                }
            }
            else if (c2[i] > 0) {
                c2Size++;
            }
        }

        double multSize = c1Size * c2Size;

        double numerator = intersection - multSize / n;
        double denominator = Math.sqrt(multSize * (1 - c1Size / n) * (1 - c2Size / n));

        return numerator/denominator;
    }

    private double maxSimilarity(double[][] clusterSimilarities){
        double max = -1;
        for (double[] clusterSimilarity : clusterSimilarities) {
            for (double v : clusterSimilarity) {
                if (v > max) {
                    max = v;
                }
            }
        }
        return max;
    }

    private ArrayList<ArrayList<double[]>> removeMerged(ArrayList<ArrayList<double[]>> binaryClusterMembership,
                                                        boolean[] merged, boolean[] newCluster){
        ArrayList<ArrayList<double[]>> newBinaryClusterMembership = new ArrayList<>();
        int i = 0;

        for (ArrayList<double[]> clusterGroup : binaryClusterMembership) {
            ArrayList<double[]> newGroup = new ArrayList<>();

            for (int j = 0; j < clusterGroup.size(); j++) {
                if (newCluster[i]) {
                    if (clusterGroup.size() > 1) {
                        ArrayList<double[]> newSingleGroup = new ArrayList<>(1);
                        newSingleGroup.add(clusterGroup.get(j));
                        newBinaryClusterMembership.add(newSingleGroup);
                    } else {
                        newBinaryClusterMembership.add(clusterGroup);
                    }
                } else if (!merged[i]) {
                    newGroup.add(clusterGroup.get(j));
                }

                i++;
            }

            if (newGroup.size() > 0) {
                newBinaryClusterMembership.add(newGroup);
            }
        }

        return newBinaryClusterMembership;
    }

    private int[] relabel(int[] labels){
        Integer[] unique = unique(labels).toArray(new Integer[0]);
        for (int i = 0; i < labels.length; i++){
            for (int n = 0; n < unique.length; n++){
                if (labels[i] == unique[n]){
                    labels[i] = n;
                    break;
                }
            }
        }
        return labels;
    }

    public static void main(String[] args) throws Exception {
        String dataset = "Trace";
        Instances inst = DatasetLoading.loadDataNullable("D:\\CMP Machine Learning\\Datasets\\UnivariateARFF\\" + dataset + "/" +
                dataset + "_TRAIN.arff");
        Instances inst2 = DatasetLoading.loadDataNullable("D:\\CMP Machine Learning\\Datasets\\UnivariateARFF\\" + dataset + "/" +
                dataset + "_TEST.arff");
        inst.setClassIndex(inst.numAttributes() - 1);
        inst.addAll(inst2);

        ArrayList<EnhancedAbstractClusterer> clusterers = new ArrayList<>();
        for (int i = 0; i < 3; i++){
            KMeans c = new KMeans();
            c.setNumClusters(inst.numClasses());
            c.setSeed(i);
            clusterers.add(c);
        }

        ACE k = new ACE(clusterers);
        k.setNumClusters(inst.numClasses());
        k.setSeed(0);
        k.buildClusterer(inst);

        System.out.println(k.clusters.length);
        System.out.println(Arrays.toString(k.assignments));
        System.out.println(Arrays.toString(k.clusters));
        System.out.println(randIndex(k.assignments, inst));
    }
}
