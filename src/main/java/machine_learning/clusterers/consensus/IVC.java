package machine_learning.clusterers.consensus;

import evaluation.storage.ClustererResults;
import experiments.data.DatasetLoading;
import machine_learning.clusterers.KMeans;
import tsml.clusterers.EnhancedAbstractClusterer;
import weka.clusterers.NumberOfClustersRequestable;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

import static utilities.ClusteringUtilities.randIndex;
import static utilities.Utilities.argMax;

public class IVC extends ConsensusClusterer implements LoadableConsensusClusterer, NumberOfClustersRequestable {

    // https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4470298

    private int k = 2;
    private int maxIterations = 200;

    private double[][] clusterCenters;
    private Random rand;

    public IVC(EnhancedAbstractClusterer[] clusterers) {
        super(clusterers);
    }

    public IVC(ArrayList<EnhancedAbstractClusterer> clusterers) {
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

        double[][] ensembleClusters = new double[clusterers.length][];
        for (int i = 0; i < ensembleClusters.length; i++) {
            ensembleClusters[i] = clusterers[i].getAssignments();
        }

        buildEnsemble(ensembleClusters);
    }

    @Override
    public void buildFromFile(String[] directoryPaths) throws Exception {
        double[][] ensembleAssignments = new double[directoryPaths.length][];

        for (int i = 0; i < directoryPaths.length; i++) {
            ClustererResults r = new ClustererResults(directoryPaths[i] + "trainFold" + seed + ".csv");
            ensembleAssignments[i] = r.getClusterValuesAsArray();
        }

        buildEnsemble(ensembleAssignments);
    }

    @Override
    public int clusterInstance(Instance inst) throws Exception {
        Instance newInst = copyInstances ? new DenseInstance(inst) : inst;
        int clsIdx = inst.classIndex();
        if (clsIdx >= 0){
            newInst.setDataset(null);
            newInst.deleteAttributeAt(clsIdx);
        }

        double minDist = Double.MAX_VALUE;
        int minIndex = -1;

        for (int i = 0; i < k; i++) {
            double dist = 0;

            for (int n = 0; n < clusterCenters[i].length; n++) {
                if (clusterers[n].clusterInstance(inst) == clusterCenters[i][n]) {
                    dist++;
                }
            }

            if (dist < minDist) {
                minDist = dist;
                minIndex = i;
            }
        }

        return minIndex;
    }

    @Override
    public int[] clusterFromFile(String[] directoryPaths) throws Exception {
        int[][] ensembleAssignments = new int[directoryPaths.length][];

        for (int i = 0; i < directoryPaths.length; i++) {
            ClustererResults r = new ClustererResults(directoryPaths[i] + "testFold" + seed + ".csv");
            ensembleAssignments[i] = r.getClusterValuesAsIntArray();
        }

        int[] cluserings = new int[ensembleAssignments[0].length];

        for (int i = 0; i < ensembleAssignments[0].length; i++) {
            double minDist = Double.MAX_VALUE;
            int minIndex = -1;

            for (int n = 0; n < k; n++) {
                double dist = 0;

                for (int j = 0; j < clusterCenters[n].length; j++) {
                    if (ensembleAssignments[j][i] == clusterCenters[n][j]) {
                        dist++;
                    }
                }

                if (dist < minDist) {
                    minDist = dist;
                    minIndex = n;
                }
            }

            cluserings[i] = minIndex;
        }

        return cluserings;
    }

    @Override
    public double[][] distributionFromFile(String[] directoryPaths) throws Exception {
        int[] clusterings = clusterFromFile(directoryPaths);

        double[][] dists = new double[clusterings.length][k];
        for (int i = 0; i < clusterings.length; i++) {
            dists[i][clusterings[i]] = 1;
        }

        return dists;
    }

    private void buildEnsemble(double[][] ensembleClusters) throws Exception {
        if (!seedClusterer) {
            rand = new Random();
        } else {
            rand = new Random(seed);
        }

        assignments = new double[ensembleClusters[0].length];

        initialClusterCenters(ensembleClusters);
        calculateClusterMembership(ensembleClusters);

        boolean finished = false;
        int iterations = 0;

        //Change cluster centers until cluster membership no longer changes
        while (!finished) {
            if (!calculateClusterMembership(ensembleClusters) || iterations == maxIterations) {
                finished = true;
            } else {
                selectClusterCenters(ensembleClusters);
            }

            iterations++;
        }
    }

    private void initialClusterCenters(double[][] ensembleClusters) {
        ArrayList<Integer> indexes = new ArrayList<>(ensembleClusters[0].length);

        for (int i = 0; i < ensembleClusters[0].length; i++) {
            indexes.add(i);
        }

        clusterCenters = new double[k][ensembleClusters.length];
        Collections.shuffle(indexes, rand);

        for (int i = 0; i < k; i++) {
            for (int n = 0; n < ensembleClusters.length; n++) {
                clusterCenters[i][n] = ensembleClusters[n][indexes.get(i)];
            }
        }
    }

    private boolean calculateClusterMembership(double[][] ensembleClusters) {
        boolean membershipChange = false;

        //Set membership of each point to the closest cluster center
        for (int i = 0; i < ensembleClusters[0].length; i++) {
            double minDist = Double.MAX_VALUE;
            int minIndex = -1;

            for (int n = 0; n < k; n++) {
                double dist = 0;

                for (int j = 0; j < clusterCenters[n].length; j++) {
                    if (ensembleClusters[j][i] == clusterCenters[n][j]) {
                        dist++;
                    }
                }

                if (dist < minDist) {
                    minDist = dist;
                    minIndex = n;
                }
            }

            //If membership of any point changed return true to keep
            //looping
            if (minIndex != assignments[i]) {
                assignments[i] = minIndex;
                membershipChange = true;
            }
        }

        if (membershipChange) {
            //Create and store an ArrayList for each cluster containing indexes of
            //points inside the cluster.
            clusters = new ArrayList[k];

            for (int i = 0; i < k; i++) {
                clusters[i] = new ArrayList<>();
            }

            for (int i = 0; i < train.numInstances(); i++) {
                clusters[(int) assignments[i]].add(i);
            }
        }

        return membershipChange;
    }

    private void selectClusterCenters(double[][] ensembleClusters) {
        for (int i = 0; i < k; i++) {
            int halfPoint = clusters[i].size() / 2;

            for (int n = 0; n < ensembleClusters.length; n++) {
                HashMap<Double, Integer> map = new HashMap<>();
                boolean foundMajority = false;

                for (int j = 0; j < clusters[i].size(); j++) {
                    double v = ensembleClusters[n][clusters[i].get(j)];

                    if (map.containsKey(v)) {
                        int count = map.get(v) + 1;
                        if (count > halfPoint) {
                            clusterCenters[i][n] = v;
                            foundMajority = true;
                            break;
                        } else {
                            map.put(v, count);
                        }
                    } else {
                        map.put(v, 1);
                    }
                }

                if (!foundMajority) {
                    int maxCount = -1;

                    for (Map.Entry<Double, Integer> entry : map.entrySet()) {
                        if (entry.getValue() > maxCount || (entry.getValue() == maxCount && rand.nextBoolean())) {
                            maxCount = entry.getValue();
                            clusterCenters[i][n] = entry.getKey();
                        }
                    }
                }
            }
        }
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

        IVC k = new IVC(clusterers);
        k.setNumClusters(inst.numClasses());
        k.setSeed(0);
        k.buildClusterer(inst);

        System.out.println(k.clusters.length);
        System.out.println(Arrays.toString(k.assignments));
        System.out.println(Arrays.toString(k.clusters));
        System.out.println(randIndex(k.assignments, inst));
    }
}
