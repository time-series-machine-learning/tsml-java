package machine_learning.clusterers.consensus;

import blogspot.software_and_algorithms.stern_library.optimization.HungarianAlgorithm;
import evaluation.storage.ClustererResults;
import experiments.data.DatasetLoading;
import machine_learning.clusterers.KMeans;
import tsml.clusterers.EnhancedAbstractClusterer;
import weka.clusterers.NumberOfClustersRequestable;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import static utilities.ClusteringUtilities.randIndex;
import static utilities.InstanceTools.deleteClassAttribute;
import static utilities.Utilities.argMax;

public class SimpleVote extends ConsensusClusterer implements LoadableConsensusClusterer, NumberOfClustersRequestable {

    private int k = 2;

    private int[][] newLabels;
    private Random rand;

    public SimpleVote(EnhancedAbstractClusterer[] clusterers) {
        super(clusterers);
    }

    public SimpleVote(ArrayList<EnhancedAbstractClusterer> clusterers) {
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
                if (clusterer instanceof NumberOfClustersRequestable)
                    ((NumberOfClustersRequestable) clusterer).setNumClusters(k);
                clusterer.buildClusterer(train);
            }
        }

        for (EnhancedAbstractClusterer clusterer: clusterers){
            if (clusterer.numberOfClusters() != k)
                throw new Exception("SimpleVote base clusterer number of clusters must match k.");
        }

        double[][] ensembleAssignments = new double[clusterers.length][];
        for (int i = 0; i < ensembleAssignments.length; i++) {
            ensembleAssignments[i] = clusterers[i].getAssignments();
        }

        buildEnsemble(ensembleAssignments);
    }

    @Override
    public void buildFromFile(String[] directoryPaths) throws Exception {
        double[][] ensembleAssignments = new double[directoryPaths.length][];

        for (int i = 0; i < directoryPaths.length; i++) {
            ClustererResults r = new ClustererResults(directoryPaths[i] + "trainFold" + seed + ".csv");

            if (r.getNumClusters() != k)
                throw new Exception("SimpleVote base clusterer number of clusters must match k.");

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

        double[] votes = new double[k];
        votes[clusterers[0].clusterInstance(inst)]++;
        for (int i = 1; i < clusterers.length; i++) {
            votes[newLabels[i - 1][clusterers[i].clusterInstance(inst)]]++;
        }

        return argMax(votes, rand);
    }

    @Override
    public int[] clusterFromFile(String[] directoryPaths) throws Exception {
        int[][] ensembleAssignments = new int[directoryPaths.length][];

        for (int i = 0; i < directoryPaths.length; i++) {
            ClustererResults r = new ClustererResults(directoryPaths[i] + "testFold" + seed + ".csv");

            if (r.getNumClusters() != k)
                throw new Exception("SimpleVote base clusterer number of clusters must match k.");

            ensembleAssignments[i] = r.getClusterValuesAsIntArray();

            if (i > 0){
                for (int n = 0; n < ensembleAssignments[i].length; n++) {
                    ensembleAssignments[i][n] = newLabels[i - 1][ensembleAssignments[i][n]];
                }
            }
        }

        int[] arr = new int[ensembleAssignments[0].length];
        for (int i = 0; i < arr.length; i++) {
            double[] votes = new double[k];
            for (int[] clusterAssignments : ensembleAssignments) {
                votes[clusterAssignments[i]]++;
            }

            arr[i] = argMax(votes, rand);
        }

        return arr;
    }

    private void buildEnsemble(double[][] ensembleAssignments){
        if (!seedClusterer) {
            rand = new Random();
        } else {
            rand = new Random(seed);
        }

        newLabels = new int[ensembleAssignments.length -1][];

        for (int i = 1; i < ensembleAssignments.length; i++) {
            double[][] contingencyTable = new double[k][k];

            for (int n = 0; n < ensembleAssignments[i].length; n++) {
                contingencyTable[(int)ensembleAssignments[0][n]][(int)ensembleAssignments[i][n]]--;
            }

            newLabels[i - 1] = new HungarianAlgorithm(contingencyTable).execute();

            for (int n = 0; n < ensembleAssignments[i].length; n++) {
                ensembleAssignments[i][n] = newLabels[i - 1][(int)ensembleAssignments[i][n]];
            }
        }

        assignments = new double[train.numInstances()];
        for (int i = 0; i < assignments.length; i++) {
            double[] votes = new double[k];
            for (double[] clusterAssignments : ensembleAssignments) {
                votes[(int) clusterAssignments[i]]++;
            }

            assignments[i] = argMax(votes, rand);
        }

        //Create and store an ArrayList for each cluster containing indexes of
        //points inside the cluster.
        clusters = new ArrayList[k];

        for (int i = 0; i < k; i++) {
            clusters[i] = new ArrayList();
        }

        for (int i = 0; i < train.numInstances(); i++) {
            clusters[(int) assignments[i]].add(i);
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

        SimpleVote k = new SimpleVote(clusterers);
        k.setNumClusters(inst.numClasses());
        k.setSeed(0);
        k.buildClusterer(inst);

        System.out.println(k.clusters.length);
        System.out.println(Arrays.toString(k.assignments));
        System.out.println(Arrays.toString(k.clusters));
        System.out.println(randIndex(k.assignments, inst));
    }
}
