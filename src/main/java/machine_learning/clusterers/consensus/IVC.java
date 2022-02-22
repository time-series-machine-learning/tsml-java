package machine_learning.clusterers.consensus;

import evaluation.storage.ClustererResults;
import experiments.data.DatasetLoading;
import machine_learning.clusterers.KMeans;
import tsml.clusterers.EnhancedAbstractClusterer;
import weka.clusterers.NumberOfClustersRequestable;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

public class IVC extends ConsensusClusterer implements LoadableConsensusClusterer, NumberOfClustersRequestable {

    // https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4470298

    private int k = 2;

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
        return 0;
    }

    @Override
    public double[] distributionForInstance(Instance inst) throws Exception {
        return null;
    }

    @Override
    public int[] clusterFromFile(String[] directoryPaths) throws Exception {
        return new int[0];
    }

    @Override
    public double[][] distributionFromFile(String[] directoryPaths) throws Exception {
        return new double[0][0];
    }

    private void buildEnsemble(double[][] ensembleClusters) throws Exception {
        assignments = new double[ensembleClusters[0].length];

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

        IVC k = new IVC(clusterers);
        k.setNumClusters(3);
        k.setSeed(0);
        k.buildClusterer(inst);

//        System.out.println(k.clusters.length);
//        System.out.println(Arrays.toString(k.assignments));
//        System.out.println(Arrays.toString(k.clusters));
//        System.out.println(randIndex(k.assignments, inst));
    }
}
