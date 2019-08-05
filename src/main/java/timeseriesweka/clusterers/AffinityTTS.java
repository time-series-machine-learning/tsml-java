package timeseriesweka.clusterers;

import experiments.data.DatasetLoading;
import timeseriesweka.elastic_distance_measures.DTW;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka_extras.clusterers.CAST;
import weka_extras.clusterers.PAM;

import java.util.ArrayList;
import java.util.Arrays;

import static utilities.ClusteringUtilities.*;
import static utilities.InstanceTools.deleteClassAttribute;

//runs but does not perform as well as intended, needs revisiting.

public class AffinityTTS extends AbstractTimeSeriesClusterer {

    //Aghabozorgi, Saeed, et al.
    //"A hybrid algorithm for clustering of time series data based on affinity search technique."
    //The Scientific World Journal 2014 (2014).

    private double affinityThreshold = 0.01;
    private int k = 2;
    private int seed = Integer.MIN_VALUE;

    private double[][] distanceMatrix;
    private ArrayList<Integer>[] subclusters;

    public AffinityTTS(){}

    @Override
    public int numberOfClusters() throws Exception { return k; }

    public void setSeed(int seed){ this.seed = seed; }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        if (copyInstances){
            data = new Instances(data);
        }

        deleteClassAttribute(data);
        zNormalise(data);

        EuclideanDistance ed = new EuclideanDistance();
        ed.setDontNormalize(true);
        distanceMatrix = createDistanceMatrix(data, ed);

        CAST cast = new CAST(distanceMatrix);
        cast.setAffinityThreshold(affinityThreshold);
        cast.buildClusterer(data);
        subclusters = cast.getClusters();
        ArrayList<double[]> affinities = cast.getClusterAffinities();

        double[][] prototypes = new double[subclusters.length][data.numAttributes()];

        for (int i = 0; i < subclusters.length; i++){
            for (int n = 0; n < data.numAttributes(); n++){
                for (int g = 0; g < subclusters[i].size(); g++){
                    prototypes[i][n] += data.get(subclusters[i].get(g)).value(n) * (1-affinities.get(i)[g]);
                }

                prototypes[i][n] /= subclusters[i].size();
            }
        }

        Instances cl = new Instances(data, subclusters.length);
        for (int i = 0; i < subclusters.length; i++){
            cl.add(new DenseInstance(1, prototypes[i]));
        }

        PAM pam = new PAM();
        pam.setDistanceFunction(new DTW());
        pam.setK(k);
        pam.setSeed(seed);
        pam.buildClusterer(cl);

        System.out.println(Arrays.toString(pam.getClusters()));

        ArrayList<Integer>[] ptClusters = pam.getClusters();
        cluster = new int[data.size()];

        for (int i = 1; i < k; i++){
            for (int n = 0; n < ptClusters[i].size(); n++){
                ArrayList<Integer> subcluster = subclusters[ptClusters[i].get(n)];

                for (int g = 0; g < subcluster.size(); g++){
                    cluster[subcluster.get(g)] = i;
                }
            }
        }

        clusters = new ArrayList[k];

        for (int i = 0; i < k; i++){
            clusters[i] = new ArrayList();
        }

        for (int i = 0; i < data.size(); i++){
            for (int n = 0; n < k; n++){
                if(n == cluster[i]){
                    clusters[n].add(i);
                    break;
                }
            }
        }
    }

    public static void main(String[] args) throws Exception{
        String dataset = "Trace";
        Instances inst = DatasetLoading.loadDataNullable("D:\\CMP Machine Learning\\Datasets\\TSC Archive\\" + dataset + "/" + dataset + "_TRAIN.arff");
        Instances inst2 = DatasetLoading.loadDataNullable("D:\\CMP Machine Learning\\Datasets\\TSC Archive\\" + dataset + "/" + dataset + "_TEST.arff");
//        Instances inst = ClassifierTools.loadData("Z:\\Data\\TSCProblems2018\\" + dataset + "/" + dataset + "_TRAIN.arff");
//        Instances inst2 = ClassifierTools.loadData("Z:\\Data\\TSCProblems2018\\" + dataset + "/" + dataset + "_TEST.arff");
        inst.setClassIndex(inst.numAttributes()-1);
        inst.addAll(inst2);

        AffinityTTS k = new AffinityTTS();
        k.seed = 0;
        k.k = inst.numClasses();
        k.buildClusterer(inst);

        System.out.println(k.clusters.length);
        System.out.println(Arrays.toString(k.clusters));
        System.out.println(randIndex(k.cluster, inst));
    }
}
