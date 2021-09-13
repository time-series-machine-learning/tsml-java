package machine_learning.clusterers.consensus;

import tsml.clusterers.EnhancedAbstractClusterer;
import weka.core.Instances;

import java.util.ArrayList;

public class ACE extends ConsensusClusterer {

    public ACE(EnhancedAbstractClusterer[] clusterers) {
        super(clusterers);
    }

    @Override
    public int numberOfClusters() throws Exception {
        return 0;
    }

    @Override
    public void buildClusterer(Instances data) throws Exception {
        super.buildClusterer(data);

        if (buildClusterers){
            for (EnhancedAbstractClusterer clusterer: clusterers){
                clusterer.buildClusterer(data);
            }
        }

        int clusterCount = 0;
        for (EnhancedAbstractClusterer clusterer: clusterers){
            clusterCount += clusterer.numberOfClusters();
        }

        double[][] binaryClusterMembership = new double[data.numInstances()][clusterCount];
        int i = 0;
        for (EnhancedAbstractClusterer clusterer: clusterers){
            for (ArrayList<Integer> cluster: clusterer.getClusters()){
                for (int n: cluster){
                    binaryClusterMembership[n][i] = 1;
                }
                i++;
            }
        }

        double[][] clusterSimilarities = new double[][]{};


    }
}
