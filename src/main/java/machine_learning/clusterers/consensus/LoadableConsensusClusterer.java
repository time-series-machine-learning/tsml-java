package machine_learning.clusterers.consensus;

public interface LoadableConsensusClusterer {

    void buildFromFile(String[] directoryPaths) throws Exception;

    int[] clusterFromFile(String[] directoryPaths) throws Exception;
}
