package tsml.classifiers.distance_based.pf.partition;
/*

purpose: // todo - docs - type the purpose of the code here

created edited by goastler on 17/02/2020
    
*/

import tsml.classifiers.TrainTimeContractable;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.List;

public interface Partitioner extends Classifier, TrainTimeContractable {
    int getPartitionIndex(Instance instance);
    List<Instances> getPartitions();
    default Instances getPartition(Instance instance) {
        return getPartitions().get(getPartitionIndex(instance));
    }
    void clean();
}
