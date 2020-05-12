package tsml.classifiers.distance_based.proximity.splitting.partition;

import java.util.List;
import weka.core.Instances;

/**
 * Purpose: data class for holding the results of a partition operation.
 * <p>
 * Contributors: goastler
 */
public class BasePartitionSet implements PartitionSet {

    public BasePartitionSet(double score, Instances data, List<Instances> partitions) {
        setScore(score);
        setData(data);
        setPartitions(partitions);
    }

    public BasePartitionSet() {}

}
