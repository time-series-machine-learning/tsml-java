package tsml.classifiers.distance_based.proximity.splitting.partition;

import java.util.List;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */

public interface PartitionSet {

    double getScore();

    Instances getData();

    List<Instances> getPartitions();

    PartitionSet setData(Instances data);

    PartitionSet setPartitions(List<Instances> partitions);

    PartitionSet setScore(double score);

    @Override
    String toString();
}
