package tsml.classifiers.distance_based.proximity.splitting;

import java.util.List;
import tsml.classifiers.distance_based.proximity.splitting.partition.PartitionSet;
import tsml.classifiers.distance_based.proximity.splitting.scoring.Scorer;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */

public interface Split extends PartitionSet {

    List<Instances> split();

    Instances getPartitionFor(Instance instance);

    int getPartitionIndexFor(Instance instance);

    Scorer getScorer();

    Split setScorer(Scorer scorer);

}
