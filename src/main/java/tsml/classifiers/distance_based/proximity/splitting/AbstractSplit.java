package tsml.classifiers.distance_based.proximity.splitting;

import java.util.List;
import tsml.classifiers.distance_based.proximity.splitting.partition.BasePartitionSet;
import tsml.classifiers.distance_based.proximity.splitting.partition.PartitionSet;
import tsml.classifiers.distance_based.proximity.splitting.scoring.ScoreUtils;
import tsml.classifiers.distance_based.proximity.splitting.scoring.Scorer;
import weka.core.Instance;
import weka.core.Instances;

public abstract class AbstractSplit implements Split {
    private final PartitionSet partitionSet = new BasePartitionSet();
    private Scorer scorer = ScoreUtils.getGlobalGiniImpurityScorer();

    /**
     * perform the split. This method should return the partitions of the data.
     */
    protected abstract List<Instances> performSplit(Instances data);

    @Override
    public final List<Instances> split() {
        final Instances data = getData();
        if(data == null) {
            throw new IllegalStateException("data must be set first");
        }
        final List<Instances> partitions = performSplit(data);
        setPartitions(partitions);
        final double score = getScorer().findScore(data, partitions);
        setScore(score);
        return getPartitions();
    }

    @Override
    public final Instances getPartitionFor(Instance instance) {
        final int index = getPartitionIndexFor(instance);
        return getPartitions().get(index);
    }

    @Override
    public final double getScore() {
        return partitionSet.getScore();
    }

    @Override
    public final Instances getData() {
        return partitionSet.getData();
    }

    @Override
    public final List<Instances> getPartitions() {
        return partitionSet.getPartitions();
    }

    @Override
    public final PartitionSet setData(final Instances data) {
        return partitionSet.setData(data);
    }

    @Override
    public final PartitionSet setPartitions(final List<Instances> partitions) {
        return partitionSet.setPartitions(partitions);
    }

    @Override
    public final PartitionSet setScore(final double score) {
        return partitionSet.setScore(score);
    }

    @Override
    public String toString() {
        return partitionSet.toString();
    }

    @Override
    public final Scorer getScorer() {
        return scorer;
    }

    @Override
    public final Split setScorer(final Scorer scorer) {
        this.scorer = scorer;
        return this;
    }
}
