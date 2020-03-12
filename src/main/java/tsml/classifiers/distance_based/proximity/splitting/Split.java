package tsml.classifiers.distance_based.proximity.splitting;

import java.util.List;
import org.junit.Assert;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public abstract class Split {
    private double score = -1;
    private Scorer scorer = Scorer.giniScore;
    private Instances data;
    private List<Instances> partitions;

    public Split(double score, Instances data, List<Instances> partitions) {
        setScore(score);
        setData(data);
        setPartitions(partitions);
    }

    // on the assumption that subclasses know what they're doing
    protected Split() {}

    public double getScore() {
        return score;
    }

    public Instances getData() {
        return data;
    }

    protected abstract List<Instances> findPartitions();

    public List<Instances> getPartitions() {
        List<Instances> partitions = getPartitions();
        if(partitions == null) {
            partitions = findPartitions();
            setPartitions(partitions);
            final double score = scorer.findScore(getData(), partitions);
            setScore(score);
        }
        return partitions;
    }

    public abstract int getPartitionIndexOf(Instance instance);

    public Instances getPartitionFor(Instance instance) {
        final int index = getPartitionIndexOf(instance);
        final List<Instances> partitions = getPartitions();
        return partitions.get(index);
    }

    public Split setScore(double score) {
        this.score = score;
        return this;
    }

    public Split setData(Instances data) {
        Assert.assertNotNull(data);
        this.data = data;
        return this;
    }

    public Split setPartitions(List<Instances> partitions) {
        if(partitions != null) {
            Assert.assertFalse(partitions.isEmpty());
        }
        this.partitions = partitions;
        return this;
    }

    public Scorer getScorer() {
        return scorer;
    }

    public Split setScorer(final Scorer scorer) {
        this.scorer = scorer;
        return this;
    }
}
