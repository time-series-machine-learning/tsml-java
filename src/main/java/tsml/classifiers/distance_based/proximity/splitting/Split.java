package tsml.classifiers.distance_based.proximity.splitting;

import java.util.List;
import org.junit.Assert;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Purpose: split data into partitions and score the split.
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
        score = scorer.findScore(data, partitions);
        return score;
    }

    public Instances getData() {
        return data;
    }

    public List<Instances> getPartitions() {
        return partitions;
    }

    protected abstract List<Instances> split();

    public List<Instances> findPartitions() {
        List<Instances> partitions = getPartitions();
        if(partitions == null) {
            partitions = split();
            setPartitions(partitions);
            double score = getScorer().findScore(getData(), partitions);
            setScore(score);
        }
        return partitions;
    }

    public abstract int getPartitionIndexOf(Instance instance);

    public Instances getPartitionFor(Instance instance) {
        final int index = getPartitionIndexOf(instance);
        final List<Instances> partitions = findPartitions();
        return partitions.get(index);
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

    public Split setScore(final double score) {
        this.score = score;
        return this;
    }
}
