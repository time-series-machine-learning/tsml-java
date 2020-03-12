package tsml.classifiers.distance_based.proximity.tmp;

import java.util.List;
import org.junit.Assert;
import weka.core.Instances;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public class Split {
    private double score = -1;
    private Instances data;
    private List<Instances> partitions;

    public Split(double score, Instances data, List<Instances> partitions) {
        Assert.assertNotNull(data);
        Assert.assertNotNull(partitions);
        Assert.assertFalse(partitions.isEmpty());
        setScore(score);
        setData(data);
        setPartitions(partitions);
    }

    public double getScore() {
        return score;
    }

    public Instances getData() {
        return data;
    }

    public List<Instances> getPartitions() {
        return partitions;
    }

    protected Split setScore(double score) {
        this.score = score;
        return this;
    }

    protected Split setData(Instances data) {
        this.data = data;
        return this;
    }

    protected Split setPartitions(List<Instances> partitions) {
        this.partitions = partitions;
        return this;
    }
}
