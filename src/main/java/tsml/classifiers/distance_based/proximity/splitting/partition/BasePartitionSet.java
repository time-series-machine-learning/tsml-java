package tsml.classifiers.distance_based.proximity.splitting.partition;

import java.util.List;
import weka.core.Instances;

/**
 * Purpose: data class for holding the results of a partition operation.
 * <p>
 * Contributors: goastler
 */
public class BasePartitionSet implements PartitionSet {
    private double score = -1;
    private Instances data;
    private List<Instances> partitions;

    public BasePartitionSet(double score, Instances data, List<Instances> partitions) {
        setScore(score);
        setData(data);
        setPartitions(partitions);
    }

    public BasePartitionSet() {}

    @Override
    public double getScore() {
        return score;
    }

    @Override
    public Instances getData() {
        return data;
    }

    @Override
    public List<Instances> getPartitions() {
        return partitions;
    }

    @Override
    public PartitionSet setData(Instances data) {
        this.data = data;
        return this;
    }

    @Override
    public PartitionSet setPartitions(List<Instances> partitions) {
        this.partitions = partitions;
        return this;
    }

    @Override
    public PartitionSet setScore(final double score) {
        this.score = score;
        return this;
    }

    @Override
    public String toString() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("PartitionSet{" +
            "score=" + score +
            ", dataSize=" + data.size());
        if(partitions != null) {
            int i = 0;
            for(Instances instances : partitions) {
                stringBuilder.append(", p" + i + "=" + instances.size());
                i++;
            }
        }
        stringBuilder.append("}");
        return  stringBuilder.toString();
    }
}
