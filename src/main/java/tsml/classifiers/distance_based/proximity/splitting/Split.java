package tsml.classifiers.distance_based.proximity.splitting;

import java.util.List;
import java.util.Random;
import org.junit.Assert;
import tsml.classifiers.distance_based.proximity.splitting.scoring.ScoreUtils;
import tsml.classifiers.distance_based.proximity.splitting.scoring.Scorer;
import tsml.classifiers.distance_based.utils.classifier_mixins.BaseClassifier;
import weka.core.Instance;
import weka.core.Instances;

public abstract class Split {
    private Random random;
    private Scorer scorer = ScoreUtils.getGlobalGiniImpurityScorer();
    private double score = -1;
    private Instances data;
    private List<Instances> partitions;

    public Split(final Random random) {
        setRandom(random);
    }

    /**
     * perform the split. This method should return the partitions of the data.
     */
    protected abstract List<Instances> performSplit(Instances data);

    public final void buildSplit() {
        final List<Instances> partitions = performSplit(data);
        setPartitions(partitions);
        final double score = scorer.findScore(data, partitions);
        setScore(score);
    }

    public final void buildSplit(Instances data) {
        setData(data);
        buildSplit();
    }

    public final Instances getPartitionFor(Instance instance) {
        final int index = getPartitionIndexFor(instance);
        return partitions.get(index);
    }

    public abstract int getPartitionIndexFor(Instance instance);

    public double[] distributionForInstance(Instance instance) {
        int index = getPartitionIndexFor(instance);
        return distributionForInstance(instance, index);
    }

    public abstract double[] distributionForInstance(Instance instance, int partitionIndex);

    public List<Instances> getPartitions() {
        return partitions;
    }

    public Instances getData() {
        return data;
    }

    public void setData(Instances data) {
        Assert.assertNotNull(data);
        this.data = data;
    }

    private void setPartitions(List<Instances> partitions) {
        Assert.assertNotNull(partitions);
        this.partitions = partitions;
    }

    public void setScore(final double score) {
        this.score = score;
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

    public final Scorer getScorer() {
        return scorer;
    }

    public final Split setScorer(final Scorer scorer) {
        Assert.assertNotNull(scorer);
        this.scorer = scorer;
        return this;
    }

    public Random getRandom() {
        return random;
    }

    public Split setRandom(final Random random) {
        Assert.assertNotNull(random);
        this.random = random;
        return this;
    }

    public double getScore() {
        return score;
    }
}
