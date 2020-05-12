package tsml.classifiers.distance_based.proximity.splitting;

import java.util.List;
import tsml.classifiers.distance_based.proximity.splitting.partition.BasePartitionSet;
import tsml.classifiers.distance_based.proximity.splitting.partition.PartitionSet;
import tsml.classifiers.distance_based.proximity.splitting.scoring.ScoreUtils;
import tsml.classifiers.distance_based.proximity.splitting.scoring.Scorer;
import tsml.classifiers.distance_based.utils.classifier_mixins.BaseClassifier;
import weka.core.Instance;
import weka.core.Instances;

public abstract class Split extends BaseClassifier {
    private final PartitionSet partitionSet = new BasePartitionSet();
    private Scorer scorer = ScoreUtils.getGlobalGiniImpurityScorer();
    private double score = -1;
    private Instances data;
    private List<Instances> partitions;

    /**
     * perform the split. This method should return the partitions of the data.
     */
    protected abstract List<Instances> performSplit(Instances data);

    @Override
    public final void buildClassifier(Instances data) {
        setData(data);
        final List<Instances> partitions = performSplit(data);
        setPartitions(partitions);
        final double score = getScorer().findScore(data, partitions);
        setScore(score);
    }

    public final Instances getPartitionFor(Instance instance) {
        final int index = getPartitionIndexFor(instance);
        return getPartitions().get(index);
    }

    public abstract int getPartitionIndexFor(Instance instance);

    public List<Instances> getPartitions() {
        return partitions;
    }

    public void setData(Instances data) {
        this.data = data;
    }

    public void setPartitions(List<Instances> partitions) {
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
        this.scorer = scorer;
        return this;
    }
}
