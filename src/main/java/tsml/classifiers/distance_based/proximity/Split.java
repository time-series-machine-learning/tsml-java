package tsml.classifiers.distance_based.proximity;

import tsml.classifiers.distance_based.pf.Scorer;
import weka.core.Instances;

import java.util.List;

public abstract class Split {
    private Instances data;
    private List<Instances> split;
    private double score; // todo should this be in a sub class instead? As split isn't *necessarily* concerned
    // with the score
    private Scorer scorer = Scorer.giniScore; // todo same here, perhaps in sub class

    public Instances getData() {
        return data;
    }

    private void setSplit(List<Instances> split) {
        this.split = split;
    }

    public List<Instances> getSplit() {
        if(split == null) {
            setSplit(split(data));
            final double score = scorer.findScore(data, split);
            setScore(score);
        }
        return split;
    }

    public double getScore() {
        return score;
    }

    protected void setScore(final double score) {
        this.score = score;
    }

    protected abstract List<Instances> split(Instances data);

    public Scorer getScorer() {
        return scorer;
    }

    public void setScorer(final Scorer scorer) {
        this.scorer = scorer;
    }
}
