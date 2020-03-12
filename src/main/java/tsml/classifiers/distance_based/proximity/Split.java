package tsml.classifiers.distance_based.proximity;

import tsml.classifiers.distance_based.pf.Scorer;
import weka.core.Instances;

import java.util.List;

/**
 * Purpose: split some given data into parts. This class should hold a) the input data; b) the split data; c) all of
 * the info used to produce the split; d) the score of the split
 *
 * Contributors: goastler
 */
public abstract class Split {

    public static abstract class Builder {
        private Instances data;
        public abstract Split build();

        public Instances getData() {
            return data;
        }

        public Builder setData(Instances data) {
            this.data = data;
            return this;
        }
    }

    private Instances data;
    private List<Instances> splitData;
    private double score;
    private Scorer scorer = Scorer.giniScore;

    public List<Instances> split(final Instances data) {
        setData(data);
        final List<Instances> splitData = splitData(data);
        setSplitData(splitData);
        final double score = scorer.findScore(data, splitData);
        setScore(score);
        return splitData;
    }

    private void setData(Instances data) {
        this.data = data;
    }

    public List<Instances> getSplitData() {
        return splitData;
    }

    private void setSplitData(List<Instances> splitData) {
        this.splitData = splitData;
    }

    protected abstract List<Instances> splitData(Instances data);

    public Instances getData() {
        return data;
    }

    public double getScore() {
        return score;
    }

    protected void setScore(final double score) {
        this.score = score;
    }

    public Scorer getScorer() {
        return scorer;
    }

    public void setScorer(final Scorer scorer) {
        this.scorer = scorer;
    }
}
