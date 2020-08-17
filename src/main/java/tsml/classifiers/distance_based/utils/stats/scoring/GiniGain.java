package tsml.classifiers.distance_based.utils.stats.scoring;

import weka.core.Instances;

import java.util.List;

public class GiniGain implements PartitionScorer {

    @Override
    public double findScore(final Instances parent, final List<Instances> children) {
        return ScoreUtils.giniGain(parent, children);
    }
}
