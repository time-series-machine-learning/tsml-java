package tsml.classifiers.distance_based.utils.stats.scoring;

import weka.core.Instances;

import java.util.List;

public class ChiSquaredGain implements SplitScorer {
    @Override public double findScore(final Instances parent, final List<Instances> children) {
        return ScoreUtils.chiSquared(parent, children);
    }
}
