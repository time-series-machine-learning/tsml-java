package tsml.classifiers.distance_based.utils.stats.scoring.v2;

import utilities.Utilities;

import java.util.List;

public class InfoEntropy implements SplitScorer {
    @Override public <A> double score(final Labels<A> parentLabels, final List<Labels<A>> childLabels) {
        return ScoreUtils.weightedEntropy(parentLabels, childLabels, InfoGain.INSTANCE::inverseEntropy);
    }
}
