package tsml.classifiers.distance_based.utils.stats.scoring.tmp;

import tsml.classifiers.distance_based.utils.stats.scoring.ScoreUtils;

import java.util.List;

public interface EntropyScorer extends SplitScorer {

    <A> double entropy(Labels<A> labels);

    @Override default <A> double score(Labels<A> parentLabels, List<Labels<A>> childLabels) {
        return ScoreUtils.weightedEntropy(parentLabels, childLabels, this::entropy);
    }

}
