package tsml.classifiers.distance_based.utils.stats.scoring.v2;

import java.util.List;

public interface EntropyScorer extends SplitScorer {

    <A> double entropy(Labels<A> labels);

    @Override default <A> double score(Labels<A> parentLabels, List<Labels<A>> childLabels) {
        return ScoreUtils.weightedEntropy(parentLabels, childLabels, this::entropy);
    }

}
