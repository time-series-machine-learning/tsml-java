package tsml.classifiers.distance_based.utils.stats.scoring;

import java.util.List;

public class InfoGain implements SplitScorer {

    @Override public <A> double score(final Labels<A> parent, final List<Labels<A>> children) {
        return GiniGain.gain(parent, children, new InfoEntropy());
    }
}
