package tsml.classifiers.distance_based.utils.stats.scoring.a;

import tsml.classifiers.distance_based.utils.stats.scoring.v2.Labels;

import java.util.List;

import static tsml.classifiers.distance_based.utils.stats.scoring.a.GiniScore.gain;

public class InfoGain implements Score {

    @Override public <A> double score(final Labels<A> parent, final List<Labels<A>> children) {
        return gain(parent, children, new InfoEntropy());
    }
}
