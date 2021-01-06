package tsml.classifiers.distance_based.utils.stats.scoring.v2;

import utilities.Utilities;

import java.util.List;

import static tsml.classifiers.distance_based.utils.stats.scoring.v2.ScoreUtils.gain;

public class InfoGain implements SplitScorer {

    public static final InfoGain INSTANCE = new InfoGain();
    
    @Override public <A> double score(final Labels<A> parentLabels, final List<Labels<A>> childLabels) {
        return gain(parentLabels, childLabels, this::entropy);
    }

    public <A> double entropy(final Labels<A> labels) {
        return labels.getDistribution().stream().mapToDouble(d -> -d * Utilities.log(d, 2)).sum();
    }
    
    public <A> double inverseEntropy(final Labels<A> labels) {
        final double entropy = entropy(labels);
//        final double worst = entropy(new Labels<>(labels.getLabelSet()));
        final int numClasses = labels.getLabelSet().size();
        final double uniformProbability = 1d / numClasses;
        final double worst = -uniformProbability * Utilities.log(uniformProbability, 2) * numClasses;
        return worst - entropy;
    }
}
