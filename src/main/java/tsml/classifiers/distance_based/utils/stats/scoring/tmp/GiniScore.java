package tsml.classifiers.distance_based.utils.stats.scoring.tmp;

import java.util.List;

import static tsml.classifiers.distance_based.utils.stats.scoring.ScoreUtils.gain;

public class GiniScore implements SplitScorer {

    public static final GiniScore INSTANCE = new GiniScore();
    
    @Override public <A> double score(final Labels<A> parentLabels, final List<Labels<A>> childLabels) {
        return gain(parentLabels, childLabels, this::entropy);
    }
    
    public <A> double entropy(final Labels<A> labels) {
        return 1 - labels.getDistribution().stream().mapToDouble(d -> Math.pow(d, 2)).sum();
    }
    
    public <A> double inverseEntropy(final Labels<A> labels) {
        final double entropy = entropy(labels);
        final int numClasses = labels.getLabelSet().size();
        final double worst = 1 - Math.pow(1d / numClasses, 2) * numClasses;
        return worst - entropy;
        //        // below is the polarised version where larger values are better
        //        // generate the worst possible score given a uniform distribution of labels
        //        final double worst = 1 - Math.pow(1d / distribution.size(), 2) * distribution.size();
        //        // then scores further from the worst become larger and therefore larger values mean better partitioning
        //        return worst - score;
    }

}
