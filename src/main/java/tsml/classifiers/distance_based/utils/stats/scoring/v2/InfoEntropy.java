package tsml.classifiers.distance_based.utils.stats.scoring.v2;

import utilities.Utilities;

import java.util.List;

public class InfoEntropy implements SplitScorer {
    @Override public <A> double score(final Labels<A> parentLabels, final List<Labels<A>> childLabels) {
        return ScoreUtils.weightedEntropy(parentLabels, childLabels, InfoGain.INSTANCE::inverseEntropy);
    }

    public static void main(String[] args) {
        for(int i = 2; i < 10; i++) {
            double a = -(1d / i * Utilities.log(1d / i,2) * i);
            System.out.println(a);
        }
    }
}
