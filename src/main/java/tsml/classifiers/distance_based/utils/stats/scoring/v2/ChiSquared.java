package tsml.classifiers.distance_based.utils.stats.scoring.v2;

import java.util.List;

public class ChiSquared implements SplitScorer {
    @Override public <A> double score(final Labels<A> parentLabels, final List<Labels<A>> childLabels) {
        final List<Double> parentDistribution = parentLabels.getDistribution();
        double sum = 0;
        for(final Labels<A> labels : childLabels) {
            labels.setLabelSet(parentLabels.getLabelSet());
            final double childSum = labels.getWeights().stream().mapToDouble(d -> d).sum();
            final List<Double> counts = labels.getCountsList();
            for(int j = 0; j < parentDistribution.size(); j++) {
                final double expected = parentDistribution.get(j) * childSum;
                double v = Math.pow(counts.get(j) - expected, 2);
                v /= expected;
                sum += v;
            }
        }
        return sum;
    }
}
