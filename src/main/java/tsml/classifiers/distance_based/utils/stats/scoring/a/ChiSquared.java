package tsml.classifiers.distance_based.utils.stats.scoring.a;

import tsml.classifiers.distance_based.utils.stats.scoring.v2.Labels;

import java.util.List;

public class ChiSquared implements Score {
    @Override public <A> double score(final Labels<A> parent, final List<Labels<A>> children) {
        final List<Double> parentDistribution = parent.getDistribution();
        double sum = 0;
        for(final Labels<A> labels : children) {
            labels.setLabelSet(parent.getLabelSet());
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
