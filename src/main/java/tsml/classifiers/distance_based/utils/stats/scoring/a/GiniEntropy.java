package tsml.classifiers.distance_based.utils.stats.scoring.a;

import tsml.classifiers.distance_based.utils.stats.scoring.v2.Labels;

import java.util.List;

public class GiniEntropy implements Entropy {
    
    public <A> double entropy(Labels<A> labels) {
        return 1d - labels.getDistribution().stream().mapToDouble(d -> Math.pow(d, 2)).sum();
    }
}
