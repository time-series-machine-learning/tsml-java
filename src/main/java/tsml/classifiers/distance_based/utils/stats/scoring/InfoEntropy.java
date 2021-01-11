package tsml.classifiers.distance_based.utils.stats.scoring;

import utilities.Utilities;

public class InfoEntropy implements PartitionEntropy {
    public <A> double entropy(Labels<A> labels) {
        return labels.getDistribution().stream().mapToDouble(d -> d * Utilities.log(d, 2)).sum() * -1;
    }
}
