package tsml.classifiers.distance_based.utils.stats.scoring.a;

import tsml.classifiers.distance_based.utils.stats.scoring.v2.Labels;
import utilities.Utilities;

import java.util.List;

public class InfoEntropy implements Entropy {
    public <A> double entropy(Labels<A> labels) {
        return labels.getDistribution().stream().mapToDouble(d -> d * Utilities.log(d, 2)).sum() * -1;
    }
}
