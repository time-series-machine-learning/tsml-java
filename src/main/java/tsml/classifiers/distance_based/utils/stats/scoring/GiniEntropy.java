package tsml.classifiers.distance_based.utils.stats.scoring;

public class GiniEntropy implements PartitionEntropy {
    
    public <A> double entropy(Labels<A> labels) {
        return 1d - labels.getDistribution().stream().mapToDouble(d -> Math.pow(d, 2)).sum();
    }
}
