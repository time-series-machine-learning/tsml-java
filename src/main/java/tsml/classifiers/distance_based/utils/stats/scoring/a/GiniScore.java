package tsml.classifiers.distance_based.utils.stats.scoring.a;

import tsml.classifiers.distance_based.utils.stats.scoring.v2.Labels;
import utilities.ArrayUtilities;

import java.util.List;

public class GiniScore implements Score {
    public <A> double score(Labels<A> parent, List<Labels<A>> children) {
        return gain(parent, children, new GiniEntropy());
    }
    
    protected static <A> double weightedInverseEntropy(Labels<A> parent, List<Labels<A>> children, Entropy entropy) {
        final double parentSum = parent.getWeightSum();
        double childEntropySum = 0;
        for(Labels<A> child : children) {
            double childEntropy = entropy.inverseEntropy(child);
            final double childSum = child.getWeightSum();
            final double proportion = childSum / parentSum;
            childEntropy *= proportion;
            childEntropySum += childEntropy;
        }
        return childEntropySum;
    }
    
    protected static <A> double weightedEntropy(Labels<A> parent, List<Labels<A>> children, Entropy entropy) {
        final double parentSum = parent.getWeightSum();
        double childEntropySum = 0;
        for(Labels<A> child : children) {
            double childEntropy = entropy.entropy(child);
            final double childSum = child.getWeightSum();
            final double proportion = childSum / parentSum;
            childEntropy *= proportion;
            childEntropySum += childEntropy;
        }
        return childEntropySum;
    }
    
    protected static <A> double gain(Labels<A> parent, List<Labels<A>> children, Entropy entropy) {
        final double parentEntropy = entropy.entropy(parent);
        double childEntropySum = weightedEntropy(parent, children, entropy);
        return parentEntropy - childEntropySum;
    }
    
}
