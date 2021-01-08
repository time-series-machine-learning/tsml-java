package tsml.classifiers.distance_based.utils.stats.scoring.a;

import tsml.classifiers.distance_based.utils.stats.scoring.v2.Labels;

import java.util.List;

import static tsml.classifiers.distance_based.utils.stats.scoring.a.GiniScore.weightedEntropy;
import static tsml.classifiers.distance_based.utils.stats.scoring.a.GiniScore.weightedInverseEntropy;

public interface Entropy extends Score {
    
    <A> double entropy(Labels<A> labels);

    @Override default <A> double score(Labels<A> parent, List<Labels<A>> children) {
        // invert the weighted entropy as entropy is inverse, i.e. larger values mean worse. Score is the other way around, larger values are better. Therefore multiply by -1 to invert the entropy into a score (though will be less than 0!)
        return weightedInverseEntropy(parent, children, this);
        
        // OR
        
        // just negate the weighted sum of entropies. I.e. larger entropies would become small and small values greater, inverting the range as required
//        return -1d * weightedEntropy(parent, children, this);
    }
    
    default <A> double inverseEntropy(Labels<A> labels) {
        // the label set contains a single entry for each unique label. This is the worst distribution possible, i.e. uniform dist, over all available classes
        // entropy is defined as lower values == better and vice versa
        // inverse entropy inverts this to higher values == better, similar to a score
        // to invert, we use the worst possible distribution of labels (the label set) to produce the worst entropy. Then subtract from that the actual entropy of all labels
        return entropy(new Labels<>(labels.getLabelSet())) - entropy(labels);
    }
}
