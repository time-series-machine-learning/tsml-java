package tsml.classifiers.distance_based.utils.stats.scoring;

import java.util.List;

public interface SplitScorer {
    
    <A> double score(Labels<A> parent, List<Labels<A>> children);
    
}
