package tsml.classifiers.distance_based.utils.stats.scoring;

import java.io.Serializable;
import java.util.List;

public interface SplitScorer extends Serializable {
    
    <A> double score(Labels<A> parent, List<Labels<A>> children);
    
}
