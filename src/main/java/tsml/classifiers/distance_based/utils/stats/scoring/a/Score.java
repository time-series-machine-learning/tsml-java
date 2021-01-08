package tsml.classifiers.distance_based.utils.stats.scoring.a;

import tsml.classifiers.distance_based.utils.stats.scoring.v2.Labels;

import java.util.List;
import java.util.stream.Collectors;

public interface Score {
    
    <A> double score(Labels<A> parent, List<Labels<A>> children);
    
}
