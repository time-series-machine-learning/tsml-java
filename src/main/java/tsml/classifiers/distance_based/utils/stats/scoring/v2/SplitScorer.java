package tsml.classifiers.distance_based.utils.stats.scoring.v2;

import tsml.classifiers.distance_based.utils.stats.scoring.v2.Labels;

import java.util.List;

public interface SplitScorer {

    <A> double score(Labels<A> parentLabels, List<Labels<A>> childLabels);
    
}
