package tsml.classifiers.distance_based.utils.stats.scoring;

import tsml.data_containers.TimeSeriesInstance;
import tsml.data_containers.utilities.Converter;
import weka.core.Instances;

import java.io.Serializable;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Purpose: score the partitioning of some data into subsets.
 */
public interface PartitionScorer extends Serializable {
    double findScore(Instances parent, List<Instances> children);

    default double findScore(TimeSeriesInstance parent, List<Instances> children) {
        return findScore(Converter.toArff(parent), children.stream().map(Converter::toArff).collect(Collectors.toList()));
    }
}
