package tsml.data_containers;

import java.util.List;

/**
 * Data structure able to store a time series instance.
 * it can be standard (univariate, no missing, equally sampled series) or
 * complex (multivariate, unequal length, unequally spaced, univariate or multivariate time series).
 *
 * Should Instances be immutable after creation? Meta data is calculated on creation, mutability can break this
 */

public class TimeSeriesInstance {
    List<TimeSeries> ts;
    Integer classLabel;

}
