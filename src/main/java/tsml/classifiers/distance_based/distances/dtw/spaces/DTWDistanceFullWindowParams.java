package tsml.classifiers.distance_based.distances.dtw.spaces;

import tsml.classifiers.distance_based.utils.collections.params.ParamMap;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.data_containers.TimeSeriesInstances;

import static tsml.classifiers.distance_based.distances.dtw.DTW.WINDOW_FLAG;
import static tsml.classifiers.distance_based.utils.collections.CollectionUtils.newArrayList;

public class DTWDistanceFullWindowParams implements ParamSpaceBuilder {

    /**
     * Build a param space containing full window for dtw
     * @param data
     * @return
     */
    @Override public ParamSpace build(final TimeSeriesInstances data) {
        ParamMap params = new ParamMap();
        params.add(WINDOW_FLAG, newArrayList(1d));
        return new ParamSpace(params);
    }
}
