package tsml.classifiers.distance_based.distances;

import tsml.classifiers.distance_based.distances.dtw.DTWDistanceConfigs;
import tsml.classifiers.distance_based.distances.ed.EDistanceConfigs;
import tsml.classifiers.distance_based.distances.erp.ERPDistanceConfigs;
import tsml.classifiers.distance_based.distances.lcss.LCSSDistanceConfigs;
import tsml.classifiers.distance_based.distances.msm.MSMDistanceConfigs;
import tsml.classifiers.distance_based.distances.twed.TWEDistanceConfigs;
import tsml.classifiers.distance_based.distances.wdtw.WDTWDistanceConfigs;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpace;
import tsml.classifiers.distance_based.utils.collections.params.ParamSpaceBuilder;
import tsml.data_containers.TimeSeriesInstances;
import weka.core.Instances;

public enum DistanceMeasureSpaceBuilder implements ParamSpaceBuilder {
    DTW(new DTWDistanceConfigs.DTWSpaceBuilder()),
    DDTW(new DTWDistanceConfigs.DDTWSpaceBuilder()),
    WDTW(new WDTWDistanceConfigs.WDTWSpaceBuilder()),
    WDDTW(new WDTWDistanceConfigs.WDDTWSpaceBuilder()),
    ERP(new ERPDistanceConfigs.ERPSpaceBuilder()),
    LCSS(new LCSSDistanceConfigs.LCSSSpaceBuilder()),
    MSM(new MSMDistanceConfigs.MSMSpaceBuilder()),
    TWED(new TWEDistanceConfigs.TWEDSpaceBuilder()),
    ED(new EDistanceConfigs.EDSpaceBuilder()),
    FULL_DTW(new DTWDistanceConfigs.FullWindowDTWSpaceBuilder()),
    FULL_DDTW(new DTWDistanceConfigs.FullWindowDDTWSpaceBuilder()),
    ;
    
    DistanceMeasureSpaceBuilder(final ParamSpaceBuilder builder) {
        this.builder = builder;
    }

    private final ParamSpaceBuilder builder;

    @Override public ParamSpace build(final TimeSeriesInstances data) {
        return builder.build(data);
    }

    @Override public ParamSpace build(final Instances data) {
        return builder.build(data);
    }
}
