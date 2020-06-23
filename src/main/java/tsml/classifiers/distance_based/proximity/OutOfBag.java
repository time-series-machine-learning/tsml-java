package tsml.classifiers.distance_based.proximity;

import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;

public class OutOfBag extends BaseTrainEstimateMethod {
    public OutOfBag(final boolean rebuildAfterBagging) {
        this.rebuildAfterBagging = rebuildAfterBagging;
    }

    public static final String REBUILD_FLAG = "r";
    private boolean rebuildAfterBagging;

    public boolean isRebuildAfterBagging() {
        return rebuildAfterBagging;
    }

    public void setRebuildAfterBagging(final boolean rebuildAfterBagging) {
        this.rebuildAfterBagging = rebuildAfterBagging;
    }

    @Override public ParamSet getParams() {
        return super.getParams().add(REBUILD_FLAG, rebuildAfterBagging);
    }

    @Override public void setParams(final ParamSet paramSet) throws Exception {
        super.setParams(paramSet);
        ParamHandlerUtils.setParam(paramSet, REBUILD_FLAG, this::setRebuildAfterBagging, Boolean.class);
    }
}

