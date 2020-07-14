package tsml.classifiers.distance_based.distances;

import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;

import java.io.Serializable;

public interface WarpingDistanceMeasure extends ParamHandler, Serializable {
    String WINDOW_SIZE_FLAG = "ws";
    String WINDOW_SIZE_PERCENTAGE_FLAG = "wsp";

    int findWindowSize(int length);

    int getWindowSize();

    void setWindowSize(int windowSize);

    double getWindowSizePercentage();

    void setWindowSizePercentage(double windowSizePercentage);

    boolean isWindowSizeInPercentage();

    @Override default void setParams(ParamSet param) throws Exception {
        ParamHandler.super.setParams(param);
    }

    @Override default ParamSet getParams() {
        return ParamHandler.super.getParams();
    }
}
