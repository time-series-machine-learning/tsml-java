package tsml.classifiers.distance_based.distances.dtw;

import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;

import java.io.Serializable;

public interface Windowed extends ParamHandler, Serializable {
    String WINDOW_SIZE_FLAG = "ws";
    String WINDOW_SIZE_PERCENTAGE_FLAG = "wsp";

    WindowParameter getWindowParameter();
    
    default int findWindowSize(int length) {
        return getWindowParameter().findWindowSize(length);
    }

    default int getWindowSize() {
        return getWindowParameter().getWindowSize();
    }

    default void setWindowSize(int windowSize) {
        getWindowParameter().setWindowSize(windowSize);
    }

    default double getWindowSizePercentage() {
        return getWindowParameter().getWindowSizePercentage();
    }

    default void setWindowSizePercentage(double windowSizePercentage) {
        getWindowParameter().setWindowSizePercentage(windowSizePercentage);
    }

    default boolean isWindowSizeInPercentage() {
        return getWindowParameter().isWindowSizeInPercentage();
    }

    @Override default void setParams(ParamSet param) throws Exception {
        ParamHandler.super.setParams(param);
    }

    @Override default ParamSet getParams() {
        return ParamHandler.super.getParams();
    }
}
