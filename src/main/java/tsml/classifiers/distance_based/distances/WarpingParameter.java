/*
 * This file is part of the UEA Time Series Machine Learning (TSML) toolbox.
 *
 * The UEA TSML toolbox is free software: you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License as published 
 * by the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * The UEA TSML toolbox is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with the UEA TSML toolbox. If not, see <https://www.gnu.org/licenses/>.
 */
 
package tsml.classifiers.distance_based.distances;
/**
 * @author George Oastler circa 2019
some comments added by Tony 2021
 */

import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandlerUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;

public class WarpingParameter implements
        WarpingDistanceMeasure, ParamHandler {
    // windowSize and windowSizePercentage -1 indicates full window
    private int windowSize = -1;
    private double windowSizePercentage = -1;
    private boolean windowSizeInPercentage = false;

    public int findWindowSize(int aLength) {
        // window should be somewhere from 0..len-1. window of 0 is ED, len-1 is Full DTW. Anything above is just
        // Full DTW
        final int result;
        if(windowSizeInPercentage) {
            if(windowSizePercentage > 1) {
                throw new IllegalArgumentException("window percentage size larger than 1");
            }
            if(windowSizePercentage < 0) {
                result = aLength - 1;
            } else {
                result = (int) (windowSizePercentage * (aLength - 1));
            }
        } else {
//            if(this.windowSize > aLength - 1) {
//                throw new IllegalArgumentException("window size larger than series length: " + this.windowSize + " vs"
//                    + " " + (aLength - 1));
//            }
            if(this.windowSize < 0) {
                result = aLength - 1;
            } else {
                result = this.windowSize;
            }
        }
        return result;
    }

    @Override public int getWindowSize() {
        return windowSize;
    }

    @Override public void setWindowSize(final int windowSize) {
        this.windowSize = windowSize;
        windowSizeInPercentage = false;
    }

    @Override public double getWindowSizePercentage() {
        return windowSizePercentage;
    }

    @Override public void setWindowSizePercentage(final double windowSizePercentage) {
        windowSizeInPercentage = true;
        this.windowSizePercentage = windowSizePercentage;
    }

    @Override public boolean isWindowSizeInPercentage() {
        return windowSizeInPercentage;
    }

    @Override
    public void setParams(final ParamSet param) throws Exception {
        WarpingDistanceMeasure.super.setParams(param);
        ParamHandlerUtils.setParam(param, WINDOW_SIZE_FLAG, this::setWindowSize, Integer.class);
        ParamHandlerUtils.setParam(param, WINDOW_SIZE_PERCENTAGE_FLAG, this::setWindowSizePercentage, Double.class);
    }

    @Override
    public ParamSet getParams() {
        final ParamSet paramSet = WarpingDistanceMeasure.super.getParams();
        if(windowSizeInPercentage) {
            paramSet.add(WINDOW_SIZE_PERCENTAGE_FLAG, windowSizePercentage);
        } else {
            paramSet.add(WINDOW_SIZE_FLAG, windowSize);
        }
        return paramSet;
    }

}
