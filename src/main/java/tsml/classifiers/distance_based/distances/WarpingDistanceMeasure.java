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
