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
 
package tsml.transformers;

import tsml.data_containers.TimeSeriesInstances;
import tsml.data_containers.utilities.Converter;
import weka.core.Instance;
import weka.core.Instances;

public abstract class BaseTrainableTransformer implements TrainableTransformer {

    protected boolean fitted;

    @Override public void fit(final TimeSeriesInstances data) {
        fitted = true;
    }

    public void reset() {
        fitted = false;
    }

    @Override public boolean isFit() {
        return fitted;
    }

    @Override public Instance transform(final Instance inst) {
        return Converter.toArff(transform(Converter.fromArff(inst)));
    }

    @Override public void fit(final Instances data) {
        fit(Converter.fromArff(data));
    }
}
