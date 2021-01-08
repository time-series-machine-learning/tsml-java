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
 
package tsml.classifiers.legacy;

import utilities.InstanceTools;
import weka.core.Instances;

/**
 * Indicates that the class can subsample the train set if the option is set
 * 
 * @author ajb
 */
public interface SubSampleTrainer {

    public void subSampleTrain(double prop, int seed);
    default Instances subSample(Instances full, double proportion, int seed){
        return InstanceTools.subSampleFixedProportion(full, proportion, seed);
    }
}
