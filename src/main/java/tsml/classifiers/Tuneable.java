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
 
/*
 Tunable interface enforces the method getDefaultParameterSearchSpace, for use 
with the general TunedClassifier class. 

ParameterSpace are created by calls to the ParameterSpace object with 
addParameter(String name, values), where values can be arrays or a List.


 */
package tsml.classifiers;

import evaluation.tuning.ParameterSpace;

/**
 * For classifiers which can be tuned, requires an overidden setOptions from abstract classifier in most cases.
 *
 * @author ajb
 */
public interface Tuneable {
    
    /**
     * getDefaultParameterSearchSpace returns the possible parameter values
     * that can be looked for with the TunedClassifier
     *
     * @return default parameter space for tuning
     */
    ParameterSpace getDefaultParameterSearchSpace();
}
