/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package machine_learning.classifiers.ensembles.weightings;

import machine_learning.classifiers.ensembles.AbstractEnsemble.EnsembleModule;


/**
 *
 * Gives equal weights to all modules, i.e simple majority vote
 * 
 * @author James Large
 */
public class EqualWeighting extends ModuleWeightingScheme {

    public EqualWeighting() {
        uniformWeighting = true;
        needTrainPreds = false;
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule trainPredictions, int numClasses) {
        return makeUniformWeighting(1.0, numClasses);
    }
    
}
