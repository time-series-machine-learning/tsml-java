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
 * Sets each module's weighting to Math.max(0.01, module.trainResults.acc - maxClassWeighting),
 * where maxClassWeighting is the proportion of cases belonging to the most common class, 
 * i.e the accuracy expected from a completely biased classifier
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class AvgCorrectedTrainAcc extends ModuleWeightingScheme {

    public AvgCorrectedTrainAcc() {
        uniformWeighting = true;
        needTrainPreds = false;
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        //made non zero (effectively 1% accuracy) in weird case that all classifiers get less than expected acc
        return makeUniformWeighting(Math.max(0.01, module.trainResults.getAcc() - (1.0/numClasses)), numClasses);
    }
    
    @Override
    public String toString() {
        return this.getClass().getSimpleName();
    }
}
