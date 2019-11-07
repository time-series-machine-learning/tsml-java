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
 * Simply uses the modules train acc as it's weighting. Extension: can raise the accuracy
 * to some power in order to scale the relative difference in accuracy between modules 
 * 
 * e.g, if raising all accuracies to power of 2, weights (0.7, 0.8, 0.9) become (0.49, 0.64, 0.81)
 * 
 * @author James Large
 */
public class TrainAcc extends ModuleWeightingScheme {

    private double power = 1.0;
    
    public TrainAcc() {
        uniformWeighting = true;
        needTrainPreds = false;
    }
    
    public TrainAcc(double power) {
        this.power = power;
        uniformWeighting = true;
        needTrainPreds = false;
    }
    
    public double getPower() { 
        return power;
    }
    
    public void setPower(double power) {
        this.power = power;
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        return makeUniformWeighting(Math.pow(module.trainResults.getAcc(), power), numClasses);
    }
    
    @Override
    public String toString() {
        return this.getClass().getSimpleName() + (power==1.0 ? "" : "(" + power + ")");
    }
    
}
