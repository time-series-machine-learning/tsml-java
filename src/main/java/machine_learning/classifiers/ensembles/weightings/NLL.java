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
 * @author James Large (james.large@uea.ac.uk)
 */
public class NLL extends ModuleWeightingScheme {

    private double power = 1.0;
    
    public NLL() {
        uniformWeighting = true;
        needTrainPreds = true;
    }
    
    public NLL(double power) {
        this.power = power;
        uniformWeighting = true;
        needTrainPreds = true;
    }
    
    public double getPower() { 
        return power;
    }
    
    public void setPower(double power) {
        this.power = power;
    }
    
    public void defineWeightings(EnsembleModule[] modules, int numClasses) {
        double[] nlls = new double[modules.length];
        double sum = .0;
        
        for (int i = 0; i < nlls.length; i++) {
            nlls[i] = Math.pow(modules[i].trainResults.findNLL(), power);
            sum += nlls[i];
        }
        
        for (int i = 0; i < nlls.length; i++) {
            nlls[i] /= sum;
            nlls[i] = 1 - nlls[i];
            modules[i].posteriorWeights = makeUniformWeighting(nlls[i], numClasses);
        }
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        throw new UnsupportedOperationException("NLL weighting cannot be defined for a single module, "
                + "only in relation to the rest, call defineWeighings(...)");
    }
    
    @Override
    public String toString() {
        return this.getClass().getSimpleName() + (power==1.0 ? "" : "(" + power + ")");
    }
}