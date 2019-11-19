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
 * Each modules weighting is defined as it's accuracy proportional to the other modules, 
 * i.e weight_i = (acc_i) / (sum of all accs)
 * 
 * @author James Large james.large@uea.ac.uk
 */
public class ProportionalTrainAcc extends ModuleWeightingScheme {

    protected double sumOfTrainAccs = 0.;
    
    public ProportionalTrainAcc() {
        uniformWeighting = true;
        needTrainPreds = false;
    }
    
    public void defineWeightings(EnsembleModule[] modules, int numClasses) {
        for (EnsembleModule m : modules) 
            sumOfTrainAccs += m.trainResults.getAcc();
        
        for (EnsembleModule m : modules) 
            m.posteriorWeights = defineWeighting(m, numClasses);
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        return makeUniformWeighting(module.trainResults.getAcc() / sumOfTrainAccs, numClasses);
    }
    
}
