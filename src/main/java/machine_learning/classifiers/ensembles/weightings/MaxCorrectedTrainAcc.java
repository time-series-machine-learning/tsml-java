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
public class MaxCorrectedTrainAcc extends ModuleWeightingScheme {

    double maxClassWeighting;
    
    public MaxCorrectedTrainAcc() {
        uniformWeighting = true;
        needTrainPreds = true;
    }
    
    public void defineWeightings(EnsembleModule[] modules, int numClasses) {
        double[] classDist = new double[numClasses];
        for (int i = 0; i < modules[0].trainResults.getTrueClassValsAsArray().length; i++)
            classDist[(int)modules[0].trainResults.getTrueClassValsAsArray()[i]]++;
        
        maxClassWeighting = classDist[0];
        for (int i = 1; i < classDist.length; i++) 
            if (classDist[i] > maxClassWeighting)
                maxClassWeighting = classDist[i];
        
        for (EnsembleModule m : modules) //by default, sets weights independently for each module
            m.posteriorWeights = defineWeighting(m, numClasses);
        
        //some schemes may sets weights for each moduel relative to the rest, and 
        //so will need to override this method
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        //made non zero (effectively 1% accuracy) in weird case that all classifiers get less than expected acc
        return makeUniformWeighting(Math.max(0.01, module.trainResults.getAcc() - maxClassWeighting), numClasses);
    }
    
    @Override
    public String toString() {
        return this.getClass().getSimpleName();
    }
}
