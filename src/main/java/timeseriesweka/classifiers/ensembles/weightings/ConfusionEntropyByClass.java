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
package timeseriesweka.classifiers.ensembles.weightings;

import timeseriesweka.classifiers.ensembles.EnsembleModule;

/**
 *
 * TODO Currently should not be used, as cannot guarantee the range of the weights,
 * which should be 0 to 1. Was a random adaption from CEN just to try it out.
 * 
 * @author James
 */
public class ConfusionEntropyByClass extends ConfusionEntropy {
    
    public ConfusionEntropyByClass() {
        uniformWeighting = false;
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
       double[] weights = new double[numClasses];
        for (int j = 0; j < numClasses; j++) 
            weights[j] = cen_j(module.trainResults.confusionMatrix, j, numClasses);

        return weights;
    }
    
}
