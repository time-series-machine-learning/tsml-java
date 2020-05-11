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
 * Simply calculates this classifier's accuracy on each class
 * 
 * @author James Large
 */
public class TrainAccByClass extends ModuleWeightingScheme {
    
    public TrainAccByClass() {
        uniformWeighting = false;
    }

    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        double[] weights = new double[numClasses];
        for (int c = 0; c < numClasses; c++) 
            weights[c] = calcClassAcc(module.trainResults.confusionMatrix, c);

        return weights;
    }
    
    protected double calcClassAcc(double [][] confMat, int c) {
        double correct = confMat[c][c];
        double wrong = .0;
        for (int i = 0; i < confMat.length; i++)
            if (i!=c)
                wrong += confMat[c][i];
        
        return correct / (wrong+correct);
    }
    
}
