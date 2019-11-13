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
 * Non uniform weighting scheme, calculates the recall for each class and uses those 
 * as the classifier's weighting
 * 
 * @author James Large
 */
public class RecallByClass extends ModuleWeightingScheme {

    public RecallByClass() {
        uniformWeighting = false;
        needTrainPreds = true;
    }
    
    @Override
    protected double[] defineWeighting(EnsembleModule module, int numClasses) {
        double[] weights = new double[numClasses];
        for (int c = 0; c < numClasses; c++) 
            weights[c] = computeRecall(module.trainResults.confusionMatrix, c);

        return weights;
    }
    
    protected double computeRecall(double[][] confMat, int c) {
        double tp = confMat[c][c]; //[actual class][predicted class]
        if (tp == .0)
            return .0000001; 
        //some very small non-zero value, in the extreme case that no classifiers
        //in the entire ensemble classified cases of this class correctly
        //happened once on adiac (37 classes)
        
        double fn = 0.0;
        
        for (int i = 0; i < confMat.length; i++)
            if (i!=c)
                fn += confMat[c][i];
        
        double recall = tp / (tp+fn);
        return recall;
    }
    
}
