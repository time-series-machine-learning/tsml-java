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
 * Non uniform weighting scheme, uses F-measure to give a weighting composed of 
 * the classifier's precision and recall *for each class*
 * 
 * @author James Large
 */
public class FScore extends ModuleWeightingScheme {

    private double beta = 1.0;
    private double power = 1.0;
    
    public FScore() {
        uniformWeighting = false;
        needTrainPreds = true;
    }
    
    public FScore(double power) {
        this.power = power;
        uniformWeighting = false;
        needTrainPreds = true;
    }
    
    public FScore(double power, double beta) {
        this.power = power;
        this.beta = beta;
        uniformWeighting = false;
        needTrainPreds = true;
    }
    
    public double getBeta() { 
        return beta;
    }
    
    public void setBeta(double beta) {
        this.beta = beta;
    }
    
    public double getPower() { 
        return power;
    }
    
    public void setPower(double power) {
        this.power = power;
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        double[] weights = new double[numClasses];
        for (int c = 0; c < numClasses; c++) {
            double weight = computeFScore(module.trainResults.confusionMatrix, c);
            weights[c] = Math.pow(weight, power);
        }
        return weights;
    }
    
    protected double computeFScore(double[][] confMat, int c) {
        double tp = confMat[c][c]; //[actual class][predicted class]
        if (tp == .0)
            return .0000001; 
        //some very small non-zero value, in the extreme case that no classifiers
        //in the entire ensemble classified cases of this class correctly
        //happened once on adiac (37 classes)
        
        double fp = 0.0, fn = 0.0;
        
        for (int i = 0; i < confMat.length; i++) {
            if (i!=c) {
                fp += confMat[i][c];
                fn += confMat[c][i];
            }
        }
        
        double precision = tp / (tp+fp);
        double recall = tp / (tp+fn);
        
        return (1+beta*beta) * (precision*recall) / ((beta*beta)*precision + recall);
    }
    
    @Override
    public String toString() {
        return this.getClass().getSimpleName() + "(" + power + "," + beta + ")";
    }
}
