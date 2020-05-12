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
 * Uses Confusion Entropy (CEN) to weight modules, which is a measure related to the 
 * entropy of a confusion matrix
 * 
 * Reportedly unreliable for 2 class matrices in some cases, implemented for completeness 
 * 
 * http://cs.tju.edu.cn/szdw/jsfjs/huqinghua/papers/A%20novel%20measure%20for%20evaluating%20classifiers.pdf
 * 
 * @author James Large
 */
public class ConfusionEntropy extends ModuleWeightingScheme {

    public ConfusionEntropy() {
        uniformWeighting = true;
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        return makeUniformWeighting(computeCEN(module.trainResults.confusionMatrix), numClasses);
    }
    
    protected double computeCEN(double[][] confMat) {
        double cen = .0;
        
        int n = confMat.length;
        double cen_j = .0;
        double p_j = .0;
        
        for (int j = 0; j < n; j++) {
            p_j = p_j(confMat, j, n);
            cen_j = cen_j(confMat, j, n);
            
            cen += p_j * cen_j;
        }
        
        return cen;
    }
    
    protected double cen_j(double[][] confMat, int j, int n) {
        double logbase = Math.log(2*(n-1));
        
        double cen_j = 0.0;
        double pK_kj = .0, pJ_kj = .0; 
        
        for (int k = 0; k < n; k++) {
            if (k != j) {
                pK_kj = pK_kj(confMat, j, k, n);
                pJ_kj = pJ_kj(confMat, j, k, n);  

                //using logb(n) = log(n) / log(b) identity to get to correct base
                cen_j -= pK_kj == .0 ? .0 : (pK_kj * (Math.log(pK_kj) / logbase)); 
                cen_j -= pJ_kj == .0 ? .0 : (pJ_kj * (Math.log(pJ_kj) / logbase));
            }
        }
        
        return cen_j;
    }
    
    protected double p_j(double[][] confMat, int j, int n) {
        double pj = 0.0;
        double den = 0.0;
        
        for (int k = 0; k < n; k++) {
            pj += confMat[j][k] + confMat[k][j];
            for (int l = 0; l < n; l++)
                den += confMat[k][l];
        }
        
        return pj / (2*den);
    }
    
    protected double pK_kj(double[][] confMat, int j, int k, int n) {
        double C_kj = confMat[k][j];
        double den = 0.0;
        
        for (int l = 0; l < n; l++)
            den += confMat[k][l] + confMat[l][k];
        
        return C_kj / den;
    }
    
    protected double pJ_kj(double[][] confMat, int j, int k, int n) {
        double C_kj = confMat[k][j];
        double den = 0.0;
        
        for (int l = 0; l < n; l++)
            den += confMat[j][l] + confMat[l][j];
        
        return C_kj / den;
    }
    
}
