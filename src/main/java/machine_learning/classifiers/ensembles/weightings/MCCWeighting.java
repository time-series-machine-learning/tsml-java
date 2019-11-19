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
 * Uses the Matthews Correlation Coefficient (MCC) to define the weighting of a module
 * MCC is a score calculated from the confusion matrix of the module's predictions
 * 
 * @author James Large
 */
public class MCCWeighting extends ModuleWeightingScheme {
    
    private double power = 1.0;
    
    public MCCWeighting() {
        uniformWeighting = true;
    }
    
    public MCCWeighting(double power) {
        this.power = power;
        uniformWeighting = true;
    }
    
    public double getPower() { 
        return power;
    }
    
    public void setPower(double power) {
        this.power = power;
    }
    
    @Override
    public double[] defineWeighting(EnsembleModule module, int numClasses) {
        //mcc returns vals in range -1,1, need it in range 0,1, so (w + 1)/2
        double weight = (module.trainResults.mcc + 1) / 2;
        return makeUniformWeighting(Math.pow(weight, power), numClasses);
    }
    
//    /**
//     * todo could easily be optimised further if really wanted
//     */
//    public double computeMCC(double[][] confusionMatrix) {
//        
//        double num=0.0;
//        for (int k = 0; k < confusionMatrix.length; ++k)
//            for (int l = 0; l < confusionMatrix.length; ++l)
//                for (int m = 0; m < confusionMatrix.length; ++m) 
//                    num += (confusionMatrix[k][k]*confusionMatrix[m][l])-
//                            (confusionMatrix[l][k]*confusionMatrix[k][m]);
//
//        if (num == 0.0)
//            return 0;
//        
//        double den1 = 0.0; 
//        double den2 = 0.0;
//        for (int k = 0; k < confusionMatrix.length; ++k) {
//            
//            double den1Part1=0.0;
//            double den2Part1=0.0;
//            for (int l = 0; l < confusionMatrix.length; ++l) {
//                den1Part1 += confusionMatrix[l][k];
//                den2Part1 += confusionMatrix[k][l];
//            }
//
//            double den1Part2=0.0;
//            double den2Part2=0.0;
//            for (int kp = 0; kp < confusionMatrix.length; ++kp)
//                if (kp!=k) {
//                    for (int lp = 0; lp < confusionMatrix.length; ++lp) {
//                        den1Part2 += confusionMatrix[lp][kp];
//                        den2Part2 += confusionMatrix[kp][lp];
//                    }
//                }
//                      
//            den1 += den1Part1 * den1Part2;
//            den2 += den2Part1 * den2Part2;
//        }
//        
//        return num / (Math.sqrt(den1)*Math.sqrt(den2));
//    }
    
    @Override
    public String toString() {
        return this.getClass().getSimpleName() + (power==1.0 ? "" : "(" + power + ")");
    }
}
