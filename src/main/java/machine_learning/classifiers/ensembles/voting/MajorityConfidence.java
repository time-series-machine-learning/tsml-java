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
package machine_learning.classifiers.ensembles.voting;

import machine_learning.classifiers.ensembles.AbstractEnsemble.EnsembleModule;
import weka.core.Instance;

/**
 * Uses the weighted confidences of each module that the instance is in EACH class (not just the most likely)
 * 
 * i.e in a 2-class problem, a module's distforinst maybe be .6 / .4, 
 *      .6 * weight_c1 will be added to class 1
 *      .4 * weight_c2 will be added to class 2 as well
 * 
 * @author James Large
 */
public class MajorityConfidence extends ModuleVotingScheme {
    
    public MajorityConfidence() {
    }
    
    public MajorityConfidence(int numClasses) {
        this.numClasses = numClasses;
    }
    
    @Override
    public void trainVotingScheme(EnsembleModule[] modules, int numClasses) {
        this.numClasses = numClasses;
    }

    @Override
    public double[] distributionForTrainInstance(EnsembleModule[] modules, int trainInstanceIndex) {
        double[] preds = new double[numClasses];
        
        for(int m = 0; m < modules.length; m++){
            double[] p=modules[m].trainResults.getProbabilityDistribution(trainInstanceIndex);
            for (int c = 0; c < numClasses; c++) {
                preds[c] += modules[m].priorWeight * 
                            modules[m].posteriorWeights[c] * p[c];
            }
        }
        
        return normalise(preds);
    }
    
    @Override
    public double[] distributionForTestInstance(EnsembleModule[] modules, int testInstanceIndex) {
        double[] preds = new double[numClasses];
        
        for(int m = 0; m < modules.length; m++){
            double[] p=modules[m].testResults.getProbabilityDistribution(testInstanceIndex);
            for (int c = 0; c < numClasses; c++) {
                preds[c] += modules[m].priorWeight * 
                            modules[m].posteriorWeights[c] * p[c];
            }
        }
        
        return normalise(preds);
    }

    @Override
    public double[] distributionForInstance(EnsembleModule[] modules, Instance testInstance) throws Exception {
        double[] preds = new double[numClasses];
        
        double[] dist;
        for(int m = 0; m < modules.length; m++){
            dist = distributionForNewInstance(modules[m], testInstance);
            
            for (int c = 0; c < numClasses; c++) {
                preds[c] += modules[m].priorWeight * 
                            modules[m].posteriorWeights[c] * 
                            dist[c];
            }
        }

        return normalise(preds);
    }
    
}
