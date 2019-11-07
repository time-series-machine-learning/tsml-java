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
import static utilities.GenericTools.indexOfMax;
import weka.core.Instance;

/**
 * Each class' probability is defined as the average of each classifier that predicts this class' weighted 
 * confidence that the instance is of this class
 * 
 * @author James Large
 */
public class AverageVoteByConfidence extends ModuleVotingScheme {
    
    public AverageVoteByConfidence() {
        
    }
    
    public AverageVoteByConfidence(int numClasses) {
        this.numClasses = numClasses;
    }
    
    @Override
    public void trainVotingScheme(EnsembleModule[] modules, int numClasses) {
        this.numClasses = numClasses;
    }

    @Override
    public double[] distributionForTrainInstance(EnsembleModule[] modules, int trainInstanceIndex) {
        double[] preds = new double[numClasses];
        int[] numPredsForClass = new int[numClasses];
        
        int pred; 
        for(int m = 0; m < modules.length; m++){
                pred = (int) modules[m].trainResults.getPredClassValue(trainInstanceIndex); 
                ++numPredsForClass[pred];
                double[] p=modules[m].trainResults.getProbabilityDistribution(trainInstanceIndex);
                preds[pred] += modules[m].priorWeight*modules[m].posteriorWeights[pred]*p[pred];
        }
        
        for (int c = 0; c < numClasses; c++) 
            if (numPredsForClass[c] != 0)
                preds[c]/=numPredsForClass[c];
    
        return normalise(preds);
    }
    
    @Override
    public double[] distributionForTestInstance(EnsembleModule[] modules, int testInstanceIndex) {
        double[] preds = new double[numClasses];
        int[] numPredsForClass = new int[numClasses];
        
        int pred; 
        for(int m = 0; m < modules.length; m++){
            pred = (int) modules[m].testResults.getPredClassValue(testInstanceIndex); 
            ++numPredsForClass[pred];
            double[] p=modules[m].testResults.getProbabilityDistribution(testInstanceIndex);
            preds[pred] += modules[m].priorWeight * 
                    modules[m].posteriorWeights[pred] * p[pred];
        }
        
        for (int c = 0; c < numClasses; c++) 
            if (numPredsForClass[c] != 0)
                preds[c]/=numPredsForClass[c];
    
        return normalise(preds);
    }

    @Override
    public double[] distributionForInstance(EnsembleModule[] modules, Instance testInstance) throws Exception {
        double[] preds = new double[numClasses];
        int[] numPredsForClass = new int[numClasses];
        
        double[] dist;
        int pred; 
        for(int m = 0; m < modules.length; m++){
            dist = distributionForNewInstance(modules[m], testInstance);
            
            pred = (int)indexOfMax(dist);
            ++numPredsForClass[pred];
            
            preds[pred] += modules[m].priorWeight * 
                        modules[m].posteriorWeights[pred] * 
                        dist[pred];
        }
        
        for (int c = 0; c < numClasses; c++) 
            if (numPredsForClass[c] != 0)
                preds[c]/=numPredsForClass[c];
    
        return normalise(preds);
    }
}
