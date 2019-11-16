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
 *
 * 
 * 
 * @author James Large james.large@uea.ac.uk
 */
public class NaiveBayesCombiner extends ModuleVotingScheme {

    //i.e the probability that this class, [*][][], is the real class given that 
    //this module, [][*][], predicted this class [][][*], based off the train confusion matrix
    //will do a form of laplace correction to form the probabilities, i.e all cells in the confusion matrix 
    //will have 1 added to them. theoretically this could majorly mess up results on train sets with small datasets
    //or heavily unbalanced class distributions
    protected double[/*actual class*/][/*module*/][/*predictedclass*/] postProbs;
    
    protected double[] priorClassProbs;
    
    protected boolean laplaceCorrection;
    
    public NaiveBayesCombiner() {
        this.needTrainPreds = true;
        this.laplaceCorrection = true;
    }
    
    public NaiveBayesCombiner(int numClasses) {
        this.numClasses = numClasses;
        this.needTrainPreds = true;
        this.laplaceCorrection = true;
    }
    
    public NaiveBayesCombiner(boolean laplaceCorrection) {
        this.needTrainPreds = true;
        this.laplaceCorrection = laplaceCorrection;
    }
    
    public NaiveBayesCombiner(boolean laplaceCorrection, int numClasses) {
        this.numClasses = numClasses;
        this.needTrainPreds = true;
        this.laplaceCorrection = laplaceCorrection;
    }
    
    @Override
    public void trainVotingScheme(EnsembleModule[] modules, int numClasses) throws Exception {
        this.numClasses = numClasses;
        
        //double[/*actual class*/][/*module*/][/*predictedclass*/] probs;
        postProbs = new double[numClasses][modules.length][numClasses];
        priorClassProbs = new double[numClasses];
        
        int correction = laplaceCorrection ? 1 : 0;
        
        for (int ac = 0; ac < numClasses; ac++) {
            double numInClass = 0;
            for (int pc = 0; pc < numClasses; pc++)
                numInClass += (modules[0].trainResults.confusionMatrix[ac][pc] + correction);
            
            priorClassProbs[ac] = numInClass / modules[0].trainResults.numInstances();
            
            for (int m = 0; m < modules.length; m++)
                for (int pc = 0; pc < numClasses; pc++)
                    postProbs[ac][m][pc] =  (modules[m].trainResults.confusionMatrix[ac][pc] + correction) / numInClass;
        }
    }
    
    @Override
    public double[] distributionForTrainInstance(EnsembleModule[] modules, int trainInstanceIndex) {
        double[] dist = new double[numClasses];
        for (int ac = 0; ac < numClasses; ac++)
            dist[ac] = 1;
        
        int pred;
        for (int m = 0; m < modules.length; m++) {
            pred = (int) modules[m].trainResults.getPredClassValue(trainInstanceIndex); 
            for (int ac = 0; ac < numClasses; ac++) {
                dist[ac] *= postProbs[ac][m][pred] *
                           modules[m].priorWeight * 
                           modules[m].posteriorWeights[pred];
            }
        }
        
        for (int ac = 0; ac < numClasses; ac++)
            dist[ac] /= priorClassProbs[ac];
        
        return normalise(dist);
    }
    
    @Override
    public double[] distributionForTestInstance(EnsembleModule[] modules, int testInstanceIndex) {
        double[] dist = new double[numClasses];
        for (int ac = 0; ac < numClasses; ac++)
            dist[ac] = 1;
        
        int pred;
        for (int m = 0; m < modules.length; m++) {
            pred = (int) modules[m].testResults.getPredClassValue(testInstanceIndex); 
            for (int ac = 0; ac < numClasses; ac++) {
                dist[ac] *= postProbs[ac][m][pred] *
                           modules[m].priorWeight * 
                           modules[m].posteriorWeights[pred];
            }
        }
        
        for (int ac = 0; ac < numClasses; ac++)
//            dist[ac] /= priorClassProbs[ac];
            dist[ac] *= priorClassProbs[ac]; //TODO double check
        
        return normalise(dist);
    }

    @Override
    public double[] distributionForInstance(EnsembleModule[] modules, Instance testInstance) throws Exception {
        double[] ensDist = new double[numClasses];
        for (int ac = 0; ac < numClasses; ac++)
            ensDist[ac] = 1;
        
        int pred;
        double[] mdist;
        for (int m = 0; m < modules.length; m++) {
            mdist = distributionForNewInstance(modules[m], testInstance);
            
            pred = (int)indexOfMax(mdist);
            for (int ac = 0; ac < numClasses; ac++) {
                ensDist[ac] *= postProbs[ac][m][pred] *
                           modules[m].priorWeight * 
                           modules[m].posteriorWeights[pred];
            }
        }
        
        for (int ac = 0; ac < numClasses; ac++)
            ensDist[ac] /= priorClassProbs[ac];
        
        return normalise(ensDist);
    }
    
}
