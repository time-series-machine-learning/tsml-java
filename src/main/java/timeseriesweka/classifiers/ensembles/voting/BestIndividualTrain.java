package timeseriesweka.classifiers.ensembles.voting;

import timeseriesweka.classifiers.ensembles.EnsembleModule;
import weka.core.Instance;

/**
 *
 * TODO what if there's tie for best?  
 * 
 * The ensemble's distribution for an instance is equal to the single 'best' individual,
 * as defined by whatever (uniform) weighting scheme is being used. 
 * 
 * Mostly just written so that I can do the best individual within the existing framework for 
 * later testing
 * 
 * @author James Large james.large@uea.ac.uk
 */
public class BestIndividualTrain extends BestIndividual {
    
    public BestIndividualTrain() {
        super();
    }
    
    public BestIndividualTrain(int numClasses) {
        super(numClasses);
    }
    
    @Override
    public void trainVotingScheme(EnsembleModule[] modules, int numClasses) throws Exception {
        super.trainVotingScheme(modules, numClasses);
        
        double bestWeight = -1;
        for (int m = 0; m < modules.length; ++m) {
            
            //checking that the weights are uniform
            double prevWeight = modules[m].posteriorWeights[0];
            for (int c = 1; c < numClasses; ++c)  {
                if (prevWeight == modules[m].posteriorWeights[c])
                    prevWeight = modules[m].posteriorWeights[c];
                else 
                    throw new Exception("BestIndividualTrain cannot be used with non-uniform weighting schemes");
            }
            
            if (modules[m].posteriorWeights[0] > bestWeight) {
                bestWeight = modules[m].posteriorWeights[0];
                bestModule = m;
            }
        }
        
        bestModulesInds.add(bestModule);
        bestModulesNames.add(modules[bestModule].getModuleName());
        
        printlnDebug(modules[bestModule].getModuleName());
    }
    
}
