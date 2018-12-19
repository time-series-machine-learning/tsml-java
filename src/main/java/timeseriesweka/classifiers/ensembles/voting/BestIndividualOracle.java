package timeseriesweka.classifiers.ensembles.voting;

import utilities.DebugPrinting;
import timeseriesweka.classifiers.ensembles.EnsembleModule;
import weka.core.Instance;

/**
 * TODO what if there's tie for best? UNTESTED
 * 
 * The ensemble's distribution for an instance is equal to the single 'best' individual,
 * as defined by THEIR TEST ACCURACY. Results must have been read from file (i.e test preds
 * already exist at train time) Weighting scheme is irrelevant, only considers accuracy.
 * 
 * Mostly just written so that I can do the best individual within the existing framework for 
 * later testing
 * 
 * @author James Large james.large@uea.ac.uk
 */
public class BestIndividualOracle extends BestIndividual {

    
    public BestIndividualOracle() {
        super();
    }
    
    public BestIndividualOracle(int numClasses) {
        super(numClasses);
    }
    
    @Override
    public void trainVotingScheme(EnsembleModule[] modules, int numClasses) throws Exception {
        super.trainVotingScheme(modules, numClasses);
        
        double bestAcc = -1;
        for (int m = 0; m < modules.length; ++m) {         
            if (modules[m].testResults.acc > bestAcc) {
                bestAcc = modules[m].testResults.acc;
                bestModule = m;
            }
        }
        
        bestModulesInds.add(bestModule);
        bestModulesNames.add(modules[bestModule].getModuleName());
        
        printlnDebug(modules[bestModule].getModuleName());
    }
}
