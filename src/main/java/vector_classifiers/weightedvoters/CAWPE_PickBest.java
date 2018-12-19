
package vector_classifiers.weightedvoters;

import timeseriesweka.classifiers.ensembles.voting.BestIndividualTrain;
import timeseriesweka.classifiers.ensembles.weightings.EqualWeighting;
import vector_classifiers.CAWPE;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class CAWPE_PickBest extends CAWPE {
    public CAWPE_PickBest() { 
        super(); //sets default classifiers etc 
        
        //overwriting relevant parts 
        ensembleIdentifier = "HESCA_PickBest"; 
        weightingScheme = new EqualWeighting();
        votingScheme = new BestIndividualTrain();
    }   
}
