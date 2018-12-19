
package vector_classifiers.weightedvoters;

import timeseriesweka.classifiers.ensembles.voting.MajorityVote;
import timeseriesweka.classifiers.ensembles.weightings.EqualWeighting;
import vector_classifiers.CAWPE;    

/**
 * Implemented as separate classifier for explicit comparison, from Kuncheva and Rodríguez (2014)
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class CAWPE_MajorityVote extends CAWPE {
    public CAWPE_MajorityVote() { 
        super(); //sets default classifiers etc 
        
        //overwriting relevant parts 
        ensembleIdentifier = "HESCA_MajorityVote"; 
        weightingScheme = new EqualWeighting();
        votingScheme = new MajorityVote();
    }   
}
