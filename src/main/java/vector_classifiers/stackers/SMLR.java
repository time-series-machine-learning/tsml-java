
package vector_classifiers.stackers;

import timeseriesweka.classifiers.ensembles.voting.stacking.StackingOnDists;
import timeseriesweka.classifiers.ensembles.weightings.EqualWeighting;
import vector_classifiers.CAWPE;
import vector_classifiers.MultiLinearRegression;

/**
 * Stacking with multi-response linear regression (MLR), Ting and Witten (1999) 
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class SMLR extends CAWPE {
    public SMLR() { 
        super(); //sets default classifiers etc 
        
        //overwriting relevant parts 
        ensembleIdentifier = "SMLR"; 
        weightingScheme = new EqualWeighting();
        votingScheme = new StackingOnDists(new MultiLinearRegression());
    }     
}