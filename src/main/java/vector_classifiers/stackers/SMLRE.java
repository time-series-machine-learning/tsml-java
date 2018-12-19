
package vector_classifiers.stackers;

import timeseriesweka.classifiers.ensembles.voting.stacking.StackingOnExtendedSetOfFeatures;
import timeseriesweka.classifiers.ensembles.weightings.EqualWeighting;
import vector_classifiers.CAWPE;
import vector_classifiers.MultiLinearRegression;

/**
 * Stacking with MLR and an extended set of meta-level attributes, Dzeroski and Zenko (2004)
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class SMLRE extends CAWPE{
    public SMLRE() { 
        super(); //sets default classifiers etc 
        
        //overwriting relevant parts 
        ensembleIdentifier = "SMLRE"; 
        weightingScheme = new EqualWeighting();
        votingScheme = new StackingOnExtendedSetOfFeatures(new MultiLinearRegression());
    }   
}
