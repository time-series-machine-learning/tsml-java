
package vector_classifiers.stackers;

import timeseriesweka.classifiers.ensembles.voting.stacking.StackingOnDists;
import timeseriesweka.classifiers.ensembles.weightings.EqualWeighting;
import vector_classifiers.CAWPE;
import vector_classifiers.MultiResponseModelTrees;

/**
 * Stacking with multi-response model trees. M5 is used to induce the
 * model trees at the meta level. Dzeroski and Zenko (2004)
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class SMM5 extends CAWPE {
    public SMM5() { 
        super(); //sets default classifiers etc 
        
        //overwriting relevant parts 
        ensembleIdentifier = "SMM5"; 
        weightingScheme = new EqualWeighting();
        votingScheme = new StackingOnDists(new MultiResponseModelTrees());
    }  
}
