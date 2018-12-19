
package timeseriesweka.classifiers.ensembles.voting;

import timeseriesweka.classifiers.ensembles.EnsembleModule;
import weka.core.Instance;

/**
 *
 * krawczyk16combination
 * http://www.sciencedirect.com/science/article/pii/S0925231216002757
 * 
 * the optional pruning parameter/capability is not implemented 
 * 
 * @author James Large
 */
public class NP_MAX extends MajorityVote {

    protected double sigma = 1.0; //no idea if this makes a difference. paper says sigma is a parameter,
    //then neglects to mention it ever again, though shouldnt be 
    //in a quick test between sigma = 0.5,1,2, seemingly only 1 or 2 differences in predictions
    //out of millions, liekly down to double precision errors. jsut ignore it 
    
    
    public NP_MAX() {
        super();
    }
    
    public NP_MAX(int numClasses) {
        super(numClasses);
    }
    
    public NP_MAX(double sigma) {
        super();
        this.sigma = sigma;
    }
    
    public NP_MAX(double sigma,int numClasses) {
        super(numClasses);
        this.sigma = sigma;
    }
    
    @Override
    public void trainVotingScheme(EnsembleModule[] modules, int numClasses) {
        this.numClasses = numClasses;
        
        for (int c = 0; c < numClasses; c++) {
            
            double norm = .0;
            
            double np_max = modules[0].posteriorWeights[c];
            for (int m = 1; m < modules.length; m++) //find max of the support functions for this class
                if (modules[m].posteriorWeights[c] > np_max)
                    np_max = modules[m].posteriorWeights[c];
                   
            double[] newWeights = new double[modules.length];
            for (int m = 0; m < modules.length; m++) { 
                newWeights[m] = Math.pow(1. / sigma * Math.sqrt(2*Math.PI), (-modules[m].posteriorWeights[c] - np_max) / 2*sigma*sigma);
                //todo find and replace with proper gaussian function
                norm += newWeights[m];
            }
            
            //pruning skipped
            
            for (int m = 0; m < modules.length; m++)
                modules[m].posteriorWeights[c] = newWeights[m]/norm;
        }
    }
}
