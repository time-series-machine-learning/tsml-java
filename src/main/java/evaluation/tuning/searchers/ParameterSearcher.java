package evaluation.tuning.searchers;

import evaluation.tuning.ParameterSet;
import evaluation.tuning.ParameterSpace;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public abstract class ParameterSearcher implements Iterable<ParameterSet> {
    protected ParameterSpace space = null;   
    protected int seed = 0;
    
    public ParameterSpace getParameterSpace() {
        return space;
    }
    
    public void setParameterSpace(ParameterSpace parameterSpace) {
        this.space = parameterSpace;
    }
    
    public void setSeed(int seed) {
        this.seed = seed;
    }
}
