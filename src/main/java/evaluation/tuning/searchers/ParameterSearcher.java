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
    protected String parameterSavingPath = null;
    
    public ParameterSpace getParameterSpace() {
        return space;
    }
    
    protected String indsToString(int[] inds) {
        StringBuilder sb = new StringBuilder();
        sb.append(inds[0]);

        for (int i = 1; i < inds.length; i++)
            sb.append("_").append(inds[i]);

        return sb.toString();
    }

    public void setParameterSpace(ParameterSpace parameterSpace) {
        this.space = parameterSpace;
    }
    
    public void setSeed(int seed) {
        this.seed = seed;
    }
    
    public void setParameterSavingPath(String path) { 
        this.parameterSavingPath = path;
    }
    
    public String getParameterSavingPath() { 
        return parameterSavingPath;   
    }
}

