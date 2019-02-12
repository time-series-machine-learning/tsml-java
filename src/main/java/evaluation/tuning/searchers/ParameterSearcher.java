package evaluation.tuning.searchers;

import evaluation.tuning.ParameterSet;
import evaluation.tuning.ParameterSpace;
import java.io.File;
import utilities.FileHandlingTools;

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
    
    protected File[] findSavedParas() { 
        if (parameterSavingPath == null || parameterSavingPath == "")
            return new File[] { };
        
        return FileHandlingTools.listFilesContaining(parameterSavingPath, "fold" + seed + "_");
    }
    
    protected int findHowManyParasAlreadySaved() { 
        return findSavedParas().length;
    }
}

