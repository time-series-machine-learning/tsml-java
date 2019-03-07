/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
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

