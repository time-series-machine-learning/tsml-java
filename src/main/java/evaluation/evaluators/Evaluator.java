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
package evaluation.evaluators;

import evaluation.storage.ClassifierResults;
import tsml.classifiers.distance_based.utils.classifiers.Copier;
import tsml.classifiers.distance_based.utils.classifiers.CopierUtils;
import tsml.classifiers.distance_based.utils.collections.params.ParamHandler;
import tsml.classifiers.distance_based.utils.collections.params.ParamSet;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Randomizable;

import java.io.Serializable;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public abstract class Evaluator implements Randomizable, ParamHandler, Serializable {
    
    int seed;
    
    /**
     * Flag for whether to clone the data. Defaults to false, as no classifier should 
     * be editing the data itself when training/testing, however setting this to true 
     * will guarantee that the same (jave) instantiations of (weka) instance(s) objects
     * can be reused in higher-level experimental code. 
     */
    boolean cloneData;
    
    /**
     * Each instance will have setClassMissing() called upon it. To ABSOLUTELY enforce that 
     * no classifier can cheat in any way (e.g some filter/transform inadvertently incorporates the class 
     * value back into the transformed data set). 
     * 
     * The only reason to leave this as false (as it has been by default, for backwards compatability reasons)
     * is that in higher level experimental code, the same (jave) instantiations of (weka) instance(s) objects are used multiple 
     * times, and the latter expects the class value to still be there (to check for correct predictions, e.g)
     */
    boolean setClassMissing;
    
    public Evaluator(int seed, boolean cloneData, boolean setClassMissing) {
        this.seed = seed;
        this.cloneData = cloneData;
        this.setClassMissing = setClassMissing;
    }
    
    @Override
    public int getSeed() {
        return seed;
    }
    
    @Override
    public void setSeed(int seed) {
        this.seed = seed;
    }

    /**
     * Flag for whether to clone the data. Defaults to false, as no classifier should 
     * be editing the data itself when training/testing, however setting this to true 
     * will guarantee that the same (jave) instantiations of (weka) instance(s) objects
     * can be reused in higher-level experimental code. 
     */
    public boolean getCloneData() {
        return cloneData;
    }

    /**
     * Flag for whether to clone the data. Defaults to false, as no classifier should 
     * be editing the data itself when training/testing, however setting this to true 
     * will guarantee that the same (jave) instantiations of (weka) instance(s) objects
     * can be reused in higher-level experimental code. 
     */
    public void setCloneData(boolean cloneData) {
        this.cloneData = cloneData;
    }
    
    /**
     * Each instance will have setClassMissing() called upon it. To ABSOLUTELY enforce that 
     * no classifier can cheat in any way (e.g some filter/transform inadvertently incorporates the class 
     * value back into the transformed data set). 
     * 
     * The only reason to leave this as false (as it has been by default, for backwards compatability reasons)
     * is that in higher level experimental code, the same (jave) instantiations of (weka) instance(s) objects are used multiple 
     * times, and the latter expects the class value to still be there (to check for correct predictions, e.g)
     */
    public boolean getSetClassMissing() {
        return setClassMissing;
    }
    
    /**
     * Each instance will have setClassMissing() called upon it. To ABSOLUTELY enforce that 
     * no classifier can cheat in any way (e.g some filter/transform inadvertently incorporates the class 
     * value back into the transformed data set). 
     * 
     * The only reason to leave this as false (as it has been by default, for backwards compatability reasons)
     * is that in higher level experimental code, the same (jave) instantiations of (weka) instance(s) objects are used multiple 
     * times, and the latter expects the class value to still be there (to check for correct predictions, e.g)
     */
    public void setSetClassMissing(boolean setClassMissing) {
        this.setClassMissing = setClassMissing;
    }
    
    public abstract ClassifierResults evaluate(Classifier classifier, Instances dataset) throws Exception;
    
    public Evaluator cloneEvaluator() {
        try {
            return (Evaluator) CopierUtils.deepCopy(this);
        } catch(Exception e) {
            throw new IllegalStateException(e);
        }
    }
}
