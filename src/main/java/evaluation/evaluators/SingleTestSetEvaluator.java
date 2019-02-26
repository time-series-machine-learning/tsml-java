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

import evaluation.evaluators.Evaluator;
import evaluation.storage.ClassifierResults;
import java.util.concurrent.TimeUnit;
import static utilities.GenericTools.indexOfMax;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Simply gathers predictions from the (assumed to have already been built/trained) passed classifier
 * on the data given, assumed to have already been sampled/set up as desired. 
 * 
 * distributionForInstance(Instance) MUST be defined, even if the classifier only really returns 
 * a one-hot distribution
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class SingleTestSetEvaluator extends Evaluator {
    
    public SingleTestSetEvaluator() {
        super(0,false,false);
    }
    public SingleTestSetEvaluator(int seed, boolean cloneData, boolean setClassMissing) {
        super(seed,cloneData,setClassMissing);
    }
    
    @Override
    public ClassifierResults evaluate(Classifier classifier, Instances dataset) throws Exception {
        Instances insts = null;
        if (cloneData)
            insts = new Instances(dataset);
        else 
            insts = dataset;
        
        ClassifierResults res = new ClassifierResults(insts.numClasses());
        res.setTimeUnit(TimeUnit.NANOSECONDS);
        
        res.turnOffZeroTimingsErrors();
        for (Instance testinst : insts) {
            double trueClassVal = testinst.classValue();
            if (setClassMissing)
                testinst.setClassMissing();
            
            long startTime = System.nanoTime();
            double[] dist = classifier.distributionForInstance(testinst);
            long predTime = System.nanoTime() - startTime;
            res.addPrediction(trueClassVal, dist, indexOfMax(dist), predTime, "");
        }
        res.turnOnZeroTimingsErrors();
        
        res.findAllStatsOnce(); 
        return res;
    }

}
