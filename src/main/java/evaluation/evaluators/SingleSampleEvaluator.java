/*
 * Copyright (C) 2019 xmw13bzu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package evaluation.evaluators;

import evaluation.storage.ClassifierResults;
import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Resamples the data provided once to form train and test sets, builds on the train
 * and evaluates on the test. Resampled according to the seed, and is deterministic,
 * using the standard InstanceTools.resampleInstances method
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class SingleSampleEvaluator extends SamplingEvaluator {
    double propInstancesInTrain = 0.5;
    
    public SingleSampleEvaluator() {
        super(0, false, false);
    }
    
    public SingleSampleEvaluator(int seed, boolean cloneData, boolean setClassMissing) {
        super(seed, cloneData, setClassMissing);
    }

    public double getPropInstancesInTrain() {
        return propInstancesInTrain;
    }

    public void setPropInstancesInTrain(double propInstancesInTrain) {
        this.propInstancesInTrain = propInstancesInTrain;
    }
    
    @Override
    public synchronized ClassifierResults evaluate(Classifier classifier, Instances dataset) throws Exception {
        Instances[] trainTest = InstanceTools.resampleInstances(dataset, seed, propInstancesInTrain);
        
        long estimateTime = System.nanoTime();
        long buildTime = System.nanoTime();
        classifier.buildClassifier(trainTest[0]);
        buildTime = System.nanoTime() - buildTime;
        
        SingleTestSetEvaluator eval = new SingleTestSetEvaluator(this.seed, this.cloneData, this.setClassMissing);
        ClassifierResults res = eval.evaluate(classifier, trainTest[1]);
        
        //eval should have set everything else 
        res.turnOffZeroTimingsErrors();
        res.setBuildTime(buildTime);
        res.setErrorEstimateTime(System.nanoTime() - estimateTime);
        res.turnOnZeroTimingsErrors();
        
        return res;
    }

}
