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

package intervals;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.Evaluator;
import intervals.IntervalHierarchy.Interval;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class IntervalClassifier extends AbstractClassifier {

    int seed = 0;
    boolean normaliseIntervals = false;

    Classifier proxy;
    Classifier target;
    
    Evaluator eval = new CrossValidationEvaluator();
    
    IntervalHierarchy intervals;
    Interval bestInterval;
    
    Instances trainHeader;
    
    public IntervalClassifier(Classifier classifier) throws Exception {
        this.proxy = classifier;
        this.target = AbstractClassifier.makeCopy(classifier);
    }
    
    public IntervalClassifier(Classifier proxy, Classifier target) {
        this.proxy = proxy;
        this.target = target;
    }

    public int getSeed() {
        return seed;
    }

    public void setSeed(int seed) {
        this.seed = seed;
    }
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        trainHeader = new Instances(data, 0); //for cropping test instances
        
        eval.setSeed(seed);
        intervals.buildHeirarchy(eval, proxy, data, normaliseIntervals);
        
        bestInterval = intervals.getBestInterval();
        Instances intervalData = IntervalCreation.crop_proportional(data, bestInterval.startPercent, bestInterval.endPercent, normaliseIntervals);     
        
        target.buildClassifier(intervalData);
    }

    
    @Override
    public double[] distributionForInstance(Instance testInst) throws Exception {
        trainHeader.add(testInst);
        Instance croppedTestInst = IntervalCreation.crop_proportional(trainHeader, bestInterval.startPercent, bestInterval.endPercent, normaliseIntervals).remove(0);     
        
        return target.distributionForInstance(testInst);
    }
}
