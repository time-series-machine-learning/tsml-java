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
import java.util.concurrent.Callable;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Todo rename so something that makes sense, decide how to implement this 'nicely'
 * 
 * Takes two instances objects designated as the train and test sets, builds on the train
 * and evaluates on the test. Effectively a helper class for actual evaluators that
 * generate train and test sets and to help with spawning threads for them
 * 
 * This is not technically an evaluator, since it takes two instances objects instead of 
 * one. Needs discussion for whether we want to change around the hierarchy of just leave 
 * this as a separate entity. Could have two kinds of evaluator, single set and presplit etc
 * 
 * Single test set sort of works towards this already, with the assumption that the classifier 
 * has already been built on some other data
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class BuildTestEvaluation {
        
    SingleTestSetEvaluator eval; 
    Instances train;
    Instances test;
    Classifier classifier;
    
    ClassifierResults result;
    
    public BuildTestEvaluation(SingleTestSetEvaluator eval, Instances train, Instances test, Classifier classifier) {
        this.eval = eval;
        this.train = train;
        this.test = test;
        this.classifier = classifier;
    }
    
    public BuildTestEvaluation(SingleTestSetEvaluator eval, Instances[] trainTest, Classifier classifier) {
        this.eval = eval;
        this.train = trainTest[0];
        this.test = trainTest[1];
        this.classifier = classifier;
    }
       
    public synchronized ClassifierResults evaluate() throws Exception { 
        long buildTime = System.nanoTime();
        classifier.buildClassifier(train);
        buildTime = System.nanoTime() - buildTime;
        
        ClassifierResults res = eval.evaluate(classifier, test);
        
        res.turnOffZeroTimingsErrors();
        res.setBuildTime(buildTime);
        res.turnOnZeroTimingsErrors();
        
        return res;
    }
    
}
