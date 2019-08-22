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

package examples;

import evaluation.evaluators.CrossValidationEvaluator;
import evaluation.evaluators.Evaluator;
import evaluation.evaluators.MultiSamplingEvaluator;
import evaluation.evaluators.SingleTestSetEvaluator;
import evaluation.storage.ClassifierResults;
import experiments.ClassifierLists;
import experiments.data.DatasetLoading;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Examples to show different ways of evaluating classifiers
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class Ex03_BasicEvaluation {

    public static void main(String[] args) throws Exception {
        
        // We'll use this data throughout, see Ex01_Datahandling
        int seed = 0;
        Instances[] trainTest = DatasetLoading.sampleItalyPowerDemand(seed);
        Instances train = trainTest[0];
        Instances test = trainTest[1];
        
        // Let's use Random Forest throughout, see Ex02_Classifiers
        Classifier classifier = ClassifierLists.setClassifierClassic("RandF", seed);
        
        
        // We saw in Ex02_Classifiers that we can build on predefined train data, and test 
        // on predefined test data by looping over the instances
        
        // We can also evaluate using evaluation.Evaluators to get back an evaluation.storage.ClassifierResults
        // object, which we'll discuss further below. This functionality is still being expanded,
        // but the API is fairly set.
        
        // Here's the most basic evaluator, which replaces the looping over the test 
        // set in the previous example
        
        // We build the classifier ourselves on the train data
        classifier.buildClassifier(train);   
        
        // Setup the evaluator
        boolean cloneData = true, setClassMissing = true; 
        Evaluator testSetEval = new SingleTestSetEvaluator(seed, cloneData, setClassMissing);
        
        // And, in this case, test on the single held-out test set. 
        ClassifierResults testResults = testSetEval.evaluate(classifier, test);
        System.out.println("Random Forest accuracy on ItalyPowerDemand: " + testResults.getAcc());
        
        
        
        
        
        
        
        
        
        // Other evaluators currently implemented are for cross validation and random stratified resamples
        // Instead of building the classifier before-hand and passing that to the evaluator with
        // the test data, these will repeatedly build the classifier on each fold or resample. 
        // Let's generate an estimate of our error from the train data through cross validation.
        
        boolean cloneClassifier = false, maintainFoldClassifiers = false;
        MultiSamplingEvaluator cvEval = new CrossValidationEvaluator(seed, cloneData, setClassMissing, cloneClassifier, maintainFoldClassifiers);
        cvEval.setNumFolds(10);
        
        ClassifierResults trainResults = cvEval.evaluate(classifier, train);
        System.out.println("Random Forest average accuracy estimate on ItalyPowerDemand: " + trainResults.getAcc());

        for (int i = 0; i < 10; i++)
            System.out.println("\tCVFold " + i + " accuracy: " + cvEval.getFoldResults()[i].getAcc());
        
        
        // We've used ClassifierResults so far to retrieve the accuracy of a set of predictions
        // This is a general purpose predictions-storage class, which gets updated relatively 
        // often. It stores predictions, meta info and timings, can calculate eval metrics 
        // over them (accuracy, auroc, f1, etc.), and supports reading/writing to file.
        
        String mockFile = trainResults.writeFullResultsToString();
        System.out.println("\n\n" + mockFile);
    }
    
}
