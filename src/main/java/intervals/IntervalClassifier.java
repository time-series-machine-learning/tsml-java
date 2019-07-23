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
import evaluation.storage.ClassifierResults;
import experiments.ClassifierLists;
import experiments.Experiments;
import experiments.Experiments.ExperimentalArguments;
import experiments.data.DatasetLoading;
import intervals.IntervalHierarchy.Interval;
import java.io.File;
import utilities.TrainAccuracyEstimate;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author James Large (james.large@uea.ac.uk)
 */
public class IntervalClassifier extends AbstractClassifier implements TrainAccuracyEstimate {

    int seed = 0;
    boolean normaliseIntervals = false;

    Classifier proxy;
    Classifier target;
    boolean proxyAndTargetSame;    
    
    Evaluator eval;
    
    IntervalHierarchy intervals;
    Interval bestInterval;
    
    Instances trainHeader;
    
    //TrainAccuracyEstimate
    boolean TAE_estimateTargetError = false;
    String TAE_trainResultsPath;
    ClassifierResults TAE_targetTrainResults;
    
    public IntervalClassifier(Classifier classifier) throws Exception {
        this.proxy = classifier;
        this.target = AbstractClassifier.makeCopy(classifier);
        
        proxyAndTargetSame = true;
    }
    
    public IntervalClassifier(Classifier proxy, Classifier target) {
        this.proxy = proxy;
        this.target = target;
        
        proxyAndTargetSame = false;
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
        
        eval= new CrossValidationEvaluator(seed, false, false, true, false);
        ((CrossValidationEvaluator)eval).setNumFolds(5);
        eval.setSeed(seed);
        intervals = new IntervalHierarchy();
        intervals.buildHeirarchy(eval, proxy, data, normaliseIntervals);
        
        bestInterval = intervals.getBestInterval();
        Instances intervalData = IntervalCreation.crop_proportional(data, bestInterval.startPercent, bestInterval.endPercent, normaliseIntervals);     
        
        if (TAE_estimateTargetError) {
            if (proxyAndTargetSame)
                TAE_targetTrainResults = bestInterval.results;
            else 
                TAE_targetTrainResults = eval.evaluate(target, intervalData);
            
            if (TAE_trainResultsPath != null && !TAE_trainResultsPath.equals("")) 
                TAE_targetTrainResults.writeFullResultsToFile(TAE_trainResultsPath);
        }
        
        target.buildClassifier(intervalData);
    }

    
    @Override
    public double[] distributionForInstance(Instance testInst) throws Exception {
        trainHeader.add(testInst);
        Instance croppedTestInst = IntervalCreation.crop_proportional(trainHeader, bestInterval.startPercent, bestInterval.endPercent, normaliseIntervals).remove(0);     
        
        return target.distributionForInstance(croppedTestInst);
    }
    
    
    


    @Override //TrainAccuracyEstimate
    public void setFindTrainAccuracyEstimate(boolean setCV) {
        TAE_estimateTargetError = setCV;
    }

    @Override //TrainAccuracyEstimate
    public void writeCVTrainToFile(String train) {
        TAE_trainResultsPath = train;
        TAE_estimateTargetError = true;
    }

    @Override //TrainAccuracyEstimate
    public ClassifierResults getTrainResults() {
        return TAE_targetTrainResults;
    }
    
    
    
    public static void main(String[] args) throws Exception {
        ExperimentalArguments exp = new ExperimentalArguments();
        exp.dataReadLocation="C:/TSCProblems2018/";//Where to get data                
        exp.resultsWriteLocation="C:/Temp/intervalExpTest/";//Where to write results   
        exp.classifierName="IntervalED";
        exp.datasetName="GunPoint";
        exp.foldId=0;
        exp.generateErrorEstimateOnTrainSet = true;
        exp.forceEvaluation = true;
                
        Classifier classifier = new IntervalClassifier(ClassifierLists.setClassifierClassic("ED", exp.foldId));
        Instances[] trainTest = DatasetLoading.sampleDataset(exp.dataReadLocation, exp.datasetName, exp.foldId);
        
        String fullResPath = exp.resultsWriteLocation + exp.classifierName + "/Predictions/" + exp.datasetName + "/";
        (new File(fullResPath)).mkdirs();
        
        Experiments.runExperiment(exp, trainTest[0], trainTest[1], classifier, fullResPath);
    }
}
