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

import experiments.ClassifierLists;
import experiments.data.DatasetLoading;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Examples to show different ways of constructing classifiers, and basic usage
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class Ex02_Classifiers {

    public static void main(String[] args) throws Exception {
        
        // We'll use this data throughout, see Ex01_Datahandling
        int seed = 0;
        Instances[] trainTest = DatasetLoading.sampleItalyPowerDemand(seed);
        Instances train = trainTest[0];
        Instances test = trainTest[1];

        // Here's the super basic workflow, this is pure weka: 
        RandomForest randf = new RandomForest();                       
        randf.setNumTrees(500);
        randf.setSeed(seed);
        
        randf.buildClassifier(train);                                   //aka fit, train
        
        double acc = .0;
        for (Instance testInst : test) {
            double pred = randf.classifyInstance(testInst);             //aka predict
            //double [] dist = randf.distributionForInstance(testInst); //aka predict_proba
            
            if (pred == testInst.classValue())
                acc++;
        }
        
        acc /= test.numInstances();
        System.out.println("Random Forest accuracy on ItalyPowerDemand: " + acc);
    
        
        
        
        
        
        
        
        
        // All classifiers implement the Classifier interface. this guarantees 
        // the buildClassifier, classifyInstance and distributionForInstance methods, 
        // which is mainly what we want
        // Most if not all classifiers should extend AbstractClassifier, which adds 
        // on a little extra common functionality
        
        
        // There are also a number of classifiers listed in experiments.ClassifierLists
        // This class is updated over time and may eventually turn in to factories etc
        // on the backend, but for now what this is just a way to get a classifier 
        // with defined settings (parameters etc). We use this to record the exact
        // parameters used in papers for example. We also use this to instantiate
        // particular classifiers from a string argument when running on clusters
        
        Classifier classifier = ClassifierLists.setClassifierClassic("RandF", seed);
        classifier.buildClassifier(train);
        classifier.distributionForInstance(test.instance(0));
        
        
        
    }
    
}
