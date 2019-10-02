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

import evaluation.MultipleClassifierEvaluation;
import evaluation.storage.ClassifierResults;
import evaluation.storage.ClassifierResultsCollection;
import experiments.Experiments;
import java.util.Arrays;

/**
 * Examples on how to handle collections of results and use the MultipleClassifierEvaluation pipeline
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public class Ex05_ThoroughEvaluation {

    public static void main(String[] args) throws Exception {
        
        // First of all, let's generate some results. You'll already have these if you ran Ex04_ThoroughExperiments,
        // but we'll do it here if not.
        
        // NOTE: Again, if you want to run this file, you'll need to define an 
        // acceptable location to write some small files as examples: 
        String resultsPath = "C:/Temp/ueatscExamples/";
        String[] classifiers = { "SVML", "ED", "C45" }; 
        String[] datasets = { "ItalyPowerDemand", "Beef" }; 
        int numFolds = 3;
        
        Experiments.ExperimentalArguments expThreaded = new Experiments.ExperimentalArguments();
        expThreaded.dataReadLocation = "src/main/java/experiments/data/tsc/"; 
        expThreaded.resultsWriteLocation = resultsPath;                      
        
        Experiments.setupAndRunMultipleExperimentsThreaded(expThreaded, classifiers, datasets, 0, numFolds);
        
        
        
        
        
        
        
        
        // We have a lot of various tools for handling results that have built up over time 
        // and are continuing to be developed. These results analysis tools in particular 
        // are firstly built with our own research output in mind, public usabiltiy second.
        // The apis and functionality are updated over time. 
        
        // Let's load back in all the results files we made: 
        
        ClassifierResultsCollection crc = new ClassifierResultsCollection();
        crc.addClassifiers(classifiers, resultsPath);
        crc.setDatasets(datasets);
        crc.setFolds(numFolds);
        crc.setSplit_Test();
        
        crc.load();
        System.out.println(crc);
        
        // We now basically have a brutally simple primitive array of results, organised as
        // [split][classifier][dataset][fold]
        // Functionality to interact with these is in its infancy, but there are two 
        // main operations:  SLICE and RETRIEVE 
        
        // slice...() will give you a sub-collection of all results of that particular
        // split/classifier/dataset/fold
        
        ClassifierResultsCollection subCrc = crc.sliceDataset("ItalyPowerDemand");
        subCrc = subCrc.sliceClassifier("ED");
        System.out.println(subCrc);
        
        // retrieve...() will get some piece of information, e.g. an eval metric, 
        // from each ClassifierResults object in the collection and return it to
        // you in a parallel array format
        
        double[][][][] subCrcAccs = subCrc.retrieveAccuracies();
        
        // We know there are only the three folds of ItalyPowerDemand in here, let's get those 
        double[] accs = subCrcAccs[0][0][0];
        System.out.println("ED test accs on ItalyPowerDemand: " + Arrays.toString(accs));
        
        
        
        
        
        
        
        
        
        
        // The MultipleClassifierEvaluation (MCE) pipeline is a bit of a beast, and is itself 
        // only a front end-api for ClassifierResultsAnalysis, which is an absolute monster.
        // These especialy will get updated over time when desire/need/motivation for
        // software engineering is found
        
        // Broadly, give it results (we plan to update MCE to take results via a ClassifierResultsCollection, 
        // but at present you load results in a very similar way), give it evaluation metrics 
        // to compare on and settings for things like diagram creation etc., call runComparison()
        // and let it split out a LOT of csv and xls files, and matlab figs + pdf files if diagram
        // creation is turned on
        
        // See MCE.main and examples for more indepth stuff and different options
        
        // Tycally we'd write somewhere else, but for this example we'll write into the resultsPath again.
        MultipleClassifierEvaluation mce = new MultipleClassifierEvaluation(resultsPath, "Ex05_ThoroughEvaluation", numFolds);
        
        mce.setTestResultsOnly(true);         // We didnt also produce e.g. train data estimates
        mce.setBuildMatlabDiagrams(false);    // turning this off for example, see *1 below
        mce.setCleanResults(true);            // deletes the individual predictions once per-object evals are found (acc in this case) to save memory
        mce.setDebugPrinting(true);          
        mce.setUseAccuracyOnly();             // using accuracy only to reduce number of files produced in this example
        mce.setDatasets(datasets);
        
        //general rule of thumb: set/add/read the classifiers as the last thing before running
        mce.readInClassifiers(classifiers, resultsPath); 

        mce.runComparison();  
        
        // A whole bunch of files should now have been spat out. Have a poke around them. 
        // There's little documentation on exactly what each output represents, but most 
        // should be clear from their file names and locations
        // Main file of interest really is the ...ResultsSheet.xls on the top level, which is 
        // a summary of the rest of it. 
        
        // *1. I have MATLAB 2016b. There's nothing particularly bespoke about the things we're doing in 
        // these scripts so all newer versions of MATLAB should run them, and maybe some older. 
        // I don't have a list of versions it runs on, unfortunately. 
        // If you'd like to reimplement these diagrams in python, you're more than welcome to. 
        // Note that despite turning figure creation off, directories and supporting files
        // for the creation of them were still made, to more easily be able to make the figures
        // afterwards.
    }
    
}
